"""Sovereign-Mahjong V6 Core Algorithms: R-NaD + Unbiased PPO + Optimizer Surgery + Self-Healing.

Implements all mathematically verified V6 training primitives:
1. Legal-masked categorical log-probs, entropy, KL divergence, and zero-division-safe means.
2. Running-RMS Advantage Scaling (`update_running_rms`):
   - Strictly active-mask only (`learning_seat_mask & (num_legal >= 2)`).
   - STRICTLY NO mean subtraction (preserves exact `E[A | s] == 0` unbiasedness!).
3. Dual-Anchor Entropy-Floored R-NaD Reference Policy (`rho_{k+1} = 0.95*pi_k + 0.035*pi_bc + 0.015*U_legal`).
4. Triple-Armor Softened Support-Set Safety Shield (`pi_bc(T=1.8)`, `tau_safe=1e-3`, `c_max=5.0`, annealed).
5. Isolated 3-Chain Optax Optimizers (`tx_policy`, `tx_oracle`, `tx_belief`) with exact AdamW weight-decay
   exemption on `bias`, `scale`, `LayerNorm`, `raw_beta_risk`, `raw_beta_offense`.
6. Outer-Reset AdamW Surgical First-Moment Reset (`reset_adam_first_moment_only`: zeros `mu` while
   preserving `nu` bit-for-bit!).
7. Self-Healing Circuit Breakers (`SelfHealingManager`) + Calibration Metrics (ECE, Brier, ROC-AUC, Birkhoff err).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mahjax.common.action import Action
from mahjong_rl.config import BeliefConfig, OracleConfig, RNaDConfig

NEG_INF = -1e9


# =============================================================================
# 1. Legal-Masked Distribution & Utility Primitives
# =============================================================================


def masked_mean(values: jnp.ndarray, mask: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    m = mask.astype(jnp.float32)
    return jnp.sum(jnp.where(m > 0.5, values, 0.0)) / jnp.maximum(jnp.sum(m), eps)


def masked_categorical_log_probs(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute log-probabilities over legal actions (illegal entries set to `NEG_INF`)."""
    mask = legal_mask.astype(jnp.bool_)
    masked = jnp.where(mask, logits.astype(jnp.float32), NEG_INF)
    log_p = jax.nn.log_softmax(masked, axis=-1)
    return jnp.where(mask, log_p, NEG_INF)


def masked_categorical_probs(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute exact normalized probabilities over legal actions (0.0 on illegal)."""
    mask = legal_mask.astype(jnp.bool_)
    masked = jnp.where(mask, logits.astype(jnp.float32), NEG_INF)
    probs = jax.nn.softmax(masked, axis=-1)
    probs = jnp.where(mask, probs, 0.0)
    return probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1e-12)


def masked_entropy(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """Shannon entropy over legal actions `H(pi(.|s))` -> `(...)`."""
    mask = legal_mask.astype(jnp.bool_)
    log_p = masked_categorical_log_probs(logits, mask)
    p = jnp.where(mask, jnp.exp(log_p), 0.0)
    return -jnp.sum(jnp.where(mask, p * log_p, 0.0), axis=-1)


def masked_kl_divergence(
    p_probs: jnp.ndarray,
    log_p: jnp.ndarray,
    log_q: jnp.ndarray,
    legal_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Exact forward KL `KL(p || q) = sum_{a in legal} p(a) * (log p(a) - log q(a))`."""
    mask = legal_mask.astype(jnp.bool_)
    safe_diff = jnp.where(mask, p_probs * (log_p - log_q), 0.0)
    return jnp.maximum(jnp.sum(safe_diff, axis=-1), 0.0)


def compute_dual_anchor_rho_probs(
    pi_k_logits: jnp.ndarray,
    pi_bc_logits: jnp.ndarray,
    legal_mask: jnp.ndarray,
    mix_bc: float = 0.035,
    mix_uniform: float = 0.015,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Dual-Anchor Entropy-Floored R-NaD reference policy `rho_{k+1}`:

        rho_{k+1} = (1 - mix_bc - mix_uniform) * pi_k + mix_bc * pi_bc + mix_uniform * U_legal

    Returns `(rho_probs, log_rho)` with illegal actions masked to `0.0` / `NEG_INF`.
    """
    mask = legal_mask.astype(jnp.bool_)
    p_k = masked_categorical_probs(pi_k_logits, mask)
    p_bc = masked_categorical_probs(pi_bc_logits, mask)
    n_legal = jnp.maximum(jnp.sum(mask.astype(jnp.float32), axis=-1, keepdims=True), 1.0)
    p_u = jnp.where(mask, 1.0 / n_legal, 0.0)

    w_k = max(0.0, 1.0 - float(mix_bc) - float(mix_uniform))
    rho = w_k * p_k + float(mix_bc) * p_bc + float(mix_uniform) * p_u
    rho = jnp.where(mask, jnp.maximum(rho, 1e-12), 0.0)
    rho = rho / jnp.maximum(jnp.sum(rho, axis=-1, keepdims=True), 1e-12)
    log_rho = jnp.where(mask, jnp.log(jnp.maximum(rho, 1e-12)), NEG_INF)
    return rho, log_rho


# =============================================================================
# 2. Running-RMS Advantage Scaling (Active-Mask Only, ZERO Mean Subtraction!)
# =============================================================================


class RunningRMSState(NamedTuple):
    """Unbiased running second-moment tracker for advantage scaling."""

    second_moment: jnp.ndarray  # () float32 EMA of E[A^2]
    steps: jnp.ndarray          # () int32


def init_running_rms(init_val: float = 1.0) -> RunningRMSState:
    return RunningRMSState(
        second_moment=jnp.array(init_val, dtype=jnp.float32),
        steps=jnp.array(0, dtype=jnp.int32),
    )


def update_running_rms(
    state: RunningRMSState,
    raw_advantages: jnp.ndarray,
    active_mask: jnp.ndarray,
    beta: float = 0.99,
    eps: float = 1e-5,
) -> Tuple[RunningRMSState, jnp.ndarray, jnp.ndarray]:
    """Scale raw advantages by running RMS strictly WITHOUT subtracting the batch mean!

    Preserves exact zero expectation (`E[A_scaled | s] == 0`) so R-NaD's fixed point is
    never biased by batch composition, while eliminating forced-move (`num_legal == 1`) dilution.
    """
    m = active_mask.astype(jnp.float32)
    count = jnp.sum(m)
    has_data = count > 0.5
    batch_sq = jnp.sum(jnp.where(m > 0.5, raw_advantages ** 2, 0.0)) / jnp.maximum(count, 1.0)

    # On first active update, seed directly from batch_sq to avoid cold-start lag
    is_first = state.steps == 0
    new_sq = jnp.where(
        has_data,
        jnp.where(is_first, batch_sq, float(beta) * state.second_moment + (1.0 - float(beta)) * batch_sq),
        state.second_moment,
    )
    new_steps = jnp.where(has_data, state.steps + 1, state.steps)
    rms = jnp.sqrt(jnp.maximum(new_sq, 0.0) + float(eps))
    # Strictly divide ONLY by rms (NEVER subtract mean!)
    scaled_adv = jnp.where(m > 0.5, raw_advantages / rms, 0.0)
    return RunningRMSState(second_moment=new_sq, steps=new_steps), scaled_adv, rms


# =============================================================================
# 3. Policy, Oracle, and Belief Losses (Sovereign-Mahjong V6)
# =============================================================================


def compute_shield_lambda(progress: float, cfg: RNaDConfig) -> float:
    """Smooth exponential annealing of Triple-Armor safety shield coefficient."""
    p = float(np.clip(progress, 0.0, 1.0))
    return float(
        cfg.shield_lambda_end
        + (cfg.shield_lambda_start - cfg.shield_lambda_end) * np.exp(-p / max(cfg.shield_decay_scale, 1e-4))
    )


def compute_policy_loss_v6(
    logits: jnp.ndarray,              # (N, 87) current policy logits
    actions: jnp.ndarray,             # (N,) int32
    old_log_probs: jnp.ndarray,       # (N,) float32
    advantages_scaled: jnp.ndarray,   # (N,) float32 (Running-RMS scaled, NO mean sub!)
    log_rho: jnp.ndarray,             # (N, 87) dual-anchor R-NaD reference log-probs
    bc_logits: jnp.ndarray,           # (N, 87) frozen BC reference logits
    legal_mask: jnp.ndarray,          # (N, 87) bool
    learning_seat_mask: jnp.ndarray,  # (N,) bool (excludes PFSP frozen opponent seats)
    cfg: RNaDConfig,
    shield_lambda: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute Clean Unbiased Asymmetric PPO + R-NaD KL Magnet + Triple-Armor Safety Shield.

    Strictly masks out BOTH non-learning seats AND forced-move steps (`num_legal <= 1`).
    """
    mask = legal_mask.astype(jnp.bool_)
    num_legal = jnp.sum(mask.astype(jnp.int32), axis=-1)
    active_mask = learning_seat_mask.astype(jnp.bool_) & (num_legal >= 2)

    log_pi = masked_categorical_log_probs(logits, mask)  # (N, 87)
    pi_probs = jnp.where(mask, jnp.exp(log_pi), 0.0)

    # Gather log pi(a_t | s_t)
    idx = jnp.arange(actions.shape[0], dtype=jnp.int32)
    act_log_p = log_pi[idx, actions.astype(jnp.int32)]

    # 1. PPO-Clip objective
    log_ratio = jnp.clip(act_log_p - old_log_probs, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    adv = jax.lax.stop_gradient(advantages_scaled)
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps)) * adv
    ppo_loss = -masked_mean(jnp.minimum(surr1, surr2), active_mask)

    # 2. R-NaD exact forward KL magnet: alpha_rnad * KL(pi || rho_k)
    kl_rnad = masked_kl_divergence(pi_probs, log_pi, log_rho, mask)
    rnad_loss = float(cfg.alpha_rnad) * masked_mean(kl_rnad, active_mask)

    # 3. Triple-Armor Softened Support-Set Safety Shield
    #    pi_bc_soft = softmax(bc_logits / T_soft), T_soft = 1.8
    #    penalizes ONLY out-of-support mass where pi_bc_soft < tau_safe, capped at c_max
    p_bc_soft = masked_categorical_probs(bc_logits / float(cfg.bc_soft_temp), mask)
    out_of_support = mask & (p_bc_soft < float(cfg.shield_tau_safe))
    penalty_weight = jnp.minimum(
        float(cfg.shield_c_max),
        jnp.log(float(cfg.shield_tau_safe) / jnp.maximum(p_bc_soft, 1e-8)),
    )
    shield_per_step = jnp.sum(jnp.where(out_of_support, pi_probs * penalty_weight, 0.0), axis=-1)
    shield_loss = jnp.asarray(shield_lambda, dtype=jnp.float32) * masked_mean(shield_per_step, active_mask)

    entropy = masked_mean(masked_entropy(logits, mask), active_mask)
    ent_bonus = -float(cfg.entropy_coef) * entropy
    total_loss = ppo_loss + rnad_loss + shield_loss + ent_bonus

    # Diagnostics (strictly over active_mask)
    approx_kl = masked_mean((ratio - 1.0) - log_ratio, active_mask)
    clip_frac = masked_mean((jnp.abs(ratio - 1.0) > float(cfg.clip_eps)).astype(jnp.float32), active_mask)

    # Behavioral anchor & Riichi/Call tendency probes (zero extra forward passes)
    log_bc = masked_categorical_log_probs(bc_logits, mask)
    bc_kl = masked_mean(masked_kl_divergence(pi_probs, log_pi, log_bc, mask), active_mask)
    riichi_legal_m = active_mask & mask[:, Action.RIICHI]
    p_riichi_sum = jnp.sum(jnp.where(riichi_legal_m, pi_probs[:, Action.RIICHI], 0.0))
    n_riichi_legal = jnp.sum(riichi_legal_m.astype(jnp.float32))
    p_riichi_legal = p_riichi_sum / jnp.maximum(n_riichi_legal, 1e-8)

    call_mask_actions = mask[:, Action.PON : Action.CHI_R_RED + 1]
    call_legal_m = active_mask & jnp.any(call_mask_actions, axis=-1)
    p_call_row = jnp.sum(jnp.where(call_mask_actions, pi_probs[:, Action.PON : Action.CHI_R_RED + 1], 0.0), axis=-1)
    p_call_sum = jnp.sum(jnp.where(call_legal_m, p_call_row, 0.0))
    n_call_legal = jnp.sum(call_legal_m.astype(jnp.float32))
    p_call_legal = p_call_sum / jnp.maximum(n_call_legal, 1e-8)

    metrics = {
        "policy_loss": total_loss,
        "ppo_loss": ppo_loss,
        "rnad_kl": masked_mean(kl_rnad, active_mask),
        "bc_kl": bc_kl,
        "shield_loss": shield_loss,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "policy_entropy": entropy,
        "active_frac": jnp.mean(active_mask.astype(jnp.float32)),
        "p_riichi_legal": p_riichi_legal,
        "p_riichi_sum": p_riichi_sum,
        "n_riichi_legal": n_riichi_legal,
        "p_call_legal": p_call_legal,
        "p_call_sum": p_call_sum,
        "n_call_legal": n_call_legal,
    }
    return total_loss, metrics


def compute_oracle_loss_v6(
    oracle_out: Dict[str, jnp.ndarray],
    targets_4x4: jnp.ndarray,              # (N, 4, 4) Birkhoff TD targets
    round_winner_ego: Optional[jnp.ndarray] = None,   # (N,) int32 in 0..4
    round_score_delta_ego: Optional[jnp.ndarray] = None,  # (N, 4) float32
    aux_valid: Optional[jnp.ndarray] = None,          # (N,) bool
    cfg: OracleConfig = OracleConfig(),
    delta_u_target_ego: Optional[jnp.ndarray] = None,     # (N, 4) float32
    valid_mask: Optional[jnp.ndarray] = None,             # (N,) bool
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """4-seat Birkhoff placement CE + direct ego-weighted delta_u + Dual-Horizon auxiliary supervision."""
    probs = jnp.clip(oracle_out["probs_4x4"], 1e-7, 1.0)
    # Cross-entropy averaged across all 4 seats (and valid batch samples)
    ce_per_seat = -jnp.sum(targets_4x4 * jnp.log(probs), axis=-1)  # (N, 4)
    if valid_mask is not None:
        main_loss = masked_mean(jnp.mean(ce_per_seat, axis=-1), valid_mask)
    else:
        main_loss = jnp.mean(ce_per_seat)

    if delta_u_target_ego is not None and "delta_u_ego" in oracle_out:
        ego_w = float(cfg.ego_seat_loss_weight)
        seat_w_raw = jnp.array([ego_w, 1.0, 1.0, 1.0], dtype=jnp.float32)
        seat_w = seat_w_raw / jnp.mean(seat_w_raw)
        du_err = oracle_out["delta_u_ego"] - delta_u_target_ego.astype(jnp.float32)
        du_huber = jnp.mean(seat_w[None, :] * optax.huber_loss(du_err, delta=0.25), axis=-1)
        delta_u_loss = masked_mean(du_huber, valid_mask) if valid_mask is not None else jnp.mean(du_huber)
    else:
        delta_u_loss = jnp.array(0.0, dtype=jnp.float32)

    # Dual-horizon auxiliary losses (if targets provided)
    if round_winner_ego is not None and aux_valid is not None:
        win_log_p = jax.nn.log_softmax(oracle_out["round_winner_logits"], axis=-1)
        idx = jnp.arange(round_winner_ego.shape[0], dtype=jnp.int32)
        win_ce = -win_log_p[idx, jnp.clip(round_winner_ego.astype(jnp.int32), 0, 4)]
        win_loss = masked_mean(win_ce, aux_valid)
    else:
        win_loss = jnp.array(0.0, dtype=jnp.float32)

    if round_score_delta_ego is not None and aux_valid is not None:
        diff = oracle_out["round_score_delta"] - round_score_delta_ego.astype(jnp.float32)
        huber = jnp.mean(optax.huber_loss(diff, delta=1.0), axis=-1)
        score_loss = masked_mean(huber, aux_valid)
    else:
        score_loss = jnp.array(0.0, dtype=jnp.float32)

    total = (
        main_loss
        + float(cfg.delta_u_loss_coef) * delta_u_loss
        + float(cfg.round_win_loss_coef) * win_loss
        + float(cfg.round_score_loss_coef) * score_loss
    )
    birkhoff_err = compute_birkhoff_error(oracle_out["probs_4x4"])
    metrics = {
        "oracle_loss": total,
        "oracle_rank_ce": main_loss,
        "oracle_delta_u_loss": delta_u_loss,
        "oracle_win_ce": win_loss,
        "oracle_score_huber": score_loss,
        "oracle_birkhoff_err": birkhoff_err,
    }
    return total, metrics


def compute_belief_loss_v6(
    belief_out: Dict[str, jnp.ndarray],
    y_ron: jnp.ndarray,            # (N, 3, 34) float32 in {0, 1}
    y_score_bin: jnp.ndarray,      # (N, 3, 34) int32 in {0, 1, 2}
    y_call_threat: jnp.ndarray,    # (N, 3, 34) float32 in {0, 1}
    y_tenpai: jnp.ndarray,         # (N, 3) float32 in {0, 1}
    cfg: BeliefConfig = BeliefConfig(),
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Strictly UNWEIGHTED BCE for `p_deal` (guarantees calibrated Bayes posterior) + aux heads."""
    # 1. STRICTLY unweighted standard BCE on deal_logits (zero positive-class reweighting!)
    deal_bce = optax.sigmoid_binary_cross_entropy(belief_out["deal_logits"], y_ron.astype(jnp.float32))
    loss_deal = jnp.mean(deal_bce)

    # 2. Conditional 3-bin score CE strictly on positive Ron slots (`y_ron > 0.5`)
    pos_mask = y_ron > 0.5
    sb_log_p = jax.nn.log_softmax(belief_out["score_bin_logits"], axis=-1)  # (N, 3, 34, 3)
    sb_oh = jax.nn.one_hot(jnp.clip(y_score_bin.astype(jnp.int32), 0, 2), 3, dtype=jnp.float32)
    sb_ce = -jnp.sum(sb_oh * sb_log_p, axis=-1)  # (N, 3, 34)
    loss_score = masked_mean(sb_ce, pos_mask)

    # 3. High-value meld-feed (call) threat BCE
    loss_call = jnp.mean(
        optax.sigmoid_binary_cross_entropy(belief_out["call_threat_logits"], y_call_threat.astype(jnp.float32))
    )

    # 4. Opponent tenpai BCE
    loss_tenpai = jnp.mean(
        optax.sigmoid_binary_cross_entropy(belief_out["tenpai_logits"], y_tenpai.astype(jnp.float32))
    )

    total = (
        loss_deal
        + float(cfg.score_bin_loss_coef) * loss_score
        + float(cfg.call_threat_loss_coef) * loss_call
        + float(cfg.tenpai_loss_coef) * loss_tenpai
    )
    metrics = {
        "belief_loss": total,
        "belief_deal_bce": loss_deal,
        "belief_score_ce": loss_score,
        "belief_call_bce": loss_call,
        "belief_tenpai_bce": loss_tenpai,
        "belief_pos_rate": jnp.mean(y_ron.astype(jnp.float32)),
        "belief_pred_mean": jnp.mean(belief_out["p_deal"]),
    }
    return total, metrics


# =============================================================================
# 4. Isolated 3-Chain Optimizers, WD Exemption, EMA, and Surgical m_t Reset
# =============================================================================


def make_weight_decay_mask(params: Any) -> Any:
    """Return a boolean PyTree: False (NO weight decay) for bias, LayerNorm, scale, and beta gates."""
    def _should_decay(path: Tuple[Any, ...], leaf: jnp.ndarray) -> bool:
        names = [str(getattr(k, "key", k)).lower() for k in path]
        leaf_name = names[-1] if names else ""
        if leaf.ndim <= 1:
            return False
        if leaf_name in ("bias", "scale", "raw_beta_risk", "raw_beta_offense"):
            return False
        if any("layernorm" in n or "layer_norm" in n or "beta" in n for n in names):
            return False
        return True

    return jax.tree_util.tree_map_with_path(_should_decay, params)


def _build_adamw_chain(lr, weight_decay: float, grad_clip: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(
            learning_rate=lr,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=weight_decay,
            mask=make_weight_decay_mask,
        ),
    )


def build_isolated_optimizers(
    rnad_cfg: RNaDConfig,
    oracle_cfg: OracleConfig,
    belief_cfg: BeliefConfig,
) -> Tuple[optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation]:
    """Construct 3 completely isolated Optax chains (`tx_policy`, `tx_oracle`, `tx_belief`).

    Also supports dynamic LR scaling on `tx_policy` via `optax.inject_hyperparams` so
    outer-reset 30-step warmup and Self-Healing LR halving work with ZERO XLA recompilation!
    """
    wd_pi = float(rnad_cfg.policy_weight_decay)
    gc_pi = float(rnad_cfg.policy_grad_clip)

    def _policy_factory(lr):
        return _build_adamw_chain(lr=lr, weight_decay=wd_pi, grad_clip=gc_pi)

    tx_policy = optax.inject_hyperparams(_policy_factory)(lr=float(rnad_cfg.policy_lr))
    tx_oracle = _build_adamw_chain(
        lr=float(oracle_cfg.lr),
        weight_decay=float(oracle_cfg.weight_decay),
        grad_clip=float(oracle_cfg.grad_clip_norm),
    )
    tx_belief = _build_adamw_chain(
        lr=float(belief_cfg.lr),
        weight_decay=float(belief_cfg.weight_decay),
        grad_clip=float(belief_cfg.grad_clip_norm),
    )
    return tx_policy, tx_oracle, tx_belief


def reset_adam_first_moment_only(opt_state: Any) -> Any:
    """R-NaD Outer-Reset Surgical Optimizer Reset:

    Recursively finds every `optax.ScaleByAdamState` inside `opt_state` and:
    - ZEROS first moment `mu` (`m_t = 0`), flushing stale momentum from epoch `k`.
    - PRESERVES second moment `nu` (`v_t`) and `count` BIT-FOR-BIT, preventing
      step-size explosions at outer epoch transitions!
    """
    def _transform(node: Any) -> Any:
        if isinstance(node, optax.ScaleByAdamState):
            zero_mu = jax.tree_util.tree_map(jnp.zeros_like, node.mu)
            return optax.ScaleByAdamState(count=node.count, mu=zero_mu, nu=node.nu)
        if isinstance(node, tuple) and hasattr(node, "_fields"):
            return type(node)(*(_transform(v) for v in node))
        if isinstance(node, tuple):
            return tuple(_transform(v) for v in node)
        if isinstance(node, list):
            return [_transform(v) for v in node]
        if isinstance(node, dict):
            return {k: _transform(v) for k, v in node.items()}
        return node

    return _transform(opt_state)


def set_injected_lr(opt_state: Any, new_lr: float, n_devices: int = 1) -> Any:
    """Update `lr` inside `optax.InjectStatefulHyperparamsState` with zero XLA recompile or moment copy."""
    if hasattr(opt_state, "hyperparams") and "lr" in opt_state.hyperparams:
        new_hp = dict(opt_state.hyperparams)
        if int(n_devices) > 1:
            lr_arr = np.full((int(n_devices),), float(new_lr), dtype=np.float32)
            new_hp["lr"] = shard_for_pmap(lr_arr, int(n_devices))
        else:
            new_hp["lr"] = jnp.asarray(new_lr, dtype=jnp.float32)
        return opt_state._replace(hyperparams=new_hp)
    return opt_state


def ema_update_params(ema_params: Any, fast_params: Any, tau: float) -> Any:
    """Exponential moving average update: `ema <- (1 - tau) * ema + tau * fast`."""
    t = float(tau)
    return jax.tree_util.tree_map(lambda e, f: (1.0 - t) * e + t * f, ema_params, fast_params)


def _pmap_sharding(n_devices: int) -> jax.sharding.NamedSharding:
    devices = jax.local_devices()[:n_devices]
    return jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(devices), ("d",)),
        jax.sharding.PartitionSpec("d"),
    )


def replicate_for_pmap(tree: Any, n_devices: int) -> Any:
    """Replicate a PyTree onto `n_devices` with explicit leading-axis `NamedSharding` (JAX 0.4–0.12 safe)."""
    sharding = _pmap_sharding(n_devices)

    def _put(x):
        a = np.asarray(jax.device_get(x))
        b = np.broadcast_to(a, (n_devices,) + a.shape).copy()
        return jax.device_put(b, sharding)

    return jax.tree_util.tree_map(_put, tree)


def shard_for_pmap(tree: Any, n_devices: int) -> Any:
    """Place a leading-axis `(n_devices, ...)` PyTree across `n_devices` with explicit `NamedSharding`."""
    sharding = _pmap_sharding(n_devices)
    return jax.tree_util.tree_map(lambda x: jax.device_put(np.asarray(jax.device_get(x)), sharding), tree)


# =============================================================================
# 5. Calibration & Structural Health Metrics
# =============================================================================


def compute_birkhoff_error(probs_4x4: jnp.ndarray) -> jnp.ndarray:
    """Mean (across batch) of max row/column sum deviation from 1.0 over `(..., 4, 4)`."""
    row_err = jnp.max(jnp.abs(jnp.sum(probs_4x4, axis=-1) - 1.0), axis=-1)
    col_err = jnp.max(jnp.abs(jnp.sum(probs_4x4, axis=-2) - 1.0), axis=-1)
    return jnp.mean(jnp.maximum(row_err, col_err))


def compute_brier_4rank(probs_4x4: np.ndarray, true_rank_4x4: np.ndarray) -> float:
    """Multi-class Brier score over 4-seat placement probability matrices."""
    p = np.asarray(probs_4x4, dtype=np.float64)
    y = np.asarray(true_rank_4x4, dtype=np.float64)
    return float(np.mean(np.sum((p - y) ** 2, axis=-1)))


def compute_ece_4rank(probs_4x4: np.ndarray, true_rank_4x4: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over predicted 4-rank probabilities."""
    p = np.asarray(probs_4x4, dtype=np.float64).reshape(-1)
    y = np.asarray(true_rank_4x4, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = max(len(p), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        cnt = int(np.sum(in_bin))
        if cnt > 0:
            ece += (cnt / total) * abs(float(np.mean(p[in_bin])) - float(np.mean(y[in_bin])))
    return float(ece)


def compute_binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact rank-based binary ROC-AUC (zero sklearn dependency; returns 0.5 if single-class)."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1) > 0.5
    ys = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(yt))
    n_neg = int(len(yt) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(ys, kind="mergesort")
    ys_sorted = ys[order]
    yt_sorted = yt[order]
    # Average ranks for ties
    ranks = np.arange(1, len(ys) + 1, dtype=np.float64)
    # Handle exact ties cleanly
    diff_mask = np.concatenate([[True], ys_sorted[1:] != ys_sorted[:-1], [True]])
    change_idx = np.flatnonzero(diff_mask)
    for start, end in zip(change_idx[:-1], change_idx[1:]):
        if end - start > 1:
            ranks[start:end] = 0.5 * (start + 1 + end)
    sum_pos_ranks = float(np.sum(ranks[yt_sorted]))
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_belief_temperature(deal_logits: np.ndarray, y_ron: np.ndarray) -> float:
    """Find optimal post-training calibration temperature `T_cal in [0.5, 2.5]` minimizing NLL."""
    logits = np.asarray(deal_logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(y_ron, dtype=np.float64).reshape(-1)
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 2.5, 81):
        z = logits / t
        # Numerically stable binary cross-entropy: max(z,0) - z*y + log(1 + exp(-|z|))
        nll = float(np.mean(np.maximum(z, 0.0) - z * labels + np.log1p(np.exp(-np.abs(z)))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


# =============================================================================
# 6. Self-Healing Circuit Breaker Manager (6 Probes + Rollback + LR Halving)
# =============================================================================


@dataclass
class HealthReport:
    healthy: bool
    reasons: Tuple[str, ...] = ()
    action_taken: str = "none"  # "none" | "rollback_and_halve_lr" | "abort"


@dataclass
class SelfHealingManager:
    """Monitors the V6 training health probes and executes automatic rollback + LR halving."""

    cfg: RNaDConfig
    max_snapshots: int = 3
    snapshots: Deque[Dict[str, Any]] = field(init=False)
    riichi_window: Deque[Tuple[float, float]] = field(init=False)
    consecutive_heals: int = 0
    current_lr_scale: float = 1.0

    def __post_init__(self) -> None:
        self.snapshots = deque(maxlen=max(1, int(self.max_snapshots)))
        self.riichi_window = deque(maxlen=max(1, int(self.cfg.riichi_probe_window)))

    def save_checkpoint(self, step: int, payload: Dict[str, Any]) -> None:
        self.snapshots.append({"step": step, "payload": jax.device_get(payload)})
        self.consecutive_heals = 0
        self.riichi_window.clear()

    def evaluate_probes(
        self,
        metrics: Dict[str, float],
        finite_grads: bool = True,
        distinct_deck_ratio: float = 1.0,
    ) -> HealthReport:
        reasons = []
        if not finite_grads or any(not np.isfinite(float(v)) for v in metrics.values()):
            reasons.append("non_finite_grads_or_metrics")
        if float(metrics.get("approx_kl", 0.0)) > float(self.cfg.max_approx_kl):
            reasons.append(f"approx_kl={metrics.get('approx_kl'):.4f}>{self.cfg.max_approx_kl}")
        if float(metrics.get("policy_entropy", 1.0)) < float(self.cfg.min_policy_entropy):
            reasons.append(f"entropy={metrics.get('policy_entropy'):.4f}<{self.cfg.min_policy_entropy}")
        if float(metrics.get("oracle_birkhoff_err", 0.0)) > float(self.cfg.max_birkhoff_err):
            reasons.append(f"birkhoff_err={metrics.get('oracle_birkhoff_err'):.5f}>{self.cfg.max_birkhoff_err}")
        if float(distinct_deck_ratio) < float(self.cfg.min_distinct_deck_ratio):
            reasons.append(f"distinct_deck_ratio={distinct_deck_ratio:.3f}<{self.cfg.min_distinct_deck_ratio}")
        if "p_riichi_legal" in metrics and float(metrics.get("n_riichi_legal", 0.0)) > 0.0:
            n_r = float(metrics.get("n_riichi_legal", 0.0))
            p_r = float(metrics["p_riichi_legal"])
            self.riichi_window.append((p_r * n_r, n_r))
            win_n = sum(x[1] for x in self.riichi_window)
            win_p = sum(x[0] for x in self.riichi_window) / max(win_n, 1e-8)
            if (
                win_n >= float(self.cfg.riichi_probe_min_samples)
                and win_p > float(self.cfg.max_p_riichi_legal)
                and p_r > float(self.cfg.max_p_riichi_legal)
            ):
                reasons.append(
                    f"p_riichi_legal_win={win_p:.3f}(curr={p_r:.3f},n={int(win_n)})>{self.cfg.max_p_riichi_legal}"
                )

        if not reasons:
            self.consecutive_heals = 0
            return HealthReport(healthy=True)

        self.consecutive_heals += 1
        if self.consecutive_heals > int(self.cfg.max_consecutive_heals) or len(self.snapshots) == 0:
            return HealthReport(healthy=False, reasons=tuple(reasons), action_taken="abort")

        self.current_lr_scale *= 0.5
        return HealthReport(healthy=False, reasons=tuple(reasons), action_taken="rollback_and_halve_lr")

    def is_snapshot_safe(self, metrics: Dict[str, float]) -> bool:
        """Return False when policy already drifted toward KL/entropy/Riichi boundaries."""
        if float(metrics.get("approx_kl", 0.0)) > 0.5 * float(self.cfg.max_approx_kl):
            return False
        if float(metrics.get("policy_entropy", 1.0)) < float(self.cfg.min_policy_entropy) + 0.05:
            return False
        if "p_riichi_legal" in metrics and float(metrics.get("n_riichi_legal", 0.0)) > 0.0:
            p_r = float(metrics["p_riichi_legal"])
            if p_r > float(self.cfg.safe_snapshot_max_p_riichi):
                return False
            if self.riichi_window:
                win_n = sum(x[1] for x in self.riichi_window)
                win_p = sum(x[0] for x in self.riichi_window) / max(win_n, 1e-8)
                if win_p > float(self.cfg.safe_snapshot_max_p_riichi):
                    return False
        return True

    def restore_latest(self) -> Optional[Dict[str, Any]]:
        if not self.snapshots:
            return None
        self.riichi_window.clear()
        snap = self.snapshots[-1]
        return jax.device_put(snap["payload"])

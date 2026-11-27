"""Phase 1 (Frozen-Policy Warmup) & Phase 2 (Full Sovereign-Mahjong V6 PureJaxRL Trainer).

Implements all verified Sovereign-Mahjong V6 architectural & systems invariants:
- Per-Rollout Single-Snapshot PFSP League (ZERO 8x vmap blowup, ZERO XLA recompilation!)
- Tri-Regime Meta-Game Table Pool (50% Homogeneous / 35% PFSP / 15% Cross-Dan Heterogeneous)
- DAG Fire-Wall: frozen `GRP_base` for all input-side analytical priors + quasi-static `B_EMA`
- Birkhoff Doubly-Stochastic Dual-Rate Simplex TD(lambda) (`lambda_intra=0.975, lambda_inter=0.70`)
- Running-RMS Advantage Scaling (active-mask only, STRICTLY NO mean subtraction!)
- Dual-Anchor Entropy-Floored R-NaD Outer Reset (`m_t = 0`, `v_t` preserved + 30-step LR warmup)
- Triple-Armor Softened Support-Set Safety Shield (`pi_bc(T=1.8)`, `tau_safe=1e-3`, `c_max=5.0`)
- 3 Isolated Optax Optimizer Chains (`tx_policy`, `tx_oracle`, `tx_belief`)
- Self-Healing Circuit Breakers (6 probes + rollback + LR halving)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import mahjax
from mahjax.common.action import Action
from mahjong_rl.algo.gae import (
    calculate_dual_rate_simplex_td_lambda,
    compute_round_outcome_targets,
    compute_terminal_rank_matrix_4x4,
)
from mahjong_rl.algo.legal_ron import (
    compute_analytical_trinity,
    compute_sparse_belief_labels,
    extract_ron_snapshot,
    extract_table_snapshot,
)
from mahjong_rl.algo.sovereign_v6 import (
    NEG_INF,
    RunningRMSState,
    SelfHealingManager,
    build_isolated_optimizers,
    compute_belief_loss_v6,
    compute_binary_roc_auc,
    compute_dual_anchor_rho_probs,
    compute_oracle_loss_v6,
    compute_policy_loss_v6,
    compute_shield_lambda,
    ema_update_params,
    init_running_rms,
    masked_categorical_log_probs,
    masked_categorical_probs,
    masked_mean,
    replicate_for_pmap,
    reset_adam_first_moment_only,
    set_injected_lr,
    shard_for_pmap,
    update_running_rms,
)
from mahjong_rl.algo.utility import (
    build_opponent_utility_features,
    rotate_utility_matrix_ego,
    sample_meta_table_pool,
)
from mahjong_rl.config import RLConfig
from mahjong_rl.features.obs_2d import clean_auto_reset_state
from mahjong_rl.features.obs_v6 import (
    GLOBAL_SCALAR_DIM_V6,
    NUM_OBS_V6_PLANES,
    encode_obs_v6,
)
from mahjong_rl.features.oracle import NUM_ORACLE_PLANES, encode_oracle_planes
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRP_FEATURE_DIM, GRPNet, encode_grp_features
from mahjong_rl.models.oracle import OracleCriticNet
from mahjong_rl.trainers.train_bc import init_policy_params
from mahjong_rl.trainers.train_belief import _init_belief_params


@dataclass
class V6RLResult:
    policy_model: SovereignPolicyNet
    policy_params: Any
    belief_model: BeliefNet
    belief_ema_params: Any
    belief_fast_params: Any
    belief_t_cal: float
    grp_model: GRPNet
    grp_base_params: Any
    oracle_model: OracleCriticNet
    oracle_params: Any
    outer_snapshots: List[Any]
    history: List[Dict[str, float]]


def _init_oracle_params(
    oracle_model: OracleCriticNet,
    belief_river_dim: int,
    rng: jax.Array,
) -> Any:
    return oracle_model.init(
        rng,
        planes=jnp.zeros((2, NUM_OBS_V6_PLANES, 34), dtype=jnp.float32),
        oracle_planes=jnp.zeros((2, NUM_ORACLE_PLANES, 34), dtype=jnp.float32),
        scalars=jnp.zeros((2, GLOBAL_SCALAR_DIM_V6), dtype=jnp.float32),
        w_table_ego=jnp.zeros((2, 4, 4), dtype=jnp.float32),
        grp_logits_abs=jnp.zeros((2, 4, 4), dtype=jnp.float32),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        river_spatial_feat=jnp.zeros((2, 34, 4 * belief_river_dim), dtype=jnp.float32),
    )


def _tree_is_finite(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.bool_(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def _clone_params(params: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)


def _clone_host_params(params: Any) -> Any:
    """Copy a PyTree onto host NumPy arrays (zero GPU VRAM footprint)."""
    return jax.tree_util.tree_map(lambda x: np.array(jax.device_get(x), copy=True), params)


def _validate_resume_shapes(name: str, expected: Any, loaded: Any, resume_path: Any) -> None:
    exp_shapes = [tuple(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(expected)]
    got_shapes = [tuple(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(loaded)]
    if exp_shapes != got_shapes:
        raise ValueError(
            f"Incompatible {name} checkpoint shapes in {resume_path} "
            f"(expected {len(exp_shapes)} leaves e.g. {exp_shapes[:2]}, got {len(got_shapes)} e.g. {got_shapes[:2]}). "
            "Remove stale rl_resume_state.pkl or set FORCE_RETRAIN_RL=True to start cleanly."
        )


def _atomic_save_pickle(path: Path | str, payload: Any) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(payload), f)
    os.replace(tmp, dst)
    return dst


def _desync_hanchan_envs(
    env: Any,
    env_state: Any,
    rng_key: jax.Array,
    num_envs: int,
    round_mode: str,
) -> Any:
    """Stagger parallel hanchan environments across East-1..South-4 so every rollout covers all rounds."""
    if round_mode not in ("east", "half") or int(num_envs) < 4:
        return env_state
    max_stagger = 480 if round_mode == "half" else 240
    targets = jnp.asarray([int((b * max_stagger) // int(num_envs)) for b in range(int(num_envs))], dtype=jnp.int32)

    def _advance_batch(st_b, step_idx, k_sub):
        keys = jax.random.split(k_sub, int(num_envs))
        active = step_idx < targets

        def _one(st, is_act, k):
            ka, ks, kr = jax.random.split(k, 3)
            clean = clean_auto_reset_state(st)
            act = mahjax.rule_based_player(clean, ka)
            raw_next = env.step(clean, act, ks)
            done = raw_next.terminated | raw_next.truncated
            next_st = jax.lax.cond(done, lambda: env.init(kr), lambda: raw_next)
            return jax.tree_util.tree_map(lambda o, n: jnp.where(is_act, n, o), st, next_st)

        return jax.vmap(_one)(st_b, active, keys)

    @jax.jit
    def _run(init_st_b, k0):
        def scan_fn(carry, step_idx):
            st_b, r = carry
            r, k_sub = jax.random.split(r)
            return (_advance_batch(st_b, step_idx, k_sub), r), None

        (final_st, _), _ = jax.lax.scan(scan_fn, (init_st_b, k0), jnp.arange(max_stagger, dtype=jnp.int32))
        return final_st

    return _run(env_state, rng_key)


def train_rl_v6(
    cfg: RLConfig,
    *,
    policy_params_init: Optional[Any] = None,
    belief_params_init: Optional[Any] = None,
    belief_t_cal: float = 1.0,
    grp_params_init: Optional[Any] = None,
    enable_self_healing: bool = True,
    multi_gpu: bool = True,
    verbose: bool = True,
    log_every: int = 5,
    resume_path: Optional[Path | str] = None,
    checkpoint_path: Optional[Path | str] = None,
    checkpoint_every: int = 50,
    on_checkpoint: Optional[Callable[[int, Path], None]] = None,
) -> V6RLResult:
    """Run Phase 1 (Frozen-Policy Warmup) + Phase 2 (Sovereign-Mahjong V6 RL).

    Supports seamless multi-session continuous online self-play resume via
    `resume_path` + periodic atomic `checkpoint_path` (preserving all 3 AdamW
    optimizer states, PFSP pool, outer R-NaD snapshots, Running-RMS, and BC anchor).
    """
    n_devices = jax.local_device_count() if multi_gpu else 1
    use_pmap = (
        n_devices > 1
        and (cfg.num_envs % n_devices == 0)
        and (cfg.minibatch_size % n_devices == 0)
    )
    envs_per_dev = cfg.num_envs // n_devices if use_pmap else cfg.num_envs

    def _rep(tree: Any) -> Any:
        return replicate_for_pmap(tree, n_devices) if use_pmap else tree

    def _unrep(tree: Any) -> Any:
        return jax.tree_util.tree_map(lambda x: x[0], tree) if use_pmap else tree

    env = mahjax.make(
        cfg.env_name,
        round_mode=cfg.round_mode,
        observe_type="dict",
        round_end="auto",
    )

    policy_net = SovereignPolicyNet(config=cfg.model)
    belief_net = BeliefNet(config=cfg.belief)
    grp_net = GRPNet(config=cfg.grp)
    oracle_net = OracleCriticNet(
        config=cfg.oracle,
        tau_prior=cfg.oracle.tau_prior,
        river_dim=cfg.belief.river_dim,
    )

    resume_state: Optional[Dict[str, Any]] = None
    if resume_path is not None and Path(resume_path).exists():
        with open(resume_path, "rb") as f:
            resume_state = pickle.load(f)

    start_update = 0
    if resume_state is not None:
        if "completed_updates" in resume_state:
            start_update = int(resume_state["completed_updates"])
        elif "history" in resume_state and resume_state["history"]:
            start_update = int(resume_state["history"][-1].get("update", -1)) + 1

    rng = jax.random.PRNGKey(cfg.seed + start_update)
    rng, k_pi, k_bel, k_grp, k_ora, k_pool, k_env, k_desync = jax.random.split(rng, 8)

    params_pi = init_policy_params(policy_net, k_pi) if policy_params_init is None else _clone_params(policy_params_init)
    params_bc = _clone_params(params_pi)
    params_rho_base = _clone_params(params_pi)

    params_b_fast = _init_belief_params(belief_net, k_bel) if belief_params_init is None else _clone_params(belief_params_init)
    params_b_ema = _clone_params(params_b_fast)

    if grp_params_init is None:
        params_grp_base = grp_net.init(k_grp, jnp.zeros((2, GRP_FEATURE_DIM), dtype=jnp.float32))
    else:
        params_grp_base = _clone_params(grp_params_init)
    params_grp_ema = _clone_params(params_grp_base)

    params_oracle = _init_oracle_params(oracle_net, cfg.belief.river_dim, k_ora)

    tx_policy, tx_oracle, tx_belief = build_isolated_optimizers(cfg.rnad, cfg.oracle, cfg.belief)
    opt_pi = tx_policy.init(params_pi)
    opt_oracle = tx_oracle.init(params_oracle)
    opt_belief = tx_belief.init(params_b_fast)

    rms_state: RunningRMSState = init_running_rms(1.0)
    pfsp_pool: Deque[Any] = deque(maxlen=max(1, int(cfg.rnad.pfsp_pool_size)))
    outer_snapshots: Deque[Any] = deque(maxlen=max(1, int(cfg.rnad.max_outer_snapshots)))
    history: List[Dict[str, float]] = []
    steps_since_outer_reset = 0
    healer = SelfHealingManager(cfg=cfg.rnad)

    if resume_state is not None:
        loaded_pi = resume_state.get("params_pi", resume_state.get("policy_params"))
        if loaded_pi is not None:
            _validate_resume_shapes("Policy", params_pi, loaded_pi, resume_path)
            params_pi = jax.device_put(loaded_pi)
        if "params_bc" in resume_state:
            _validate_resume_shapes("BC anchor", params_bc, resume_state["params_bc"], resume_path)
            params_bc = jax.device_put(resume_state["params_bc"])
        if "params_rho_base" in resume_state:
            _validate_resume_shapes("R-NaD rho_base", params_rho_base, resume_state["params_rho_base"], resume_path)
            params_rho_base = jax.device_put(resume_state["params_rho_base"])
        else:
            params_rho_base = _clone_params(params_pi)
        loaded_ora = resume_state.get("params_oracle", resume_state.get("oracle_params"))
        if loaded_ora is not None:
            _validate_resume_shapes("Oracle", params_oracle, loaded_ora, resume_path)
            params_oracle = jax.device_put(loaded_ora)
        loaded_b_fast = resume_state.get("params_b_fast", resume_state.get("belief_ema_params"))
        if loaded_b_fast is not None:
            _validate_resume_shapes("Belief fast", params_b_fast, loaded_b_fast, resume_path)
            params_b_fast = jax.device_put(loaded_b_fast)
        loaded_b_ema = resume_state.get("params_b_ema", resume_state.get("belief_ema_params"))
        if loaded_b_ema is not None:
            _validate_resume_shapes("Belief EMA", params_b_ema, loaded_b_ema, resume_path)
            params_b_ema = jax.device_put(loaded_b_ema)
        if "opt_pi" in resume_state:
            opt_pi = jax.device_put(resume_state["opt_pi"])
        if "opt_oracle" in resume_state:
            opt_oracle = jax.device_put(resume_state["opt_oracle"])
        if "opt_belief" in resume_state:
            opt_belief = jax.device_put(resume_state["opt_belief"])
        if "rms_state" in resume_state:
            rms_state = jax.device_put(resume_state["rms_state"])
        if "pfsp_pool" in resume_state and resume_state["pfsp_pool"]:
            for snap in resume_state["pfsp_pool"]:
                pfsp_pool.append(_clone_host_params(snap))
        if "outer_snapshots" in resume_state and resume_state["outer_snapshots"]:
            for snap in resume_state["outer_snapshots"]:
                outer_snapshots.append(_clone_host_params(snap))
        if "history" in resume_state and resume_state["history"]:
            history = list(resume_state["history"])
        if start_update <= int(cfg.rnad.warmup_updates):
            steps_since_outer_reset = 0
        elif "steps_since_outer_reset" in resume_state:
            steps_since_outer_reset = int(resume_state["steps_since_outer_reset"])
        if "healer_lr_scale" in resume_state:
            healer.current_lr_scale = float(resume_state["healer_lr_scale"])
        if verbose:
            print(
                f"[Phase 1/2 RL] Resumed from {resume_path} at update {start_update}/{cfg.num_updates} "
                f"(lr_scale={healer.current_lr_scale:.3g}, pfsp={len(pfsp_pool)}, snaps={len(outer_snapshots)})",
                flush=True,
            )

    if not pfsp_pool:
        pfsp_pool.append(_clone_host_params(params_pi))

    w_tables, is_pfsp_env, learning_seat_mask = sample_meta_table_pool(
        cfg.num_envs,
        k_pool,
        phoenix_7d_only=cfg.rnad.phoenix_7d_only,
        pfsp_frac=cfg.rnad.pfsp_frac,
    )
    is_pfsp_np = np.asarray(jax.device_get(is_pfsp_env), dtype=bool)
    sp_idx_np = np.flatnonzero(~is_pfsp_np)
    pfsp_idx_np = np.flatnonzero(is_pfsp_np)

    if use_pmap:
        # Ensure equal static (n_sp_per_dev, n_pfsp_per_dev) per GPU while striding evenly across seats/rules
        rem_pfsp = len(pfsp_idx_np) % n_devices
        if rem_pfsp != 0:
            move_to_sp = pfsp_idx_np[-rem_pfsp:]
            pfsp_idx_np = pfsp_idx_np[:-rem_pfsp]
            sp_idx_np = np.concatenate([sp_idx_np, move_to_sp])
            move_j = jnp.asarray(move_to_sp, dtype=jnp.int32)
            learning_seat_mask = learning_seat_mask.at[move_j].set(True)
            is_pfsp_env = is_pfsp_env.at[move_j].set(False)
        n_sp_per_dev = len(sp_idx_np) // n_devices
        n_pfsp_per_dev = len(pfsp_idx_np) // n_devices
        dev_orders = [
            np.concatenate([sp_idx_np[d::n_devices], pfsp_idx_np[d::n_devices]])
            for d in range(n_devices)
        ]
        reorder_idx = jnp.asarray(np.concatenate(dev_orders), dtype=jnp.int32)
    else:
        n_sp_per_dev = int(len(sp_idx_np))
        n_pfsp_per_dev = int(len(pfsp_idx_np))
        reorder_idx = jnp.asarray(np.concatenate([sp_idx_np, pfsp_idx_np]), dtype=jnp.int32)

    w_tables = w_tables[reorder_idx]
    is_pfsp_env = is_pfsp_env[reorder_idx]
    learning_seat_mask = learning_seat_mask[reorder_idx]

    # Initialize parallel environments and stagger East-1..South-4 phases before sharding
    env_keys = jax.random.split(k_env, cfg.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(env_keys)
    env_state = _desync_hanchan_envs(env, env_state, k_desync, cfg.num_envs, cfg.round_mode)
    if use_pmap:
        env_state = shard_for_pmap(
            jax.tree_util.tree_map(lambda x: x.reshape((n_devices, envs_per_dev) + x.shape[1:]), env_state),
            n_devices,
        )
        w_tables_exec = shard_for_pmap(w_tables.reshape((n_devices, envs_per_dev) + w_tables.shape[1:]), n_devices)
        learn_mask_exec = shard_for_pmap(learning_seat_mask.reshape((n_devices, envs_per_dev) + learning_seat_mask.shape[1:]), n_devices)
    else:
        w_tables_exec = w_tables
        learn_mask_exec = learning_seat_mask

    np_rng = np.random.default_rng(cfg.seed + start_update)

    # =========================================================================
    # JIT/PMAP 1: Vectorized Rollout (Per-Rollout Single-Snapshot PFSP, Zero Recompile!)
    # =========================================================================
    def _rollout_single_dev(
        init_env_state,
        p_pi,
        p_opp,
        p_b_ema,
        p_grp_base,
        p_grp_ema,
        p_oracle,
        w_tab,
        learn_mask,
        t_cal,
        roll_rng,
    ):
        grp_base_fn = lambda gf: grp_net.apply(p_grp_base, gf)
        opp_temp = float(cfg.rnad.opp_temperature)

        def eval_single_env_step(st, w_abs, l_mask_4, step_rng, is_self_play: bool):
            k_drop, k_act, k_step, k_reset = jax.random.split(step_rng, 4)
            clean_st = clean_auto_reset_state(st)
            cp = clean_st.current_player.astype(jnp.int32)
            legal_mask = clean_st.legal_action_mask.astype(jnp.bool_)

            obs = encode_obs_v6(clean_st)
            ora_planes = encode_oracle_planes(clean_st)
            grp_f = encode_grp_features(clean_st)
            tbl = extract_table_snapshot(clean_st)
            rsnap = None if cfg.rnad.freeze_belief else extract_ron_snapshot(clean_st)

            w_ego_4x4 = rotate_utility_matrix_ego(w_abs, cp)
            w_me = w_ego_4x4[0]
            w_opp_5d = build_opponent_utility_features(
                w_ego_4x4[1:],
                jnp.ones((3,), dtype=jnp.float32),
                rng_key=k_drop,
                dropout_prob=cfg.rnad.utility_dropout,
            )

            b_out = belief_net.apply(
                p_b_ema,
                planes=obs["planes"],
                scalars=obs["scalars"],
                river_tokens=obs["river_tokens"],
                river_tile_ids=obs["river_tile_ids"],
                river_valid_mask=obs["river_valid_mask"],
                w_opp_5d=w_opp_5d,
                temperature_cal=t_cal,
            )
            grp_curr_4x4, grp_logits_base = grp_net.apply(p_grp_base, grp_f, return_logits=True)

            du_ron_3opp, du_ron_87, du_agari_3, du_ryu, call_threat = compute_analytical_trinity(
                s_table=tbl,
                p_deal_3opp=b_out["p_deal"],
                p_score_bins=b_out["p_score_bin"],
                p_call_threat_3=b_out["p_call_threat"],
                p_tenpai_3=b_out["p_tenpai"],
                w_me=w_me,
                last_draw_type=obs["last_draw_type"],
                is_last_draw_red=obs["is_last_draw_red"],
                grp_fn=grp_base_fn,
            )

            ora_out = oracle_net.apply(
                p_oracle,
                planes=obs["planes"],
                oracle_planes=ora_planes,
                scalars=obs["scalars"],
                w_table_ego=w_ego_4x4,
                grp_logits_abs=grp_logits_base,
                current_player=cp,
                river_spatial_feat=b_out["river_spatial_feat"],
            )

            pol_kwargs = dict(
                planes=obs["planes"],
                delta_u_ron_3opp=du_ron_3opp,
                call_threat_plane=call_threat,
                scalars=obs["scalars"],
                river_tokens=obs["river_tokens"],
                river_tile_ids=obs["river_tile_ids"],
                river_valid_mask=obs["river_valid_mask"],
                last_draw_type=obs["last_draw_type"],
                is_last_draw_red=obs["is_last_draw_red"],
                target_type=obs["target_type"],
                target_is_red=obs["target_is_red"],
                w_ego=w_me,
                delta_u_agari_3=du_agari_3,
                delta_u_ryuukyoku=du_ryu,
                delta_u_ron_87=du_ron_87,
                legal_action_mask=legal_mask,
            )
            logits_pi = policy_net.apply(p_pi, **pol_kwargs)
            p_riichi_pi = masked_categorical_probs(logits_pi, legal_mask)[Action.RIICHI]
            if is_self_play:
                is_learn = jnp.bool_(True)
                log_p_act = masked_categorical_log_probs(logits_pi, legal_mask)
                action = jax.random.categorical(k_act, log_p_act).astype(jnp.int32)
                old_log_p = log_p_act[action]
            else:
                logits_opp = policy_net.apply(p_opp, **pol_kwargs)
                is_learn = l_mask_4[cp]
                if opp_temp <= 0.0:
                    opp_act = jnp.argmax(jnp.where(legal_mask, logits_opp, NEG_INF)).astype(jnp.int32)
                    log_p_pi = masked_categorical_log_probs(logits_pi, legal_mask)
                    act_pi = jax.random.categorical(k_act, log_p_pi).astype(jnp.int32)
                    action = jnp.where(is_learn, act_pi, opp_act)
                    old_log_p = jnp.where(is_learn, log_p_pi[action], jnp.float32(0.0))
                else:
                    scaled_opp = logits_opp / jnp.float32(opp_temp)
                    active_logits = jnp.where(is_learn, logits_pi, scaled_opp)
                    log_p_act = masked_categorical_log_probs(active_logits, legal_mask)
                    action = jax.random.categorical(k_act, log_p_act).astype(jnp.int32)
                    old_log_p = log_p_act[action]

            raw_next = env.step(clean_st, action, k_step)
            done = raw_next.terminated | raw_next.truncated
            round_ended = (
                done
                | (raw_next.round_state.round != clean_st.round_state.round)
                | (raw_next.round_state.honba != clean_st.round_state.honba)
            )
            pending_has_winner = jnp.any(clean_st.pending_winners)
            pending_winner = jnp.argmax(clean_st.pending_winners.astype(jnp.int32)).astype(jnp.int32)
            step_winner_abs = jnp.where(
                (action == Action.TSUMO) | (action == Action.RON),
                cp,
                jnp.where((action == Action.PASS) & pending_has_winner, pending_winner, jnp.int32(4)),
            )
            scores_curr = clean_st.round_state.score.astype(jnp.int32)
            scores_post = raw_next.round_state.score.astype(jnp.int32)
            term_rank_4x4 = compute_terminal_rank_matrix_4x4(
                scores_post, raw_next.round_state.init_wind, done
            )
            grp_ema_next_4x4 = grp_net.apply(p_grp_ema, encode_grp_features(raw_next))

            next_st = jax.lax.cond(done, lambda: env.init(k_reset), lambda: raw_next)

            trans = {
                "planes": obs["planes"],
                "oracle_planes": ora_planes,
                "scalars": obs["scalars"],
                "river_tokens": obs["river_tokens"],
                "river_tile_ids": obs["river_tile_ids"].astype(jnp.int8),
                "river_valid_mask": obs["river_valid_mask"],
                "last_draw_type": obs["last_draw_type"].astype(jnp.int8),
                "is_last_draw_red": obs["is_last_draw_red"],
                "target_type": obs["target_type"].astype(jnp.int8),
                "target_is_red": obs["target_is_red"],
                "delta_u_ron_3opp": du_ron_3opp,
                "delta_u_ron_87": du_ron_87,
                "delta_u_agari_3": du_agari_3,
                "delta_u_ryuukyoku": du_ryu,
                "call_threat_plane": call_threat,
                "grp_logits_base": grp_logits_base,
                "grp_curr_4x4": grp_curr_4x4,
                "grp_ema_next_4x4": grp_ema_next_4x4,
                "v_oracle_4x4": ora_out["probs_4x4"],
                "v_delta_u_ego": ora_out["delta_u_ego"],
                "belief_pred_mean": jnp.mean(b_out["p_deal"]),
                "term_rank_4x4": term_rank_4x4,
                "w_table_ego": w_ego_4x4,
                "w_opp_5d": w_opp_5d,
                "action": action,
                "old_log_prob": old_log_p,
                "p_riichi": p_riichi_pi,
                "legal_mask": legal_mask,
                "current_player": cp.astype(jnp.int8),
                "is_learn": is_learn,
                "done": done,
                "round_ended": round_ended,
                "scores_curr": scores_curr,
                "scores_post": scores_post,
                "step_winner_abs": step_winner_abs.astype(jnp.int8),
            }
            if cfg.rnad.freeze_belief:
                trans["river_spatial_feat"] = b_out["river_spatial_feat"].astype(jnp.float16)
            if rsnap is not None:
                trans["rsnap"] = rsnap
            return next_st, trans

        def scan_step(carry, _):
            st_b, r = carry
            r, k_sub = jax.random.split(r)
            step_keys = jax.random.split(k_sub, envs_per_dev)
            parts_st, parts_tr = [], []
            if n_sp_per_dev > 0:
                st_sp, tr_sp = jax.vmap(
                    lambda s, w, lm_, k: eval_single_env_step(s, w, lm_, k, is_self_play=True)
                )(
                    jax.tree_util.tree_map(lambda x: x[:n_sp_per_dev], st_b),
                    w_tab[:n_sp_per_dev],
                    learn_mask[:n_sp_per_dev],
                    step_keys[:n_sp_per_dev],
                )
                parts_st.append(st_sp)
                parts_tr.append(tr_sp)
            if n_pfsp_per_dev > 0:
                st_pf, tr_pf = jax.vmap(
                    lambda s, w, lm_, k: eval_single_env_step(s, w, lm_, k, is_self_play=False)
                )(
                    jax.tree_util.tree_map(lambda x: x[n_sp_per_dev:], st_b),
                    w_tab[n_sp_per_dev:],
                    learn_mask[n_sp_per_dev:],
                    step_keys[n_sp_per_dev:],
                )
                parts_st.append(st_pf)
                parts_tr.append(tr_pf)
            if len(parts_st) == 1:
                next_st_b, trans_b = parts_st[0], parts_tr[0]
            else:
                next_st_b = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), parts_st[0], parts_st[1])
                trans_b = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), parts_tr[0], parts_tr[1])
            return (next_st_b, r), trans_b

        (final_st_b, _), traj_tb = jax.lax.scan(
            scan_step, (init_env_state, roll_rng), None, length=cfg.num_steps
        )

        # Compute bootstrap V_oracle(s_T) at final_st_b for Dual-Rate Simplex TD(lambda)
        def eval_bootstrap(st, w_abs):
            clean_st = clean_auto_reset_state(st)
            cp = clean_st.current_player.astype(jnp.int32)
            obs = encode_obs_v6(clean_st)
            ora_planes = encode_oracle_planes(clean_st)
            grp_f = encode_grp_features(clean_st)
            w_ego_4x4 = rotate_utility_matrix_ego(w_abs, cp)
            w_opp_5d = build_opponent_utility_features(
                w_ego_4x4[1:], jnp.ones((3,), dtype=jnp.float32), rng_key=None, dropout_prob=0.0
            )
            b_out = belief_net.apply(
                p_b_ema,
                planes=obs["planes"],
                scalars=obs["scalars"],
                river_tokens=obs["river_tokens"],
                river_tile_ids=obs["river_tile_ids"],
                river_valid_mask=obs["river_valid_mask"],
                w_opp_5d=w_opp_5d,
                temperature_cal=t_cal,
            )
            _, grp_logits_base = grp_net.apply(p_grp_base, grp_f, return_logits=True)
            ora_out = oracle_net.apply(
                p_oracle,
                planes=obs["planes"],
                oracle_planes=ora_planes,
                scalars=obs["scalars"],
                w_table_ego=w_ego_4x4,
                grp_logits_abs=grp_logits_base,
                current_player=cp,
                river_spatial_feat=b_out["river_spatial_feat"],
            )
            return ora_out["probs_4x4"]

        v_boot_b = jax.vmap(eval_bootstrap)(final_st_b, w_tab)  # (envs_per_dev, 4, 4)
        return final_st_b, traj_tb, v_boot_b

    rollout_fn = jax.pmap(_rollout_single_dev) if use_pmap else jax.jit(_rollout_single_dev)

    n_total = cfg.num_envs * cfg.num_steps
    n_per_dev = envs_per_dev * cfg.num_steps
    mb_per_dev = min(cfg.minibatch_size // n_devices if use_pmap else cfg.minibatch_size, n_per_dev)
    max_prep_c = max(mb_per_dev, min(n_per_dev, 256))
    prep_chunk = max(d for d in range(1, min(n_per_dev, max_prep_c) + 1) if n_per_dev % d == 0)

    # =========================================================================
    # JIT/PMAP 2: Dual-Rate Simplex TD + Tail Masking + Running-RMS + Cached Anchors
    # =========================================================================
    def _make_prepare_fn(precompute_bc_rho: bool):
        def _prepare_single_dev(traj_tb, v_boot_b, rms_st, p_bc, p_rho_base):
            # Transpose (T, B, ...) -> (B, T, ...)
            traj_bt = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_tb)

            targets_4x4 = calculate_dual_rate_simplex_td_lambda(
                v_oracle_4x4=traj_bt["v_oracle_4x4"],
                v_bootstrap_4x4=v_boot_b,
                grp_ema_next_4x4=traj_bt["grp_ema_next_4x4"],
                terminal_rank_4x4=traj_bt["term_rank_4x4"],
                dones=traj_bt["done"],
                round_ended=traj_bt["round_ended"],
                lambda_intra=cfg.rnad.lambda_intra,
                lambda_inter=cfg.rnad.lambda_inter,
            )  # (envs_per_dev, T, 4, 4)

            b_idx = jnp.arange(envs_per_dev, dtype=jnp.int32)[:, None]
            t_idx = jnp.arange(cfg.num_steps, dtype=jnp.int32)[None, :]
            cp_bt = traj_bt["current_player"].astype(jnp.int32)

            win_ego, score_delta_ego, aux_valid = compute_round_outcome_targets(
                scores_curr=traj_bt["scores_curr"],
                scores_post=traj_bt["scores_post"],
                round_ended=traj_bt["round_ended"],
                dones=traj_bt["done"],
                step_winner_abs=traj_bt["step_winner_abs"],
                current_players=cp_bt,
            )

            num_legal = jnp.sum(traj_bt["legal_mask"].astype(jnp.int32), axis=-1)
            base_active = traj_bt["is_learn"] & (num_legal >= 2)
            if cfg.rnad.mask_unterminated_tail:
                comp_cnt = jnp.sum((base_active & aux_valid).astype(jnp.float32))
                if use_pmap:
                    comp_cnt = jax.lax.psum(comp_cnt, axis_name="devices")
                round_valid = jnp.where(comp_cnt > 0.5, aux_valid, jnp.ones_like(aux_valid))
            else:
                round_valid = jnp.ones_like(aux_valid)
            active_mask = base_active & round_valid

            # Ego-frame target utility shift relative to frozen GRP_base(s_t) and advantage via delta_u_head
            ego_seats = (cp_bt[:, :, None] + jnp.arange(4, dtype=jnp.int32)[None, None, :]) % 4
            targets_ego = targets_4x4[b_idx[:, :, None], t_idx[:, :, None], ego_seats]
            grp_curr_ego = traj_bt["grp_curr_4x4"][b_idx[:, :, None], t_idx[:, :, None], ego_seats]
            delta_u_target_ego = jnp.sum((targets_ego - grp_curr_ego) * traj_bt["w_table_ego"], axis=-1)
            raw_adv = delta_u_target_ego[..., 0] - traj_bt["v_delta_u_ego"][..., 0]

            # Honest pre-update Oracle R^2 (overall & late-wall) and mean advantage on completed active steps
            cnt_act = jnp.sum(active_mask.astype(jnp.float32))
            y_true = delta_u_target_ego[..., 0]
            sum_y = jnp.sum(jnp.where(active_mask, y_true, 0.0))
            sum_adv = jnp.sum(jnp.where(active_mask, raw_adv, 0.0))
            sum_sq_res = jnp.sum(jnp.where(active_mask, raw_adv ** 2, 0.0))
            sum_y_sq = jnp.sum(jnp.where(active_mask, y_true ** 2, 0.0))

            late_mask = active_mask & (traj_bt["scalars"][..., 15] < (25.0 / 70.0))
            cnt_late = jnp.sum(late_mask.astype(jnp.float32))
            sum_y_late = jnp.sum(jnp.where(late_mask, y_true, 0.0))
            sum_res_late = jnp.sum(jnp.where(late_mask, raw_adv ** 2, 0.0))
            sum_y2_late = jnp.sum(jnp.where(late_mask, y_true ** 2, 0.0))

            comp_frac = jnp.mean(aux_valid.astype(jnp.float32))
            bel_pred_mean = jnp.mean(traj_bt["belief_pred_mean"])
            if use_pmap:
                cnt_act = jax.lax.psum(cnt_act, axis_name="devices")
                sum_y = jax.lax.psum(sum_y, axis_name="devices")
                sum_adv = jax.lax.psum(sum_adv, axis_name="devices")
                sum_sq_res = jax.lax.psum(sum_sq_res, axis_name="devices")
                sum_y_sq = jax.lax.psum(sum_y_sq, axis_name="devices")
                cnt_late = jax.lax.psum(cnt_late, axis_name="devices")
                sum_y_late = jax.lax.psum(sum_y_late, axis_name="devices")
                sum_res_late = jax.lax.psum(sum_res_late, axis_name="devices")
                sum_y2_late = jax.lax.psum(sum_y2_late, axis_name="devices")
                comp_frac = jax.lax.pmean(comp_frac, axis_name="devices")
                bel_pred_mean = jax.lax.pmean(bel_pred_mean, axis_name="devices")
            denom_act = jnp.maximum(cnt_act, 1.0)
            y_mean = sum_y / denom_act
            adv_mean = sum_adv / denom_act
            ss_tot = jnp.maximum(sum_y_sq - cnt_act * (y_mean ** 2), 1e-8)
            oracle_r2 = jnp.where(cnt_act > 1.5, 1.0 - (sum_sq_res / ss_tot), jnp.array(0.0, dtype=jnp.float32))

            y_mean_late = sum_y_late / jnp.maximum(cnt_late, 1.0)
            ss_tot_late = jnp.maximum(sum_y2_late - cnt_late * (y_mean_late ** 2), 1e-8)
            oracle_r2_late = jnp.where(
                cnt_late > 1.5, 1.0 - (sum_res_late / ss_tot_late), jnp.array(0.0, dtype=jnp.float32)
            )

            new_rms, scaled_adv, rms_val = update_running_rms(
                rms_st,
                raw_adv,
                active_mask,
                beta=cfg.rnad.running_rms_beta,
                eps=cfg.rnad.running_rms_eps,
            )
            if use_pmap:
                synced_sq = jax.lax.pmean(new_rms.second_moment, axis_name="devices")
                synced_steps = jax.lax.pmax(new_rms.steps, axis_name="devices")
                rms_val = jnp.sqrt(jnp.maximum(synced_sq, 0.0) + float(cfg.rnad.running_rms_eps))
                scaled_adv = jnp.where(active_mask, raw_adv / rms_val, 0.0)
                new_rms = RunningRMSState(second_moment=synced_sq, steps=synced_steps)

            # Measure both confounded gap and unconfounded REINFORCE Riichi gradient E[(1_R - p_R) * A]
            can_r = active_mask & traj_bt["legal_mask"][..., Action.RIICHI]
            did_r = can_r & (traj_bt["action"] == Action.RIICHI)
            did_dama = can_r & (traj_bt["action"] != Action.RIICHI)
            sum_r = jnp.sum(jnp.where(did_r, raw_adv, 0.0))
            cnt_r = jnp.sum(did_r.astype(jnp.float32))
            sum_d = jnp.sum(jnp.where(did_dama, raw_adv, 0.0))
            cnt_d = jnp.sum(did_dama.astype(jnp.float32))
            ind_r = (traj_bt["action"] == Action.RIICHI).astype(jnp.float32)
            pg_r_sum = jnp.sum(jnp.where(can_r, (ind_r - traj_bt["p_riichi"]) * raw_adv, 0.0))
            pg_r_cnt = jnp.sum(can_r.astype(jnp.float32))
            if use_pmap:
                sum_r = jax.lax.psum(sum_r, axis_name="devices")
                cnt_r = jax.lax.psum(cnt_r, axis_name="devices")
                sum_d = jax.lax.psum(sum_d, axis_name="devices")
                cnt_d = jax.lax.psum(cnt_d, axis_name="devices")
                pg_r_sum = jax.lax.psum(pg_r_sum, axis_name="devices")
                pg_r_cnt = jax.lax.psum(pg_r_cnt, axis_name="devices")
            has_both_r = (cnt_r > 0.5) & (cnt_d > 0.5)
            riichi_adv_gap = jnp.where(
                has_both_r,
                (sum_r / jnp.maximum(cnt_r, 1.0)) - (sum_d / jnp.maximum(cnt_d, 1.0)),
                jnp.array(0.0, dtype=jnp.float32),
            )
            riichi_pg = jnp.where(
                pg_r_cnt > 0.5,
                pg_r_sum / jnp.maximum(pg_r_cnt, 1.0),
                jnp.array(0.0, dtype=jnp.float32),
            )

            # Pack strictly required arrays for minibatch updates
            keep_keys = [
                "planes",
                "oracle_planes",
                "scalars",
                "w_table_ego",
                "grp_logits_base",
                "current_player",
            ]
            if cfg.rnad.freeze_belief:
                keep_keys.append("river_spatial_feat")
            else:
                keep_keys.extend(["river_tokens", "river_tile_ids", "river_valid_mask", "w_opp_5d"])
            if precompute_bc_rho:
                keep_keys.extend([
                    "river_tokens",
                    "river_tile_ids",
                    "river_valid_mask",
                    "last_draw_type",
                    "is_last_draw_red",
                    "target_type",
                    "target_is_red",
                    "delta_u_ron_3opp",
                    "delta_u_ron_87",
                    "delta_u_agari_3",
                    "delta_u_ryuukyoku",
                    "call_threat_plane",
                    "action",
                    "old_log_prob",
                    "legal_mask",
                ])
            flat = {
                k: traj_bt[k].reshape((n_per_dev,) + traj_bt[k].shape[2:])
                for k in dict.fromkeys(keep_keys)
            }
            flat["targets_4x4"] = targets_4x4.reshape((n_per_dev, 4, 4))
            flat["delta_u_target_ego"] = delta_u_target_ego.reshape((n_per_dev, 4))
            flat["round_valid"] = round_valid.reshape((n_per_dev,))
            flat["win_ego"] = win_ego.reshape((n_per_dev,))
            flat["score_delta_ego"] = score_delta_ego.reshape((n_per_dev, 4))
            flat["aux_valid"] = aux_valid.reshape((n_per_dev,))

            if precompute_bc_rho:
                flat["adv_scaled"] = scaled_adv.reshape((n_per_dev,))
                flat["is_learn_eff"] = (traj_bt["is_learn"] & round_valid).reshape((n_per_dev,))

                def _eval_bc_rho_chunk(_, chunk):
                    mask_c = chunk["legal_mask"]
                    pol_in = dict(
                        planes=chunk["planes"],
                        delta_u_ron_3opp=chunk["delta_u_ron_3opp"],
                        call_threat_plane=chunk["call_threat_plane"],
                        scalars=chunk["scalars"],
                        river_tokens=chunk["river_tokens"],
                        river_tile_ids=chunk["river_tile_ids"].astype(jnp.int32),
                        river_valid_mask=chunk["river_valid_mask"],
                        last_draw_type=chunk["last_draw_type"].astype(jnp.int32),
                        is_last_draw_red=chunk["is_last_draw_red"],
                        target_type=chunk["target_type"].astype(jnp.int32),
                        target_is_red=chunk["target_is_red"],
                        w_ego=chunk["w_table_ego"][:, 0, :],
                        delta_u_agari_3=chunk["delta_u_agari_3"],
                        delta_u_ryuukyoku=chunk["delta_u_ryuukyoku"],
                        delta_u_ron_87=chunk["delta_u_ron_87"],
                        legal_action_mask=mask_c,
                    )
                    bc_lg = jax.lax.stop_gradient(policy_net.apply(p_bc, **pol_in))
                    if int(cfg.rnad.outer_reset_steps) <= 0:
                        rho_lg = bc_lg
                    else:
                        rho_lg = jax.lax.stop_gradient(policy_net.apply(p_rho_base, **pol_in))
                    _, log_rho_c = compute_dual_anchor_rho_probs(
                        rho_lg,
                        bc_lg,
                        mask_c,
                        mix_bc=cfg.rnad.rho_mix_bc,
                        mix_uniform=cfg.rnad.rho_mix_uniform,
                    )
                    return None, (bc_lg.astype(jnp.float32), log_rho_c.astype(jnp.float32))

                n_chunks = n_per_dev // prep_chunk
                chunk_keys = (
                    "planes",
                    "delta_u_ron_3opp",
                    "call_threat_plane",
                    "scalars",
                    "river_tokens",
                    "river_tile_ids",
                    "river_valid_mask",
                    "last_draw_type",
                    "is_last_draw_red",
                    "target_type",
                    "target_is_red",
                    "w_table_ego",
                    "delta_u_agari_3",
                    "delta_u_ryuukyoku",
                    "delta_u_ron_87",
                    "legal_mask",
                )
                chunk_in = {
                    k: flat[k].reshape((n_chunks, prep_chunk) + flat[k].shape[1:])
                    for k in chunk_keys
                }
                _, (bc_lg_c, log_rho_c) = jax.lax.scan(_eval_bc_rho_chunk, None, chunk_in)
                flat["bc_logits"] = bc_lg_c.reshape((n_per_dev, Action.NUM_ACTION))
                flat["log_rho"] = log_rho_c.reshape((n_per_dev, Action.NUM_ACTION))

            if cfg.rnad.freeze_belief:
                ron_overflow = jnp.bool_(False)
            else:
                eff_ron_k_max = max(int(cfg.rnad.ron_k_max), 2 * n_per_dev)
                rsnap_flat = jax.tree_util.tree_map(
                    lambda x: x.reshape((n_per_dev,) + x.shape[2:]), traj_bt["rsnap"]
                )
                y_ron, y_sb, y_ct, y_ten, ron_overflow = compute_sparse_belief_labels(
                    rsnap_flat, k_max=eff_ron_k_max
                )
                if use_pmap:
                    ron_overflow = jax.lax.pmax(ron_overflow.astype(jnp.int32), axis_name="devices") > 0
                flat["y_ron"] = y_ron
                flat["y_score_bin"] = y_sb
                flat["y_call_threat"] = y_ct
                flat["y_tenpai"] = y_ten
            return (
                flat,
                new_rms,
                rms_val,
                riichi_adv_gap,
                riichi_pg,
                oracle_r2,
                oracle_r2_late,
                adv_mean,
                comp_frac,
                bel_pred_mean,
                ron_overflow,
            )

        return jax.pmap(_prepare_single_dev, axis_name="devices") if use_pmap else jax.jit(_prepare_single_dev)

    prepare_warmup_fn = _make_prepare_fn(precompute_bc_rho=False)
    prepare_full_fn = _make_prepare_fn(precompute_bc_rho=True)

    # =========================================================================
    # JIT/PMAP 3: Decoupled Isolated Optimizers (`update_ora_mb` + `update_bel_pi_mb`)
    # =========================================================================
    def _ora_step_fn(p_ora, o_ora, p_b_ema, mb_ora, t_cal):
        if cfg.rnad.freeze_belief:
            river_feat = jax.lax.stop_gradient(mb_ora["river_spatial_feat"].astype(jnp.float32))
        else:
            b_ema_out = belief_net.apply(
                p_b_ema,
                planes=mb_ora["planes"],
                scalars=mb_ora["scalars"],
                river_tokens=mb_ora["river_tokens"],
                river_tile_ids=mb_ora["river_tile_ids"].astype(jnp.int32),
                river_valid_mask=mb_ora["river_valid_mask"],
                w_opp_5d=mb_ora["w_opp_5d"],
                temperature_cal=t_cal,
            )
            river_feat = jax.lax.stop_gradient(b_ema_out["river_spatial_feat"])

        def ora_loss_fn(p):
            out = oracle_net.apply(
                p,
                planes=mb_ora["planes"].astype(jnp.float32),
                oracle_planes=mb_ora["oracle_planes"].astype(jnp.float32),
                scalars=mb_ora["scalars"],
                w_table_ego=mb_ora["w_table_ego"],
                grp_logits_abs=mb_ora["grp_logits_base"],
                current_player=mb_ora["current_player"].astype(jnp.int32),
                river_spatial_feat=river_feat,
            )
            return compute_oracle_loss_v6(
                out,
                targets_4x4=mb_ora["targets_4x4"],
                round_winner_ego=mb_ora["win_ego"],
                round_score_delta_ego=mb_ora["score_delta_ego"],
                aux_valid=mb_ora["aux_valid"],
                cfg=cfg.oracle,
                delta_u_target_ego=mb_ora["delta_u_target_ego"],
                valid_mask=mb_ora["round_valid"],
            )

        (_, ora_metrics), g_ora = jax.value_and_grad(ora_loss_fn, has_aux=True)(p_ora)
        if use_pmap:
            g_ora = jax.lax.pmean(g_ora, axis_name="devices")
        u_ora, o_ora = tx_oracle.update(g_ora, o_ora, p_ora)
        p_ora = optax.apply_updates(p_ora, u_ora)
        out_metrics = {**ora_metrics, "ora_finite_grads": _tree_is_finite(g_ora).astype(jnp.float32)}
        if use_pmap:
            out_metrics = jax.lax.pmean(out_metrics, axis_name="devices")
        return p_ora, o_ora, out_metrics

    update_ora_mb = jax.pmap(_ora_step_fn, axis_name="devices") if use_pmap else jax.jit(_ora_step_fn)

    def _make_bel_pi_step(update_policy: bool):
        def step_fn(
            p_pi,
            o_pi,
            p_b_fast,
            o_bel,
            p_b_ema,
            mb,
            shield_lam,
            ron_overflow,
        ):
            # 1. Belief Update (frozen by default; if enabled, guarded against ron_k_max overflow)
            if not cfg.rnad.freeze_belief:
                def bel_loss_fn(p):
                    out = belief_net.apply(
                        p,
                        planes=mb["planes"],
                        scalars=mb["scalars"],
                        river_tokens=mb["river_tokens"],
                        river_tile_ids=mb["river_tile_ids"].astype(jnp.int32),
                        river_valid_mask=mb["river_valid_mask"],
                        w_opp_5d=mb["w_opp_5d"],
                        temperature_cal=1.0,
                    )
                    return compute_belief_loss_v6(
                        out,
                        y_ron=mb["y_ron"],
                        y_score_bin=mb["y_score_bin"],
                        y_call_threat=mb["y_call_threat"],
                        y_tenpai=mb["y_tenpai"],
                        cfg=cfg.belief,
                    )

                (_, bel_metrics), g_bel = jax.value_and_grad(bel_loss_fn, has_aux=True)(p_b_fast)
                if use_pmap:
                    g_bel = jax.lax.pmean(g_bel, axis_name="devices")

                def _apply_bel(_):
                    u_bel, new_o_bel = tx_belief.update(g_bel, o_bel, p_b_fast)
                    new_p_fast = optax.apply_updates(p_b_fast, u_bel)
                    new_p_ema = ema_update_params(p_b_ema, new_p_fast, cfg.belief.ema_tau)
                    return new_p_fast, new_o_bel, new_p_ema

                p_b_fast, o_bel, p_b_ema = jax.lax.cond(
                    ron_overflow,
                    lambda _: (p_b_fast, o_bel, p_b_ema),
                    _apply_bel,
                    None,
                )
                finite_all = ron_overflow | _tree_is_finite(g_bel)
            else:
                bel_metrics = {
                    "belief_loss": jnp.array(0.0, dtype=jnp.float32),
                    "belief_deal_bce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_score_ce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_call_bce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_tenpai_bce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_pos_rate": jnp.array(0.0, dtype=jnp.float32),
                    "belief_pred_mean": jnp.array(0.0, dtype=jnp.float32),
                }
                finite_all = jnp.bool_(True)

            # 2. Policy Update (Isolated chain tx_policy — skipped in Phase 1 warmup!)
            if update_policy:
                pol_in = dict(
                    planes=mb["planes"],
                    delta_u_ron_3opp=mb["delta_u_ron_3opp"],
                    call_threat_plane=mb["call_threat_plane"],
                    scalars=mb["scalars"],
                    river_tokens=mb["river_tokens"],
                    river_tile_ids=mb["river_tile_ids"].astype(jnp.int32),
                    river_valid_mask=mb["river_valid_mask"],
                    last_draw_type=mb["last_draw_type"].astype(jnp.int32),
                    is_last_draw_red=mb["is_last_draw_red"],
                    target_type=mb["target_type"].astype(jnp.int32),
                    target_is_red=mb["target_is_red"],
                    w_ego=mb["w_table_ego"][:, 0, :],
                    delta_u_agari_3=mb["delta_u_agari_3"],
                    delta_u_ryuukyoku=mb["delta_u_ryuukyoku"],
                    delta_u_ron_87=mb["delta_u_ron_87"],
                    legal_action_mask=mb["legal_mask"],
                )
                bc_logits = jax.lax.stop_gradient(mb["bc_logits"])
                log_rho = jax.lax.stop_gradient(mb["log_rho"])

                def pi_loss_fn(p):
                    logits, aux = policy_net.apply(p, return_aux=True, **pol_in)
                    loss, pm = compute_policy_loss_v6(
                        logits=logits,
                        actions=mb["action"],
                        old_log_probs=mb["old_log_prob"],
                        advantages_scaled=mb["adv_scaled"],
                        log_rho=log_rho,
                        bc_logits=bc_logits,
                        legal_mask=mb["legal_mask"],
                        learning_seat_mask=mb["is_learn_eff"],
                        cfg=cfg.rnad,
                        shield_lambda=shield_lam,
                    )
                    pm["beta_risk"] = aux["beta_risk"]
                    pm["beta_offense"] = aux["beta_offense"]
                    return loss, pm

                (_, pi_metrics), g_pi = jax.value_and_grad(pi_loss_fn, has_aux=True)(p_pi)
                if use_pmap:
                    g_pi = jax.lax.pmean(g_pi, axis_name="devices")
                u_pi, o_pi = tx_policy.update(g_pi, o_pi, p_pi)
                p_pi = optax.apply_updates(p_pi, u_pi)
                finite_all = finite_all & _tree_is_finite(g_pi)
            else:
                pi_metrics = {
                    "policy_loss": jnp.array(0.0, dtype=jnp.float32),
                    "ppo_loss": jnp.array(0.0, dtype=jnp.float32),
                    "rnad_kl": jnp.array(0.0, dtype=jnp.float32),
                    "bc_kl": jnp.array(0.0, dtype=jnp.float32),
                    "shield_loss": jnp.array(0.0, dtype=jnp.float32),
                    "approx_kl": jnp.array(0.0, dtype=jnp.float32),
                    "clip_frac": jnp.array(0.0, dtype=jnp.float32),
                    "policy_entropy": jnp.array(1.0, dtype=jnp.float32),
                    "active_frac": jnp.array(0.0, dtype=jnp.float32),
                    "p_riichi_legal": jnp.array(0.0, dtype=jnp.float32),
                    "p_riichi_sum": jnp.array(0.0, dtype=jnp.float32),
                    "n_riichi_legal": jnp.array(0.0, dtype=jnp.float32),
                    "p_call_legal": jnp.array(0.0, dtype=jnp.float32),
                    "p_call_sum": jnp.array(0.0, dtype=jnp.float32),
                    "n_call_legal": jnp.array(0.0, dtype=jnp.float32),
                    "beta_risk": jnp.array(0.0, dtype=jnp.float32),
                    "beta_offense": jnp.array(0.0, dtype=jnp.float32),
                }

            all_metrics = {**bel_metrics, **pi_metrics, "finite_grads": finite_all.astype(jnp.float32)}
            if use_pmap:
                all_metrics = jax.lax.pmean(all_metrics, axis_name="devices")
            return p_pi, o_pi, p_b_fast, o_bel, p_b_ema, all_metrics

        return jax.pmap(step_fn, axis_name="devices") if use_pmap else jax.jit(step_fn)

    update_bel_pi_warmup = None if cfg.rnad.freeze_belief else _make_bel_pi_step(update_policy=False)
    update_bel_pi_full = _make_bel_pi_step(update_policy=True)

    # =========================================================================
    # Main Training Loop (Phase 1 Warmup -> Phase 2 Sovereign V6 RL)
    # =========================================================================
    t_cal_arr = _rep(jnp.asarray(belief_t_cal, dtype=jnp.float32))

    # Replicate states across all GPUs if multi-GPU `pmap` is active
    params_pi = _rep(params_pi)
    opt_pi = _rep(opt_pi)
    params_oracle = _rep(params_oracle)
    opt_oracle = _rep(opt_oracle)
    params_b_fast = _rep(params_b_fast)
    opt_belief = _rep(opt_belief)
    params_b_ema = _rep(params_b_ema)
    params_grp_base_exec = _rep(params_grp_base)
    params_grp_ema_exec = _rep(params_grp_ema)
    params_bc_exec = _rep(params_bc)
    params_rho_base = _rep(params_rho_base)
    rms_state = _rep(rms_state)

    def _save_resume_ckpt(completed_upd: int) -> None:
        if checkpoint_path is None:
            return
        payload = {
            "completed_updates": int(completed_upd),
            "params_pi": _unrep(params_pi),
            "params_bc": _unrep(params_bc_exec),
            "params_rho_base": _unrep(params_rho_base),
            "params_oracle": _unrep(params_oracle),
            "params_b_fast": _unrep(params_b_fast),
            "params_b_ema": _unrep(params_b_ema),
            "opt_pi": _unrep(opt_pi),
            "opt_oracle": _unrep(opt_oracle),
            "opt_belief": _unrep(opt_belief),
            "rms_state": _unrep(rms_state),
            "pfsp_pool": list(pfsp_pool),
            "outer_snapshots": list(outer_snapshots),
            "history": list(history),
            "steps_since_outer_reset": int(steps_since_outer_reset),
            "healer_lr_scale": float(healer.current_lr_scale),
            "t_cal": float(belief_t_cal),
        }
        saved = _atomic_save_pickle(checkpoint_path, payload)
        if verbose:
            print(f"[Phase 1/2 RL] Saved resumable checkpoint at upd {completed_upd} -> {saved}", flush=True)
        if on_checkpoint is not None:
            on_checkpoint(completed_upd, saved)

    # Initial healthy checkpoint before updates (stored un-replicated on host)
    healer.save_checkpoint(
        start_update,
        {
            "params_pi": _unrep(params_pi),
            "params_rho_base": _unrep(params_rho_base),
            "opt_pi": _unrep(opt_pi),
            "params_oracle": _unrep(params_oracle),
            "opt_oracle": _unrep(opt_oracle),
            "params_b_fast": _unrep(params_b_fast),
            "opt_belief": _unrep(opt_belief),
            "params_b_ema": _unrep(params_b_ema),
            "rms_state": _unrep(rms_state),
        },
    )

    if start_update >= cfg.num_updates:
        if verbose:
            print(
                f"[Phase 1/2 RL] Already reached target updates ({start_update}/{cfg.num_updates}); skipping loop.",
                flush=True,
            )
    elif verbose:
        print(
            f"[Phase 1/2 RL] devices={n_devices} ({'pmap' if use_pmap else 'jit'}) | "
            f"envs={cfg.num_envs} x T={cfg.num_steps} ({n_total:,} steps/upd) | "
            f"updates={start_update}->{cfg.num_updates} (warmup={cfg.rnad.warmup_updates}) | "
            f"freeze_belief={cfg.rnad.freeze_belief} | lambda_intra={cfg.rnad.lambda_intra} | "
            f"Compiling rollout + update XLA graphs on first update...",
            flush=True,
        )
    t_rl_start = time.perf_counter()
    last_eff_lr: Optional[float] = None
    cached_opp_key: Optional[int] = None
    cached_opp_host: Any = None
    cached_opp_exec: Any = None
    warmup_unlocked = start_update >= int(cfg.rnad.warmup_updates)
    extra_warmup_used = 0
    max_extra_warmup = 25
    phase2_step = max(0, start_update - int(cfg.rnad.warmup_updates))
    aborted = False

    ORA_BUF_KEYS = (
        "planes",
        "oracle_planes",
        "scalars",
        "w_table_ego",
        "grp_logits_base",
        "current_player",
        "river_spatial_feat",
        "targets_4x4",
        "delta_u_target_ego",
        "win_ego",
        "score_delta_ego",
        "aux_valid",
        "round_valid",
    )
    ORA_FP16_KEYS = frozenset(("planes", "oracle_planes", "river_spatial_feat"))
    ora_src_keys = (
        "planes",
        "oracle_planes",
        "scalars",
        "w_table_ego",
        "grp_logits_base",
        "current_player",
        "targets_4x4",
        "delta_u_target_ego",
        "win_ego",
        "score_delta_ego",
        "aux_valid",
        "round_valid",
    ) + (
        ("river_spatial_feat",)
        if cfg.rnad.freeze_belief
        else ("river_tokens", "river_tile_ids", "river_valid_mask", "w_opp_5d")
    )
    use_ora_replay = bool(cfg.rnad.freeze_belief and int(cfg.oracle.replay_updates) > 1)
    ora_replay: Deque[Dict[str, np.ndarray]] = deque(maxlen=max(1, int(cfg.oracle.replay_updates)))
    n_mb_per_epoch = max(1, n_per_dev // mb_per_dev)
    ora_update_epochs = max(1, int(cfg.oracle.update_epochs))

    for upd in range(start_update, cfg.num_updates):
        t_upd_start = time.perf_counter()
        in_warmup = not warmup_unlocked

        # Outer R-NaD hard reset at every `outer_reset_steps` (when > 0) or healthy PFSP snapshotting (when <= 0)
        if (
            (not in_warmup)
            and int(cfg.rnad.outer_reset_steps) > 0
            and phase2_step > 0
            and (phase2_step % int(cfg.rnad.outer_reset_steps) == 0)
        ):
            snap = _clone_host_params(_unrep(params_pi))
            pfsp_pool.append(snap)
            outer_snapshots.append(snap)
            params_rho_base = _rep(jax.device_put(snap))
            # Surgical AdamW reset: zero ONLY m_t, preserve 100% of v_t!
            opt_pi = _rep(reset_adam_first_moment_only(_unrep(opt_pi)))
            steps_since_outer_reset = 0
            last_eff_lr = None
        elif (
            (not in_warmup)
            and int(cfg.rnad.outer_reset_steps) <= 0
            and int(cfg.rnad.pfsp_snapshot_interval) > 0
            and phase2_step > 0
            and (phase2_step % int(cfg.rnad.pfsp_snapshot_interval) == 0)
            and history
            and history[-1].get("healed", 0.0) < 0.5
            and healer.is_snapshot_safe(history[-1])
        ):
            snap = _clone_host_params(_unrep(params_pi))
            pfsp_pool.append(snap)
            outer_snapshots.append(snap)

        # Apply smooth post-reset & Phase-2-entry warmup + self-healing LR scaling (zero-copy!)
        if in_warmup:
            frac = 1.0
        elif steps_since_outer_reset < int(cfg.rnad.reset_warmup_steps):
            frac = 0.1 + 0.9 * (steps_since_outer_reset + 1) / max(int(cfg.rnad.reset_warmup_steps), 1)
            steps_since_outer_reset += 1
        else:
            frac = 1.0
        eff_lr = float(cfg.rnad.policy_lr) * healer.current_lr_scale * frac
        if last_eff_lr is None or abs(eff_lr - last_eff_lr) > 1e-12:
            opt_pi = set_injected_lr(opt_pi, eff_lr, n_devices=n_devices if use_pmap else 1)
            last_eff_lr = eff_lr

        # Per-Rollout Single-Snapshot PFSP sampling (50% permanent BC anchor + cached device weights)
        if len(pfsp_pool) > 1 and np_rng.random() < 0.50:
            params_opp = params_bc_exec
        else:
            opp_idx = int(np_rng.integers(0, len(pfsp_pool)))
            chosen_opp = pfsp_pool[opp_idx]
            if cached_opp_key != id(chosen_opp):
                cached_opp_host = chosen_opp
                cached_opp_key = id(chosen_opp)
                cached_opp_exec = _rep(jax.device_put(chosen_opp))
            params_opp = cached_opp_exec

        rng, k_roll = jax.random.split(rng)
        k_roll_exec = shard_for_pmap(jax.random.split(k_roll, n_devices), n_devices) if use_pmap else k_roll
        env_state, traj_tb, v_boot_b = rollout_fn(
            env_state,
            params_pi,
            params_opp,
            params_b_ema,
            params_grp_base_exec,
            params_grp_ema_exec,
            params_oracle,
            w_tables_exec,
            learn_mask_exec,
            t_cal_arr,
            k_roll_exec,
        )

        # Structural deck uniqueness probe
        raw_decks = np.asarray(jax.device_get(env_state.round_state.deck))
        decks_sig = raw_decks.reshape((cfg.num_envs, -1))[:, :8]
        distinct_ratio = len({row.tobytes() for row in decks_sig}) / max(cfg.num_envs, 1)

        prepare_fn = prepare_warmup_fn if in_warmup else prepare_full_fn
        (
            flat_batch,
            rms_state,
            rms_val,
            riichi_adv_gap_arr,
            riichi_pg_arr,
            oracle_r2_arr,
            oracle_r2_late_arr,
            adv_mean_arr,
            comp_frac_arr,
            bel_pred_mean_arr,
            ron_overflow_flag,
        ) = prepare_fn(
            traj_tb, v_boot_b, rms_state, params_bc_exec, params_rho_base
        )
        ron_overflow_host = bool(np.asarray(jax.device_get(_unrep(ron_overflow_flag))))
        if ron_overflow_host and verbose:
            print(
                f"[RL WARNING] upd {upd + 1}: ron_k_max overflowed — skipped BeliefNet update to prevent poisoning!",
                flush=True,
            )

        phase2_total = max(int(cfg.num_updates) - int(cfg.rnad.warmup_updates) - 1, 1)
        progress = 0.0 if in_warmup else min(1.0, float(phase2_step) / float(phase2_total))
        shield_lam_val = compute_shield_lambda(progress, cfg.rnad)
        shield_lam = _rep(jnp.asarray(shield_lam_val, dtype=jnp.float32))

        upd_metrics_acc: Dict[str, List[float]] = {}
        all_finite = True

        # 1. Decoupled Oracle Update (host-RAM sliding replay when freeze_belief & replay_updates > 1)
        if use_ora_replay:
            rv_raw = np.asarray(jax.device_get(flat_batch["round_valid"]), dtype=bool)
            rv_np = rv_raw.reshape((n_devices * n_per_dev,)) if use_pmap else rv_raw.reshape((n_per_dev,))
            if bool(np.any(rv_np)):
                chunk_host: Dict[str, np.ndarray] = {}
                for k in ORA_BUF_KEYS:
                    a_np = np.asarray(jax.device_get(flat_batch[k]))
                    if use_pmap:
                        a_np = a_np.reshape((n_devices * n_per_dev,) + a_np.shape[2:])
                    a_val = a_np[rv_np]
                    if k in ORA_FP16_KEYS:
                        a_val = a_val.astype(np.float16, copy=False)
                    chunk_host[k] = a_val
                ora_replay.append(chunk_host)

            if ora_replay:
                chunks_list = list(ora_replay)
                lengths = np.array([len(c["round_valid"]) for c in chunks_list], dtype=np.int64)
                cum = np.cumsum(lengths)
                starts = np.empty_like(cum)
                starts[0] = 0
                starts[1:] = cum[:-1]
                total_rows = int(cum[-1])
                mb_total = mb_per_dev * (n_devices if use_pmap else 1)
                n_ora_mbs = ora_update_epochs * n_mb_per_epoch
                for _ in range(n_ora_mbs):
                    g_idx = np_rng.integers(0, total_rows, size=mb_total)
                    chunk_ids = np.searchsorted(cum, g_idx, side="right")
                    local_ids = g_idx - starts[chunk_ids]
                    order_masks = [
                        (cid, np.flatnonzero(chunk_ids == cid))
                        for cid in range(len(chunks_list))
                        if np.any(chunk_ids == cid)
                    ]
                    mb_ora_host: Dict[str, np.ndarray] = {}
                    for k in ORA_BUF_KEYS:
                        sample_leaf = chunks_list[0][k]
                        buf = np.empty((mb_total,) + sample_leaf.shape[1:], dtype=sample_leaf.dtype)
                        for cid, pos in order_masks:
                            buf[pos] = chunks_list[cid][k][local_ids[pos]]
                        if use_pmap:
                            buf = buf.reshape((n_devices, mb_per_dev) + buf.shape[1:])
                        mb_ora_host[k] = buf
                    mb_ora = (
                        shard_for_pmap(mb_ora_host, n_devices)
                        if use_pmap
                        else jax.tree_util.tree_map(jax.device_put, mb_ora_host)
                    )
                    params_oracle, opt_oracle, m_ora_dict = update_ora_mb(
                        params_oracle, opt_oracle, params_b_ema, mb_ora, t_cal_arr
                    )
                    m_ora_single = _unrep(m_ora_dict)
                    if float(m_ora_single["ora_finite_grads"]) < 0.5:
                        all_finite = False
                    for mk, mv in m_ora_single.items():
                        upd_metrics_acc.setdefault(mk, []).append(float(mv))
        else:
            for _ in range(ora_update_epochs):
                perm_o = np_rng.permutation(n_per_dev)
                for start in range(0, n_per_dev, mb_per_dev):
                    idx_np = perm_o[start : start + mb_per_dev]
                    if idx_np.size < mb_per_dev:
                        continue
                    idx_j = jnp.asarray(idx_np, dtype=jnp.int32)
                    if use_pmap:
                        mb_ora = {k: flat_batch[k][:, idx_j] for k in ora_src_keys}
                    else:
                        mb_ora = {k: flat_batch[k][idx_j] for k in ora_src_keys}
                    params_oracle, opt_oracle, m_ora_dict = update_ora_mb(
                        params_oracle, opt_oracle, params_b_ema, mb_ora, t_cal_arr
                    )
                    m_ora_single = _unrep(m_ora_dict)
                    if float(m_ora_single["ora_finite_grads"]) < 0.5:
                        all_finite = False
                    for mk, mv in m_ora_single.items():
                        upd_metrics_acc.setdefault(mk, []).append(float(mv))

        # 2. Strictly On-Policy Belief / Policy Update (skipped during warmup when freeze_belief=True)
        bel_pi_step_fn = update_bel_pi_warmup if in_warmup else update_bel_pi_full
        if bel_pi_step_fn is not None:
            for _ in range(cfg.update_epochs):
                perm = np_rng.permutation(n_per_dev)
                for start in range(0, n_per_dev, mb_per_dev):
                    idx_np = perm[start : start + mb_per_dev]
                    if idx_np.size < mb_per_dev:
                        continue
                    idx_j = jnp.asarray(idx_np, dtype=jnp.int32)
                    if use_pmap:
                        mb = jax.tree_util.tree_map(lambda x: x[:, idx_j], flat_batch)
                    else:
                        mb = jax.tree_util.tree_map(lambda x: x[idx_j], flat_batch)
                    (
                        params_pi,
                        opt_pi,
                        params_b_fast,
                        opt_belief,
                        params_b_ema,
                        m_dict,
                    ) = bel_pi_step_fn(
                        params_pi,
                        opt_pi,
                        params_b_fast,
                        opt_belief,
                        params_b_ema,
                        mb,
                        shield_lam,
                        ron_overflow_flag,
                    )
                    m_single = _unrep(m_dict)
                    if float(m_single["finite_grads"]) < 0.5:
                        all_finite = False
                    for mk, mv in m_single.items():
                        upd_metrics_acc.setdefault(mk, []).append(float(mv))
        else:
            for mk, mv in (
                ("belief_loss", 0.0),
                ("belief_deal_bce", 0.0),
                ("belief_score_ce", 0.0),
                ("belief_call_bce", 0.0),
                ("belief_tenpai_bce", 0.0),
                ("belief_pos_rate", 0.0),
                ("belief_pred_mean", 0.0),
                ("policy_loss", 0.0),
                ("ppo_loss", 0.0),
                ("rnad_kl", 0.0),
                ("bc_kl", 0.0),
                ("shield_loss", 0.0),
                ("approx_kl", 0.0),
                ("clip_frac", 0.0),
                ("policy_entropy", 1.0),
                ("active_frac", 0.0),
                ("p_riichi_legal", 0.0),
                ("p_riichi_sum", 0.0),
                ("n_riichi_legal", 0.0),
                ("p_call_legal", 0.0),
                ("p_call_sum", 0.0),
                ("n_call_legal", 0.0),
                ("beta_risk", 0.0),
                ("beta_offense", 0.0),
            ):
                upd_metrics_acc[mk] = [mv]

        jax.block_until_ready((params_pi, params_oracle))
        upd_wall_s = max(time.perf_counter() - t_upd_start, 1e-6)
        summary = {
            "update": float(upd),
            "in_warmup": float(in_warmup),
            "adv_rms": float(_unrep(rms_val)),
            "adv_mean": float(np.asarray(jax.device_get(_unrep(adv_mean_arr)))),
            "oracle_r2": float(np.asarray(jax.device_get(_unrep(oracle_r2_arr)))),
            "oracle_r2_late": float(np.asarray(jax.device_get(_unrep(oracle_r2_late_arr)))),
            "completed_round_frac": float(np.asarray(jax.device_get(_unrep(comp_frac_arr)))),
            "riichi_adv_gap": float(np.asarray(jax.device_get(_unrep(riichi_adv_gap_arr)))),
            "riichi_pg": float(np.asarray(jax.device_get(_unrep(riichi_pg_arr)))),
            "ron_overflow": float(ron_overflow_host),
            "eff_policy_lr": float(eff_lr),
            "shield_lambda": float(shield_lam_val),
            "distinct_deck_ratio": float(distinct_ratio),
            "update_wall_s": float(upd_wall_s),
            "sps": float(n_total / upd_wall_s),
        }
        for mk, vals in upd_metrics_acc.items():
            summary[mk] = float(np.mean(vals)) if vals else 0.0
        summary["finite_grads"] = 1.0 if all_finite else 0.0
        if cfg.rnad.freeze_belief:
            summary["belief_pred_mean"] = float(np.asarray(jax.device_get(_unrep(bel_pred_mean_arr))))
        # Exact weighted average across minibatches (avoids 0-dilution on sparse Riichi/Call states)
        sum_n_rii = float(np.sum(upd_metrics_acc.get("n_riichi_legal", [0.0])))
        sum_p_rii = float(np.sum(upd_metrics_acc.get("p_riichi_sum", [0.0])))
        epoch_scale = float(n_devices) / max(float(cfg.update_epochs), 1.0)
        summary["p_riichi_legal"] = (sum_p_rii / sum_n_rii) if sum_n_rii > 0.0 else 0.0
        summary["p_riichi_sum"] = sum_p_rii * epoch_scale
        summary["n_riichi_legal"] = sum_n_rii * epoch_scale
        sum_n_call = float(np.sum(upd_metrics_acc.get("n_call_legal", [0.0])))
        sum_p_call = float(np.sum(upd_metrics_acc.get("p_call_sum", [0.0])))
        summary["p_call_legal"] = (sum_p_call / sum_n_call) if sum_n_call > 0.0 else 0.0
        summary["p_call_sum"] = sum_p_call * epoch_scale
        summary["n_call_legal"] = sum_n_call * epoch_scale

        # Oracle readiness gate before unlocking Phase 2 + clear warmup noise from RunningRMS
        if in_warmup and ((upd + 1) >= int(cfg.rnad.warmup_updates)):
            ora_ready = (
                all_finite
                and np.isfinite(summary.get("oracle_rank_ce", float("nan")))
                and float(summary.get("oracle_birkhoff_err", 1.0)) <= float(cfg.rnad.max_birkhoff_err)
            )
            if ora_ready:
                warmup_unlocked = True
                rms_state = _rep(init_running_rms(1.0))
                if enable_self_healing:
                    healer.save_checkpoint(
                        upd + 1,
                        {
                            "params_pi": _unrep(params_pi),
                            "params_rho_base": _unrep(params_rho_base),
                            "opt_pi": _unrep(opt_pi),
                            "params_oracle": _unrep(params_oracle),
                            "opt_oracle": _unrep(opt_oracle),
                            "params_b_fast": _unrep(params_b_fast),
                            "opt_belief": _unrep(opt_belief),
                            "params_b_ema": _unrep(params_b_ema),
                            "rms_state": _unrep(rms_state),
                        },
                    )
            elif extra_warmup_used < max_extra_warmup:
                extra_warmup_used += 1
                if verbose:
                    print(
                        f"[RL WARMUP-GATE] Extending warmup ({extra_warmup_used}/{max_extra_warmup}) at upd {upd + 1}: "
                        f"finite={all_finite}, birkhoff_err={summary.get('oracle_birkhoff_err', 1.0):.5f}",
                        flush=True,
                    )
            else:
                raise RuntimeError(
                    f"Phase 1 warmup readiness gate failed after {max_extra_warmup} extra updates: "
                    f"finite={all_finite}, birkhoff_err={summary.get('oracle_birkhoff_err', 1.0):.5f}."
                )

        # Self-healing probe evaluation
        heal_reasons_str = ""
        health_action = "none"
        if enable_self_healing and (not in_warmup):
            report = healer.evaluate_probes(
                summary,
                finite_grads=all_finite,
                distinct_deck_ratio=distinct_ratio,
            )
            health_action = report.action_taken
            heal_reasons_str = ", ".join(report.reasons)
            summary["healed"] = 1.0 if report.action_taken in ("rollback_and_halve_lr", "abort") else 0.0
            if report.action_taken in ("rollback_and_halve_lr", "abort"):
                ora_replay.clear()
                restored = healer.restore_latest()
                if restored is not None:
                    params_pi = _rep(restored["params_pi"])
                    if "params_rho_base" in restored:
                        params_rho_base = _rep(restored["params_rho_base"])
                    opt_pi = _rep(reset_adam_first_moment_only(restored["opt_pi"]))
                    params_oracle = _rep(restored["params_oracle"])
                    opt_oracle = _rep(restored["opt_oracle"])
                    params_b_fast = _rep(restored["params_b_fast"])
                    opt_belief = _rep(restored["opt_belief"])
                    params_b_ema = _rep(restored["params_b_ema"])
                    rms_state = _rep(restored["rms_state"])
                    last_eff_lr = None
                if report.action_taken == "abort":
                    aborted = True
            elif report.healthy and healer.is_snapshot_safe(summary) and ((upd + 1) % max(int(cfg.rnad.audit_interval), 1) == 0):
                healer.save_checkpoint(
                    upd + 1,
                    {
                        "params_pi": _unrep(params_pi),
                        "params_rho_base": _unrep(params_rho_base),
                        "opt_pi": _unrep(opt_pi),
                        "params_oracle": _unrep(params_oracle),
                        "opt_oracle": _unrep(opt_oracle),
                        "params_b_fast": _unrep(params_b_fast),
                        "opt_belief": _unrep(opt_belief),
                        "params_b_ema": _unrep(params_b_ema),
                        "rms_state": _unrep(rms_state),
                    },
                )
        if not in_warmup and (not aborted):
            phase2_step += 1
        history.append(summary)
        if verbose and (
            upd == start_update
            or (not in_warmup and phase2_step == 1)
            or (upd + 1) % max(int(log_every), 1) == 0
            or (upd + 1) == cfg.num_updates
            or health_action != "none"
        ):
            phase_tag = "WARMUP" if in_warmup else "PHASE2"
            elapsed_rl = max(time.perf_counter() - t_rl_start, 1e-6)
            done_this_run = max(upd - start_update + 1, 1)
            eta_s = (elapsed_rl / done_this_run) * (cfg.num_updates - upd - 1)
            if health_action == "rollback_and_halve_lr":
                heal_str = f" [SELF-HEALED! lr_scale={healer.current_lr_scale:.3g} | {heal_reasons_str}]"
            elif health_action == "abort":
                heal_str = f" [ABORTED(max_heals)! restored healthy ckpt | {heal_reasons_str}]"
            else:
                heal_str = ""
            print(
                f"[RL {phase_tag}] upd {upd + 1}/{cfg.num_updates} ({upd_wall_s:.1f}s, {summary['sps']:,.0f} sps) | "
                f"ora_ce={summary.get('oracle_rank_ce', 0.0):.3f} r2={summary.get('oracle_r2', 0.0):.3f}({summary.get('oracle_r2_late', 0.0):.2f}) "
                f"adv_m={summary.get('adv_mean', 0.0):+.4f} ent={summary.get('policy_entropy', 0.0):.3f} "
                f"kl={summary.get('approx_kl', 0.0):.4f} bc_kl={summary.get('bc_kl', 0.0):.4f} "
                f"p_rii={summary.get('p_riichi_legal', 0.0):.3f}(n={int(summary.get('n_riichi_legal', 0.0))}) "
                f"rii_pg={summary.get('riichi_pg', 0.0):+.4f} p_call={summary.get('p_call_legal', 0.0):.3f} "
                f"b_r={summary.get('beta_risk', 0.0):.3f} b_o={summary.get('beta_offense', 0.0):.3f} | "
                f"ETA={eta_s/60:.1f}m{heal_str}",
                flush=True,
            )

        if aborted:
            if checkpoint_path is not None:
                _save_resume_ckpt(upd + 1)
            break

        if checkpoint_path is not None and checkpoint_every > 0 and ((upd + 1) % int(checkpoint_every) == 0):
            _save_resume_ckpt(upd + 1)

    if (not aborted) and checkpoint_path is not None and start_update < cfg.num_updates:
        if checkpoint_every <= 0 or (cfg.num_updates % int(checkpoint_every) != 0):
            _save_resume_ckpt(cfg.num_updates)

    # Unreplicate final weights onto device 0 / host cleanly
    final_pi = jax.device_put(jax.device_get(_unrep(params_pi)), jax.local_devices()[0])
    final_b_ema = jax.device_put(jax.device_get(_unrep(params_b_ema)), jax.local_devices()[0])
    final_b_fast = jax.device_put(jax.device_get(_unrep(params_b_fast)), jax.local_devices()[0])
    final_ora = jax.device_put(jax.device_get(_unrep(params_oracle)), jax.local_devices()[0])
    if (not aborted) and start_update < cfg.num_updates and (phase2_step > 0 or not outer_snapshots):
        outer_snapshots.append(_clone_host_params(final_pi))

    return V6RLResult(
        policy_model=policy_net,
        policy_params=final_pi,
        belief_model=belief_net,
        belief_ema_params=final_b_ema,
        belief_fast_params=final_b_fast,
        belief_t_cal=belief_t_cal,
        grp_model=grp_net,
        grp_base_params=params_grp_base,
        oracle_model=oracle_net,
        oracle_params=final_ora,
        outer_snapshots=list(outer_snapshots),
        history=history,
    )

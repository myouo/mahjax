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
        tau_prior=cfg.grp.tau_prior,
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

    w_tables, is_pfsp_env, learning_seat_mask = sample_meta_table_pool(cfg.num_envs, k_pool)
    if use_pmap:
        # Interleave Hom/PFSP/Het regimes evenly across GPUs before sharding (prevents 2x seat imbalance!)
        interleave_idx = jnp.concatenate([jnp.arange(d, cfg.num_envs, n_devices) for d in range(n_devices)])
        w_tables = w_tables[interleave_idx]
        is_pfsp_env = is_pfsp_env[interleave_idx]
        learning_seat_mask = learning_seat_mask[interleave_idx]

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

        def eval_single_env_step(st, w_abs, l_mask_4, step_rng):
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
            _, grp_logits_base = grp_net.apply(p_grp_base, grp_f, return_logits=True)

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
            logits_opp = policy_net.apply(p_opp, **pol_kwargs)

            is_learn = l_mask_4[cp]
            active_logits = jnp.where(is_learn, logits_pi, logits_opp)
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
                "grp_ema_next_4x4": grp_ema_next_4x4,
                "v_oracle_4x4": ora_out["probs_4x4"],
                "term_rank_4x4": term_rank_4x4,
                "w_table_ego": w_ego_4x4,
                "w_opp_5d": w_opp_5d,
                "action": action,
                "old_log_prob": old_log_p,
                "legal_mask": legal_mask,
                "current_player": cp.astype(jnp.int8),
                "is_learn": is_learn,
                "done": done,
                "round_ended": round_ended,
                "scores_curr": scores_curr,
                "scores_post": scores_post,
                "step_winner_abs": step_winner_abs.astype(jnp.int8),
            }
            if rsnap is not None:
                trans["rsnap"] = rsnap
            return next_st, trans

        def scan_step(carry, _):
            st_b, r = carry
            r, k_sub = jax.random.split(r)
            step_keys = jax.random.split(k_sub, envs_per_dev)
            next_st_b, trans_b = jax.vmap(eval_single_env_step)(st_b, w_tab, learn_mask, step_keys)
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

    # =========================================================================
    # JIT/PMAP 2: Dual-Rate Simplex TD + Running-RMS + Sparse K_max Ron Labeling
    # =========================================================================
    def _prepare_single_dev(traj_tb, v_boot_b, rms_st):
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

        # Ego advantage: <(target - V_oracle)[cp], w_me>
        b_idx = jnp.arange(envs_per_dev, dtype=jnp.int32)[:, None]
        t_idx = jnp.arange(cfg.num_steps, dtype=jnp.int32)[None, :]
        cp_bt = traj_bt["current_player"].astype(jnp.int32)
        p_diff_me = targets_4x4[b_idx, t_idx, cp_bt] - traj_bt["v_oracle_4x4"][b_idx, t_idx, cp_bt]
        w_me_bt = traj_bt["w_table_ego"][:, :, 0, :]
        raw_adv = jnp.sum(p_diff_me * w_me_bt, axis=-1)

        num_legal = jnp.sum(traj_bt["legal_mask"].astype(jnp.int32), axis=-1)
        active_mask = traj_bt["is_learn"] & (num_legal >= 2)
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

        win_ego, score_delta_ego, aux_valid = compute_round_outcome_targets(
            scores_curr=traj_bt["scores_curr"],
            scores_post=traj_bt["scores_post"],
            round_ended=traj_bt["round_ended"],
            dones=traj_bt["done"],
            step_winner_abs=traj_bt["step_winner_abs"],
            current_players=cp_bt,
        )

        # Measure exact advantage gap between Action.RIICHI and dama/discard on riichi-legal states
        can_r = active_mask & traj_bt["legal_mask"][..., Action.RIICHI]
        did_r = can_r & (traj_bt["action"] == Action.RIICHI)
        did_dama = can_r & (traj_bt["action"] != Action.RIICHI)
        sum_r = jnp.sum(jnp.where(did_r, raw_adv, 0.0))
        cnt_r = jnp.sum(did_r.astype(jnp.float32))
        sum_d = jnp.sum(jnp.where(did_dama, raw_adv, 0.0))
        cnt_d = jnp.sum(did_dama.astype(jnp.float32))
        if use_pmap:
            sum_r = jax.lax.psum(sum_r, axis_name="devices")
            cnt_r = jax.lax.psum(cnt_r, axis_name="devices")
            sum_d = jax.lax.psum(sum_d, axis_name="devices")
            cnt_d = jax.lax.psum(cnt_d, axis_name="devices")
        has_both_r = (cnt_r > 0.5) & (cnt_d > 0.5)
        riichi_adv_gap = jnp.where(
            has_both_r,
            (sum_r / jnp.maximum(cnt_r, 1.0)) - (sum_d / jnp.maximum(cnt_d, 1.0)),
            jnp.array(0.0, dtype=jnp.float32),
        )

        # Flatten (envs_per_dev, T, ...) -> (n_per_dev, ...)
        n_per_dev = envs_per_dev * cfg.num_steps
        flat = jax.tree_util.tree_map(lambda x: x.reshape((n_per_dev,) + x.shape[2:]), traj_bt)
        flat["targets_4x4"] = targets_4x4.reshape((n_per_dev, 4, 4))
        flat["adv_scaled"] = scaled_adv.reshape((n_per_dev,))
        flat["win_ego"] = win_ego.reshape((n_per_dev,))
        flat["score_delta_ego"] = score_delta_ego.reshape((n_per_dev, 4))
        flat["aux_valid"] = aux_valid.reshape((n_per_dev,))

        if cfg.rnad.freeze_belief:
            ron_overflow = jnp.bool_(False)
        else:
            eff_ron_k_max = max(int(cfg.rnad.ron_k_max), 2 * n_per_dev)
            y_ron, y_sb, y_ct, y_ten, ron_overflow = compute_sparse_belief_labels(
                flat["rsnap"], k_max=eff_ron_k_max
            )
            if use_pmap:
                ron_overflow = jax.lax.pmax(ron_overflow.astype(jnp.int32), axis_name="devices") > 0
            flat["y_ron"] = y_ron
            flat["y_score_bin"] = y_sb
            flat["y_call_threat"] = y_ct
            flat["y_tenpai"] = y_ten
            del flat["rsnap"]
        return flat, new_rms, rms_val, riichi_adv_gap, ron_overflow

    prepare_batch_fn = (
        jax.pmap(_prepare_single_dev, axis_name="devices") if use_pmap else jax.jit(_prepare_single_dev)
    )

    # =========================================================================
    # JIT/PMAP 3: Isolated 3-Chain Minibatch Update (Warmup vs Full RL)
    # =========================================================================
    def _make_update_step(update_policy: bool):
        def step_fn(
            p_pi,
            o_pi,
            p_ora,
            o_ora,
            p_b_fast,
            o_bel,
            p_b_ema,
            p_bc,
            p_rho_base,
            mb,
            shield_lam,
            t_cal,
            ron_overflow,
        ):
            b_ema_out = belief_net.apply(
                p_b_ema,
                planes=mb["planes"],
                scalars=mb["scalars"],
                river_tokens=mb["river_tokens"],
                river_tile_ids=mb["river_tile_ids"].astype(jnp.int32),
                river_valid_mask=mb["river_valid_mask"],
                w_opp_5d=mb["w_opp_5d"],
                temperature_cal=t_cal,
            )
            river_feat = jax.lax.stop_gradient(b_ema_out["river_spatial_feat"])

            # 1. Oracle Update (Isolated chain tx_oracle)
            def ora_loss_fn(p):
                out = oracle_net.apply(
                    p,
                    planes=mb["planes"],
                    oracle_planes=mb["oracle_planes"],
                    scalars=mb["scalars"],
                    w_table_ego=mb["w_table_ego"],
                    grp_logits_abs=mb["grp_logits_base"],
                    current_player=mb["current_player"].astype(jnp.int32),
                    river_spatial_feat=river_feat,
                )
                return compute_oracle_loss_v6(
                    out,
                    targets_4x4=mb["targets_4x4"],
                    round_winner_ego=mb["win_ego"],
                    round_score_delta_ego=mb["score_delta_ego"],
                    aux_valid=mb["aux_valid"],
                    cfg=cfg.oracle,
                )

            (_, ora_metrics), g_ora = jax.value_and_grad(ora_loss_fn, has_aux=True)(p_ora)
            if use_pmap:
                g_ora = jax.lax.pmean(g_ora, axis_name="devices")
            u_ora, o_ora = tx_oracle.update(g_ora, o_ora, p_ora)
            p_ora = optax.apply_updates(p_ora, u_ora)

            # 2. Belief Update (frozen by default; if enabled, guarded against ron_k_max overflow)
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
                finite_all = _tree_is_finite(g_ora) & (ron_overflow | _tree_is_finite(g_bel))
            else:
                bel_metrics = {
                    "belief_loss": jnp.array(0.0, dtype=jnp.float32),
                    "belief_deal_bce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_score_ce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_call_bce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_tenpai_bce": jnp.array(0.0, dtype=jnp.float32),
                    "belief_pos_rate": jnp.array(0.0, dtype=jnp.float32),
                    "belief_pred_mean": jnp.mean(b_ema_out["p_deal"]),
                }
                finite_all = _tree_is_finite(g_ora)

            # 3. Policy Update (Isolated chain tx_policy — skipped in Phase 1 warmup!)
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
                bc_logits = jax.lax.stop_gradient(policy_net.apply(p_bc, **pol_in))
                rho_base_logits = jax.lax.stop_gradient(policy_net.apply(p_rho_base, **pol_in))
                _, log_rho = compute_dual_anchor_rho_probs(
                    rho_base_logits,
                    bc_logits,
                    mb["legal_mask"],
                    mix_bc=cfg.rnad.rho_mix_bc,
                    mix_uniform=cfg.rnad.rho_mix_uniform,
                )

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
                        learning_seat_mask=mb["is_learn"],
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

            all_metrics = {**ora_metrics, **bel_metrics, **pi_metrics, "finite_grads": finite_all.astype(jnp.float32)}
            if use_pmap:
                all_metrics = jax.lax.pmean(all_metrics, axis_name="devices")
            return p_pi, o_pi, p_ora, o_ora, p_b_fast, o_bel, p_b_ema, all_metrics

        return jax.pmap(step_fn, axis_name="devices") if use_pmap else jax.jit(step_fn)

    update_mb_warmup = _make_update_step(update_policy=False)
    update_mb_full = _make_update_step(update_policy=True)

    # =========================================================================
    # Main Training Loop (Phase 1 Warmup -> Phase 2 Sovereign V6 RL)
    # =========================================================================
    n_total = cfg.num_envs * cfg.num_steps
    n_per_dev = envs_per_dev * cfg.num_steps
    mb_per_dev = min(cfg.minibatch_size // n_devices if use_pmap else cfg.minibatch_size, n_per_dev)
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

    for upd in range(start_update, cfg.num_updates):
        t_upd_start = time.perf_counter()
        in_warmup = not warmup_unlocked

        # Outer R-NaD hard reset at every `outer_reset_steps` in Phase 2
        if (not in_warmup) and phase2_step > 0 and (phase2_step % int(cfg.rnad.outer_reset_steps) == 0):
            snap = _clone_host_params(_unrep(params_pi))
            pfsp_pool.append(snap)
            outer_snapshots.append(snap)
            params_rho_base = _rep(jax.device_put(snap))
            # Surgical AdamW reset: zero ONLY m_t, preserve 100% of v_t!
            opt_pi = _rep(reset_adam_first_moment_only(_unrep(opt_pi)))
            steps_since_outer_reset = 0
            last_eff_lr = None

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

        # Per-Rollout Single-Snapshot PFSP sampling (25% permanent BC anchor + cached device weights)
        if len(pfsp_pool) > 1 and np_rng.random() < 0.25:
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

        flat_batch, rms_state, rms_val, riichi_adv_gap_arr, ron_overflow_flag = prepare_batch_fn(
            traj_tb, v_boot_b, rms_state
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

        step_fn = update_mb_warmup if in_warmup else update_mb_full
        upd_metrics_acc: Dict[str, List[float]] = {}
        all_finite = True

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
                    params_oracle,
                    opt_oracle,
                    params_b_fast,
                    opt_belief,
                    params_b_ema,
                    m_dict,
                ) = step_fn(
                    params_pi,
                    opt_pi,
                    params_oracle,
                    opt_oracle,
                    params_b_fast,
                    opt_belief,
                    params_b_ema,
                    params_bc_exec,
                    params_rho_base,
                    mb,
                    shield_lam,
                    t_cal_arr,
                    ron_overflow_flag,
                )
                m_single = _unrep(m_dict)
                if float(m_single["finite_grads"]) < 0.5:
                    all_finite = False
                for mk, mv in m_single.items():
                    upd_metrics_acc.setdefault(mk, []).append(float(mv))

        jax.block_until_ready(params_pi)
        upd_wall_s = max(time.perf_counter() - t_upd_start, 1e-6)
        summary = {
            "update": float(upd),
            "in_warmup": float(in_warmup),
            "adv_rms": float(_unrep(rms_val)),
            "riichi_adv_gap": float(np.asarray(jax.device_get(_unrep(riichi_adv_gap_arr)))),
            "ron_overflow": float(ron_overflow_host),
            "eff_policy_lr": float(eff_lr),
            "shield_lambda": float(shield_lam_val),
            "distinct_deck_ratio": float(distinct_ratio),
            "update_wall_s": float(upd_wall_s),
            "sps": float(n_total / upd_wall_s),
        }
        for mk, vals in upd_metrics_acc.items():
            summary[mk] = float(np.mean(vals)) if vals else 0.0
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
            elif report.healthy and ((upd + 1) % max(int(cfg.rnad.audit_interval), 1) == 0):
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
                f"ora_ce={summary.get('oracle_rank_ce', 0.0):.3f} bel_p={summary.get('belief_pred_mean', 0.0):.4f} "
                f"ent={summary.get('policy_entropy', 0.0):.3f} kl={summary.get('approx_kl', 0.0):.4f} "
                f"bc_kl={summary.get('bc_kl', 0.0):.4f} p_rii={summary.get('p_riichi_legal', 0.0):.3f}(n={int(summary.get('n_riichi_legal', 0.0))}) "
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

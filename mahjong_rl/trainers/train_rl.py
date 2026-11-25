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
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    reset_adam_first_moment_only,
    set_injected_lr,
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


def train_rl_v6(
    cfg: RLConfig,
    *,
    policy_params_init: Optional[Any] = None,
    belief_params_init: Optional[Any] = None,
    belief_t_cal: float = 1.0,
    grp_params_init: Optional[Any] = None,
    enable_self_healing: bool = True,
) -> V6RLResult:
    """Run Phase 1 (Frozen-Policy Warmup) + Phase 2 (Sovereign-Mahjong V6 RL)."""
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

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_pi, k_bel, k_grp, k_ora, k_pool, k_env = jax.random.split(rng, 7)

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
    w_tables, is_pfsp_env, learning_seat_mask = sample_meta_table_pool(cfg.num_envs, k_pool)

    # Initialize parallel environments
    env_keys = jax.random.split(k_env, cfg.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(env_keys)

    # PFSP elite pool (single-snapshot per rollout -> zero recompile!) + SD-TAP snapshot buffer
    pfsp_pool: Deque[Any] = deque(maxlen=max(1, int(cfg.rnad.pfsp_pool_size)))
    pfsp_pool.append(_clone_params(params_pi))
    outer_snapshots: List[Any] = [_clone_params(params_pi)]

    healer = SelfHealingManager(cfg=cfg.rnad)
    np_rng = np.random.default_rng(cfg.seed)

    # =========================================================================
    # JIT 1: Vectorized Rollout (Per-Rollout Single-Snapshot PFSP, Zero Recompile!)
    # =========================================================================
    @jax.jit
    def rollout_fn(
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
            rsnap = extract_ron_snapshot(clean_st)

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
            step_winner_abs = jnp.where(
                (action == Action.TSUMO) | (action == Action.RON),
                cp,
                jnp.int32(4),
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
                "rsnap": rsnap,
            }
            return next_st, trans

        def scan_step(carry, _):
            st_b, r = carry
            r, k_sub = jax.random.split(r)
            step_keys = jax.random.split(k_sub, cfg.num_envs)
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

        v_boot_b = jax.vmap(eval_bootstrap)(final_st_b, w_tab)  # (B, 4, 4)
        return final_st_b, traj_tb, v_boot_b

    # =========================================================================
    # JIT 2: Dual-Rate Simplex TD + Running-RMS + Sparse K_max Ron Labeling
    # =========================================================================
    @jax.jit
    def prepare_batch_fn(traj_tb, v_boot_b, rms_st):
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
        )  # (B, T, 4, 4)

        # Ego advantage: <(target - V_oracle)[cp], w_me>
        b_idx = jnp.arange(cfg.num_envs, dtype=jnp.int32)[:, None]
        t_idx = jnp.arange(cfg.num_steps, dtype=jnp.int32)[None, :]
        cp_bt = traj_bt["current_player"].astype(jnp.int32)
        p_diff_me = targets_4x4[b_idx, t_idx, cp_bt] - traj_bt["v_oracle_4x4"][b_idx, t_idx, cp_bt]  # (B, T, 4)
        w_me_bt = traj_bt["w_table_ego"][:, :, 0, :]  # (B, T, 4)
        raw_adv = jnp.sum(p_diff_me * w_me_bt, axis=-1)  # (B, T)

        num_legal = jnp.sum(traj_bt["legal_mask"].astype(jnp.int32), axis=-1)
        active_mask = traj_bt["is_learn"] & (num_legal >= 2)
        new_rms, scaled_adv, rms_val = update_running_rms(
            rms_st,
            raw_adv,
            active_mask,
            beta=cfg.rnad.running_rms_beta,
            eps=cfg.rnad.running_rms_eps,
        )

        win_ego, score_delta_ego, aux_valid = compute_round_outcome_targets(
            scores_curr=traj_bt["scores_curr"],
            scores_post=traj_bt["scores_post"],
            round_ended=traj_bt["round_ended"],
            dones=traj_bt["done"],
            step_winner_abs=traj_bt["step_winner_abs"],
            current_players=cp_bt,
        )

        # Flatten (B, T, ...) -> (N, ...)
        n_total = cfg.num_envs * cfg.num_steps
        flat = jax.tree_util.tree_map(lambda x: x.reshape((n_total,) + x.shape[2:]), traj_bt)
        flat["targets_4x4"] = targets_4x4.reshape((n_total, 4, 4))
        flat["adv_scaled"] = scaled_adv.reshape((n_total,))
        flat["win_ego"] = win_ego.reshape((n_total,))
        flat["score_delta_ego"] = score_delta_ego.reshape((n_total, 4))
        flat["aux_valid"] = aux_valid.reshape((n_total,))

        # Sparse K_max compacted legal-ron labeling (runs Yaku.judge ONLY on ~1.5% active slots!)
        y_ron, y_sb, y_ct, y_ten, _ = compute_sparse_belief_labels(
            flat["rsnap"], k_max=cfg.rnad.ron_k_max
        )
        flat["y_ron"] = y_ron
        flat["y_score_bin"] = y_sb
        flat["y_call_threat"] = y_ct
        flat["y_tenpai"] = y_ten
        del flat["rsnap"]
        return flat, new_rms, rms_val

    # =========================================================================
    # JIT 3: Isolated 3-Chain Minibatch Update (Warmup vs Full RL)
    # =========================================================================
    def _make_update_step(update_policy: bool):
        @jax.jit
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
        ):
            # Recompute river_spatial_feat via frozen B_EMA for History-State Oracle
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
            u_ora, o_ora = tx_oracle.update(g_ora, o_ora, p_ora)
            p_ora = optax.apply_updates(p_ora, u_ora)

            # 2. Belief Update (Isolated chain tx_belief + quasi-static EMA)
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
            u_bel, o_bel = tx_belief.update(g_bel, o_bel, p_b_fast)
            p_b_fast = optax.apply_updates(p_b_fast, u_bel)
            p_b_ema = ema_update_params(p_b_ema, p_b_fast, cfg.belief.ema_tau)

            finite_all = _tree_is_finite(g_ora) & _tree_is_finite(g_bel)

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
                u_pi, o_pi = tx_policy.update(g_pi, o_pi, p_pi)
                p_pi = optax.apply_updates(p_pi, u_pi)
                finite_all = finite_all & _tree_is_finite(g_pi)
            else:
                pi_metrics = {
                    "policy_loss": jnp.array(0.0, dtype=jnp.float32),
                    "ppo_loss": jnp.array(0.0, dtype=jnp.float32),
                    "rnad_kl": jnp.array(0.0, dtype=jnp.float32),
                    "shield_loss": jnp.array(0.0, dtype=jnp.float32),
                    "approx_kl": jnp.array(0.0, dtype=jnp.float32),
                    "clip_frac": jnp.array(0.0, dtype=jnp.float32),
                    "policy_entropy": jnp.array(1.0, dtype=jnp.float32),
                    "active_frac": jnp.array(0.0, dtype=jnp.float32),
                    "beta_risk": jnp.array(0.0, dtype=jnp.float32),
                    "beta_offense": jnp.array(0.0, dtype=jnp.float32),
                }

            all_metrics = {**ora_metrics, **bel_metrics, **pi_metrics, "finite_grads": finite_all.astype(jnp.float32)}
            return p_pi, o_pi, p_ora, o_ora, p_b_fast, o_bel, p_b_ema, all_metrics

        return step_fn

    update_mb_warmup = _make_update_step(update_policy=False)
    update_mb_full = _make_update_step(update_policy=True)

    # =========================================================================
    # Main Training Loop (Phase 1 Warmup -> Phase 2 Sovereign V6 RL)
    # =========================================================================
    n_total = cfg.num_envs * cfg.num_steps
    mb_size = min(cfg.minibatch_size, n_total)
    history: List[Dict[str, float]] = []
    steps_since_outer_reset = cfg.rnad.reset_warmup_steps
    t_cal_arr = jnp.asarray(belief_t_cal, dtype=jnp.float32)

    # Initial healthy checkpoint before updates
    healer.save_checkpoint(
        0,
        {
            "params_pi": params_pi,
            "opt_pi": opt_pi,
            "params_oracle": params_oracle,
            "opt_oracle": opt_oracle,
            "params_b_fast": params_b_fast,
            "opt_belief": opt_belief,
            "params_b_ema": params_b_ema,
            "rms_state": rms_state,
        },
    )

    for upd in range(cfg.num_updates):
        t_upd_start = time.perf_counter()
        in_warmup = upd < int(cfg.rnad.warmup_updates)
        phase2_step = max(0, upd - int(cfg.rnad.warmup_updates))

        # Outer R-NaD hard reset at every `outer_reset_steps` in Phase 2
        if (not in_warmup) and phase2_step > 0 and (phase2_step % int(cfg.rnad.outer_reset_steps) == 0):
            snap = _clone_params(params_pi)
            pfsp_pool.append(snap)
            outer_snapshots.append(snap)
            params_rho_base = snap
            # Surgical AdamW reset: zero ONLY m_t, preserve 100% of v_t!
            opt_pi = reset_adam_first_moment_only(opt_pi)
            steps_since_outer_reset = 0

        # Apply smooth post-reset warmup & self-healing LR scaling (zero recompile!)
        if steps_since_outer_reset < int(cfg.rnad.reset_warmup_steps):
            frac = 0.1 + 0.9 * (steps_since_outer_reset + 1) / max(int(cfg.rnad.reset_warmup_steps), 1)
            steps_since_outer_reset += 1
        else:
            frac = 1.0
        eff_lr = float(cfg.rnad.policy_lr) * healer.current_lr_scale * frac
        opt_pi = set_injected_lr(opt_pi, eff_lr)

        # Per-Rollout Single-Snapshot PFSP sampling (zero recompile!)
        opp_idx = int(np_rng.integers(0, len(pfsp_pool)))
        params_opp = pfsp_pool[opp_idx]

        rng, k_roll = jax.random.split(rng)
        env_state, traj_tb, v_boot_b = rollout_fn(
            env_state,
            params_pi,
            params_opp,
            params_b_ema,
            params_grp_base,
            params_grp_ema,
            params_oracle,
            w_tables,
            learning_seat_mask,
            t_cal_arr,
            k_roll,
        )

        # Structural deck uniqueness probe
        decks_sig = np.asarray(jax.device_get(env_state.round_state.deck[:, :8]))
        distinct_ratio = len({row.tobytes() for row in decks_sig}) / max(cfg.num_envs, 1)

        flat_batch, rms_state, rms_val = prepare_batch_fn(traj_tb, v_boot_b, rms_state)

        progress = upd / max(cfg.num_updates - 1, 1)
        shield_lam = jnp.asarray(compute_shield_lambda(progress, cfg.rnad), dtype=jnp.float32)

        step_fn = update_mb_warmup if in_warmup else update_mb_full
        upd_metrics_acc: Dict[str, List[float]] = {}
        all_finite = True

        for _ in range(cfg.update_epochs):
            perm = np_rng.permutation(n_total)
            for start in range(0, n_total, mb_size):
                idx_np = perm[start : start + mb_size]
                if idx_np.size < mb_size:
                    continue
                idx_j = jnp.asarray(idx_np, dtype=jnp.int32)
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
                    params_bc,
                    params_rho_base,
                    mb,
                    shield_lam,
                    t_cal_arr,
                )
                if float(m_dict["finite_grads"]) < 0.5:
                    all_finite = False
                for mk, mv in m_dict.items():
                    upd_metrics_acc.setdefault(mk, []).append(float(mv))

        jax.block_until_ready(params_pi)
        upd_wall_s = max(time.perf_counter() - t_upd_start, 1e-6)
        summary = {
            "update": float(upd),
            "in_warmup": float(in_warmup),
            "adv_rms": float(rms_val),
            "eff_policy_lr": float(eff_lr),
            "shield_lambda": float(shield_lam),
            "distinct_deck_ratio": float(distinct_ratio),
            "update_wall_s": float(upd_wall_s),
            "sps": float(n_total / upd_wall_s),
        }
        for mk, vals in upd_metrics_acc.items():
            summary[mk] = float(np.mean(vals)) if vals else 0.0

        # Self-healing probe evaluation
        if enable_self_healing and (not in_warmup):
            report = healer.evaluate_probes(
                summary,
                finite_grads=all_finite,
                distinct_deck_ratio=distinct_ratio,
            )
            summary["healed"] = 1.0 if report.action_taken == "rollback_and_halve_lr" else 0.0
            if report.action_taken == "rollback_and_halve_lr":
                restored = healer.restore_latest()
                if restored is not None:
                    params_pi = restored["params_pi"]
                    opt_pi = reset_adam_first_moment_only(restored["opt_pi"])
                    params_oracle = restored["params_oracle"]
                    opt_oracle = restored["opt_oracle"]
                    params_b_fast = restored["params_b_fast"]
                    opt_belief = restored["opt_belief"]
                    params_b_ema = restored["params_b_ema"]
                    rms_state = restored["rms_state"]
            elif report.healthy and ((upd + 1) % max(int(cfg.rnad.audit_interval), 1) == 0):
                healer.save_checkpoint(
                    upd + 1,
                    {
                        "params_pi": params_pi,
                        "opt_pi": opt_pi,
                        "params_oracle": params_oracle,
                        "opt_oracle": opt_oracle,
                        "params_b_fast": params_b_fast,
                        "opt_belief": opt_belief,
                        "params_b_ema": params_b_ema,
                        "rms_state": rms_state,
                    },
                )
        history.append(summary)

    # Always append final policy to outer_snapshots for Phase 3 SD-TAP
    outer_snapshots.append(_clone_params(params_pi))

    return V6RLResult(
        policy_model=policy_net,
        policy_params=params_pi,
        belief_model=belief_net,
        belief_ema_params=params_b_ema,
        belief_fast_params=params_b_fast,
        belief_t_cal=belief_t_cal,
        grp_model=grp_net,
        grp_base_params=params_grp_base,
        oracle_model=oracle_net,
        oracle_params=params_oracle,
        outer_snapshots=outer_snapshots,
        history=history,
    )

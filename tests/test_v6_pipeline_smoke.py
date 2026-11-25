"""V6 Invariant Tests 7 & 8: Per-Rollout PFSP Zero-Recompile & Full End-to-End GPU Smoke."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjong_rl.algo.utility import W_M_LEAGUE, W_PHOENIX_7D, W_PHOENIX_10D
from mahjong_rl.config import (
    BCConfig,
    BeliefConfig,
    EvalConfig,
    GRPConfig,
    ModelConfig,
    OracleConfig,
    RLConfig,
    RNaDConfig,
)
from mahjong_rl.data.mjai_dataset import build_shards_from_policy_rollout
from mahjong_rl.eval.duplicate_eval import evaluate_duplicate_1v3
from mahjong_rl.eval.mjai_bridge import make_v6_policy_actor
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.trainers.distill_sdtap import distill_sdtap
from mahjong_rl.trainers.train_bc import init_policy_params, train_bc
from mahjong_rl.trainers.train_belief import train_belief
from mahjong_rl.trainers.train_grp import train_grp
from mahjong_rl.trainers.train_rl import train_rl_v6


def test_per_rollout_pfsp_zero_xla_recompilation():
    """Test 7: Swapping single-snapshot `params_opp` PyTree across rollouts triggers ZERO recompiles."""
    p_cfg = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    p_net = SovereignPolicyNet(config=p_cfg)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    params_pi = init_policy_params(p_net, k1)
    params_opp_1 = init_policy_params(p_net, k2)
    params_opp_2 = init_policy_params(p_net, k3)

    trace_count = {"n": 0}

    @jax.jit
    def pfsp_step(p_active, p_opp, planes):
        trace_count["n"] += 1
        kw = dict(
            planes=planes,
            delta_u_ron_3opp=jnp.zeros((2, 3, 34)),
            call_threat_plane=jnp.zeros((2, 34)),
            scalars=jnp.zeros((2, 56)),
            river_tokens=jnp.zeros((2, 4, 24, 23)),
            river_tile_ids=jnp.full((2, 4, 24), -1, dtype=jnp.int32),
            river_valid_mask=jnp.zeros((2, 4, 24), dtype=jnp.bool_),
            last_draw_type=jnp.full((2,), -1, dtype=jnp.int32),
            is_last_draw_red=jnp.zeros((2,), dtype=jnp.bool_),
            target_type=jnp.full((2,), -1, dtype=jnp.int32),
            target_is_red=jnp.zeros((2,), dtype=jnp.bool_),
            w_ego=jnp.broadcast_to(W_PHOENIX_7D[None, :], (2, 4)),
            delta_u_agari_3=jnp.zeros((2, 3)),
            delta_u_ryuukyoku=jnp.zeros((2,)),
            delta_u_ron_87=jnp.zeros((2, 87)),
            legal_action_mask=jnp.ones((2, 87), dtype=jnp.bool_),
        )
        return p_net.apply(p_active, **kw) + p_net.apply(p_opp, **kw)

    dummy_planes = jnp.zeros((2, 53, 34), dtype=jnp.float32)
    _ = pfsp_step(params_pi, params_opp_1, dummy_planes)
    _ = pfsp_step(params_pi, params_opp_2, dummy_planes)
    _ = pfsp_step(params_pi, params_pi, dummy_planes)
    assert trace_count["n"] == 1, f"Expected ZERO recompiles (1 trace), got {trace_count['n']}"


def test_v6_full_pipeline_end_to_end_smoke(tmp_path):
    """Test 8: Full Sovereign-Mahjong V6 pipeline (Phase 0-A/B/C -> Phase 1/2 RL -> Phase 3 SD-TAP -> Eval)."""
    # 1. Generate compact V6 shards from real MahJax episodes
    shard_dir = tmp_path / "v6_shards"
    shards = build_shards_from_policy_rollout(
        out_dir=shard_dir,
        num_games=2,
        env_name="red_4p",
        round_mode="single",
        max_steps_per_game=140,
        ron_k_max=128,
        seed=0,
    )
    assert len(shards) == 1

    small_model = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    small_belief = BeliefConfig(
        num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32,
        batch_size=64, num_epochs=1, seed=0,
    )
    small_grp = GRPConfig(hidden_dim=32, num_layers=2, batch_size=64, num_epochs=1)
    small_oracle = OracleConfig(num_blocks=1, channels=32, attn_heads=4, global_hidden_dim=32)

    # 2. Phase 0-A: BeliefNet + Post-Training Temperature Scaling
    b_model, b_params, t_cal, b_hist = train_belief(shards, cfg=small_belief, suit_augment=True)
    assert 0.5 <= t_cal <= 2.5
    assert np.isfinite(b_hist[-1]["belief_loss"])

    # 3. Phase 0-B: Birkhoff (4, 4) GRP_base
    g_model, g_params, g_hist = train_grp(shards, cfg=small_grp, seed=0)
    assert g_hist[-1]["grp_birkhoff_err"] < 1e-3
    assert np.isfinite(g_hist[-1]["grp_ce"])

    # 4. Phase 0-C: Strictly Isomorphic BC (with frozen B_EMA + GRP_base + beta_risk/offense)
    bc_cfg = BCConfig(model=small_model, batch_size=64, num_epochs=1, suit_augment=True, seed=0)
    _, bc_params, bc_hist = train_bc(
        shards,
        belief_model=b_model,
        belief_params=b_params,
        grp_model=g_model,
        grp_params=g_params,
        cfg=bc_cfg,
        belief_t_cal=t_cal,
    )
    assert np.isfinite(bc_hist[-1]["bc_loss"])
    assert 0.0 < bc_hist[-1]["beta_risk"] < small_model.beta_risk_max

    # 5. Phase 1 (Warmup) + Phase 2 (Sovereign-Mahjong V6 RL with Outer Reset + PFSP)
    rl_cfg = RLConfig(
        round_mode="single",
        model=small_model,
        belief=small_belief,
        grp=small_grp,
        oracle=small_oracle,
        rnad=RNaDConfig(
            outer_reset_steps=1,
            reset_warmup_steps=1,
            ron_k_max=128,
            warmup_updates=1,
            audit_interval=2,
        ),
        num_envs=4,
        num_steps=16,
        num_updates=3,
        update_epochs=1,
        minibatch_size=32,
        seed=0,
    )
    rl_res = train_rl_v6(
        rl_cfg,
        policy_params_init=bc_params,
        belief_params_init=b_params,
        belief_t_cal=t_cal,
        grp_params_init=g_params,
        enable_self_healing=True,
    )
    assert len(rl_res.history) == 3
    assert len(rl_res.outer_snapshots) >= 2
    for row in rl_res.history:
        assert np.isfinite(row["oracle_loss"])
        assert np.isfinite(row["belief_loss"])
        assert np.isfinite(row["policy_loss"])
        assert row["oracle_birkhoff_err"] < 1e-3

    # 6. Phase 3: Offline Behavioral-Space Self-Distillation (SD-TAP)
    deploy_model, deploy_params, dt_hist = distill_sdtap(
        shards,
        rl_res.outer_snapshots,
        belief_model=rl_res.belief_model,
        belief_ema_params=rl_res.belief_ema_params,
        grp_model=rl_res.grp_model,
        grp_base_params=rl_res.grp_base_params,
        model_cfg=small_model,
        belief_t_cal=t_cal,
        batch_size=64,
        num_epochs=1,
    )
    assert np.isfinite(dt_hist[-1]["sdtap_kl"])

    # 7. Deployment Bridge + Zero-Shot Rule Switching (Phoenix 10d vs M-League) + Duplicate 1v3
    actor_10d = make_v6_policy_actor(
        deploy_model, deploy_params,
        rl_res.belief_model, rl_res.belief_ema_params,
        rl_res.grp_model, rl_res.grp_base_params,
        w_rule=W_PHOENIX_10D,
        belief_t_cal=t_cal,
    )
    actor_ml = make_v6_policy_actor(
        deploy_model, deploy_params,
        rl_res.belief_model, rl_res.belief_ema_params,
        rl_res.grp_model, rl_res.grp_base_params,
        w_rule=W_M_LEAGUE,
        belief_t_cal=t_cal,
    )
    eval_cfg = EvalConfig(round_mode="single", num_walls=1, max_steps=120, seed=0)
    res_10d = evaluate_duplicate_1v3(actor_10d, mahjax.rule_based_player, eval_cfg)
    res_ml = evaluate_duplicate_1v3(actor_ml, mahjax.rule_based_player, eval_cfg)
    assert res_10d["total_games"] == 4.0 and np.isfinite(res_10d["tenhou_pt_mean"])
    assert res_ml["total_games"] == 4.0 and np.isfinite(res_ml["tenhou_pt_mean"])

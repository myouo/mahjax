"""Smoke tests for the params-as-arguments Duplicate 1v3 evaluator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjong_rl.config import BeliefConfig, EvalConfig, GRPConfig, ModelConfig
from mahjong_rl.eval.compiled_eval import CompiledDuplicateEvaluator
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRPNet


def _small_models():
    mc = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    bc = BeliefConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    gc = GRPConfig(hidden_dim=32, num_layers=2, sinkhorn_iters=10)
    return SovereignPolicyNet(config=mc), BeliefNet(config=bc), GRPNet(config=gc)


def _init_params(policy, belief, grp, key):
    k1, k2, k3 = jax.random.split(key, 3)
    planes = jnp.zeros((1, 53, 34))
    scalars = jnp.zeros((1, 56))
    river_tokens = jnp.zeros((1, 4, 24, 23))
    river_ids = jnp.full((1, 4, 24), -1, dtype=jnp.int32)
    river_valid = jnp.zeros((1, 4, 24), dtype=jnp.bool_)
    p_pol = policy.init(
        k1, planes=planes, delta_u_ron_3opp=jnp.zeros((1, 3, 34)),
        call_threat_plane=jnp.zeros((1, 34)), scalars=scalars,
        river_tokens=river_tokens, river_tile_ids=river_ids, river_valid_mask=river_valid,
        last_draw_type=jnp.full((1,), -1, dtype=jnp.int32),
        is_last_draw_red=jnp.zeros((1,), dtype=jnp.bool_),
        target_type=jnp.full((1,), -1, dtype=jnp.int32),
        target_is_red=jnp.zeros((1,), dtype=jnp.bool_),
        w_ego=jnp.full((1, 4), 0.25), delta_u_agari_3=jnp.zeros((1, 3)),
        delta_u_ryuukyoku=jnp.zeros((1,)), delta_u_ron_87=jnp.zeros((1, 87)),
        legal_action_mask=jnp.ones((1, 87), dtype=jnp.bool_),
    )
    p_bel = belief.init(
        k2, planes=planes, scalars=scalars, river_tokens=river_tokens,
        river_tile_ids=river_ids, river_valid_mask=river_valid,
        w_opp_5d=jnp.zeros((1, 3, 5)), temperature_cal=1.0,
    )
    p_grp = grp.init(k3, jnp.zeros((1, 29)))
    return p_pol, p_bel, p_grp


def test_compiled_duplicate_evaluator_both_modes():
    policy, belief, grp = _small_models()
    p_pol, p_bel, p_grp = _init_params(policy, belief, grp, jax.random.PRNGKey(0))
    cfg = EvalConfig(env_name="red_4p", round_mode="single", num_walls=1, max_steps=120, seed=0)
    ev = CompiledDuplicateEvaluator(
        policy_model=policy,
        belief_model=belief,
        grp_model=grp,
        grp_base_params=p_grp,
        eval_cfg=cfg,
        belief_t_cal=1.0,
        greedy=True,
        chunk_walls=1,
    )

    vs_params = ev.evaluate(p_pol, p_bel, p_pol, p_bel)
    vs_rule = ev.evaluate_vs_rule(p_pol, p_bel)
    for res in (vs_params, vs_rule):
        assert res["total_games"] == 4.0
        assert np.isfinite(res["tenhou_pt_mean"])
        assert 0.0 <= res["finished_game_ratio"] <= 1.0
        assert len(res["wall_mean_pts"]) >= 1

    # Second call must reuse the compiled executable (and still be correct).
    again = ev.evaluate(p_pol, p_bel, p_pol, p_bel)
    assert again["total_games"] == 4.0
    assert again["tenhou_pt_mean"] == pytest.approx(vs_params["tenhou_pt_mean"])

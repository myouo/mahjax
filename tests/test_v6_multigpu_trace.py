"""Regression: the V6 actor builder must be tracer-safe with >1 local device.

On Kaggle (2x Tesla T4) the periodic Duplicate evaluator calls
``make_v6_policy_actor`` *inside* a ``jax.jit`` trace.  The multi-GPU branch used
to convert the (tracer) policy/belief/GRP params with ``np.asarray(device_get(x))``,
which raises ``TracerArrayConversionError``.  This test simulates a 2-device host on
a single-device machine and verifies the trace succeeds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import mahjax
from mahjong_rl.config import BeliefConfig, GRPConfig, ModelConfig
from mahjong_rl.eval.mjai_bridge import make_v6_policy_actor
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRPNet


def test_actor_builder_is_tracer_safe(monkeypatch):
    monkeypatch.setattr(jax, "local_device_count", lambda: 2)

    mc = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    bc = BeliefConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    gc = GRPConfig(hidden_dim=32, num_layers=2, sinkhorn_iters=10)
    policy, belief, grp = (
        SovereignPolicyNet(config=mc),
        BeliefNet(config=bc),
        GRPNet(config=gc),
    )

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
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

    env = mahjax.make("red_4p", round_mode="single", observe_type="dict", round_end="confirm")
    state = env.init(k4)

    @jax.jit
    def act(pol_p, bel_p, grp_p, st, key):
        actor = make_v6_policy_actor(
            policy, pol_p, belief, bel_p, grp, grp_p, belief_t_cal=1.0, greedy=True
        )
        return actor(st, key)

    action = act(p_pol, p_bel, p_grp, state, jax.random.PRNGKey(1))
    assert 0 <= int(action) < 87

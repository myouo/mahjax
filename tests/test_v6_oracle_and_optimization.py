"""V6 Invariant Tests 5 & 6: Birkhoff Sinkhorn, Dual-Rate TD, Optimizer Surgery & Self-Healing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mahjong_rl.algo.gae import calculate_dual_rate_simplex_td_lambda
from mahjong_rl.algo.sovereign_v6 import (
    SelfHealingManager,
    build_isolated_optimizers,
    compute_birkhoff_error,
    compute_policy_loss_v6,
    init_running_rms,
    make_weight_decay_mask,
    reset_adam_first_moment_only,
    update_running_rms,
)
from mahjong_rl.algo.utility import W_PHOENIX_7D, build_opponent_utility_features
from mahjong_rl.config import BeliefConfig, GRPConfig, ModelConfig, OracleConfig, RNaDConfig
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRP_FEATURE_DIM, GRPNet, log_sinkhorn_4x4
from mahjong_rl.models.oracle import OracleCriticNet
from mahjong_rl.trainers.train_bc import init_policy_params


def test_birkhoff_sinkhorn_zero_sum_and_dual_rate_td_closure():
    """Test 5: Birkhoff doubly-stochastic manifold, exact zero-sum conservation, and zero cross-episode leak."""
    # 1. GRPNet & OracleCriticNet doubly-stochastic conservation + zero-sum under Phoenix 7d
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
    grp_net = GRPNet(config=GRPConfig(hidden_dim=32, num_layers=2, sinkhorn_iters=10))
    grp_params = grp_net.init(k1, jnp.zeros((4, GRP_FEATURE_DIM)))
    x_grp = jax.random.normal(k2, (8, GRP_FEATURE_DIM))
    grp_probs, grp_logits = grp_net.apply(grp_params, x_grp, return_logits=True)

    assert float(compute_birkhoff_error(grp_probs)) < 1e-3
    # Under Phoenix 7d (mean=0, zero-sum!), sum_seat <P_seat, W_7d> MUST equal 0!
    v_per_seat = jnp.sum(grp_probs * W_PHOENIX_7D[None, None, :], axis=-1)  # (8, 4)
    assert float(jnp.max(jnp.abs(jnp.sum(v_per_seat, axis=-1)))) < 1e-4

    ora_net = OracleCriticNet(config=OracleConfig(num_blocks=1, channels=32, attn_heads=4, global_hidden_dim=32), river_dim=8)
    ora_params = ora_net.init(
        k3,
        planes=jnp.zeros((8, 53, 34)),
        oracle_planes=jnp.zeros((8, 26, 34)),
        scalars=jnp.zeros((8, 56)),
        w_table_ego=jnp.zeros((8, 4, 4)),
        grp_logits_abs=grp_logits,
        current_player=jnp.arange(8, dtype=jnp.int32) % 4,
        river_spatial_feat=jnp.zeros((8, 34, 32)),
    )
    ora_out = ora_net.apply(
        ora_params,
        planes=jnp.zeros((8, 53, 34)),
        oracle_planes=jnp.zeros((8, 26, 34)),
        scalars=jnp.zeros((8, 56)),
        w_table_ego=jnp.zeros((8, 4, 4)),
        grp_logits_abs=grp_logits,
        current_player=jnp.arange(8, dtype=jnp.int32) % 4,
        river_spatial_feat=jnp.zeros((8, 34, 32)),
    )
    assert float(compute_birkhoff_error(ora_out["probs_4x4"])) < 1e-3

    # 2. Dual-Rate Simplex TD(lambda): Birkhoff closure + exact episode cutoff
    B, T = 2, 10
    v_ora = log_sinkhorn_4x4(jax.random.normal(k1, (B, T, 4, 4)), num_iters=10)
    v_boot = log_sinkhorn_4x4(jax.random.normal(k2, (B, 4, 4)), num_iters=10)
    grp_ema = log_sinkhorn_4x4(jax.random.normal(k3, (B, T, 4, 4)), num_iters=10)

    # Exact terminal permutation at t=4
    perm_mat = jnp.eye(4, dtype=jnp.float32)
    term_4x4 = jnp.broadcast_to(perm_mat[None, None, :, :], (B, T, 4, 4))
    dones = jnp.zeros((B, T), dtype=jnp.bool_).at[:, 4].set(True)
    round_ended = jnp.zeros((B, T), dtype=jnp.bool_).at[:, 2].set(True).at[:, 4].set(True)

    targets1 = calculate_dual_rate_simplex_td_lambda(
        v_ora, v_boot, grp_ema, term_4x4, dones, round_ended, lambda_intra=0.975, lambda_inter=0.70
    )
    # Birkhoff closure at EVERY (b, t)!
    assert float(compute_birkhoff_error(targets1)) < 1e-3
    # At t=4 (done=True), target MUST equal exact terminal permutation matrix!
    np.testing.assert_allclose(np.asarray(targets1[:, 4]), np.asarray(term_4x4[:, 4]), atol=1e-6)

    # Corrupt everything AFTER episode 1 (t >= 5 and v_boot) -> steps t <= 4 MUST be 100% unaffected!
    v_ora_corrupt = v_ora.at[:, 5:].set(jnp.flip(perm_mat, axis=-1))
    v_boot_corrupt = jnp.broadcast_to(jnp.flip(perm_mat, axis=-1)[None, :, :], (B, 4, 4))
    targets2 = calculate_dual_rate_simplex_td_lambda(
        v_ora_corrupt, v_boot_corrupt, grp_ema, term_4x4, dones, round_ended, lambda_intra=0.975, lambda_inter=0.70
    )
    np.testing.assert_allclose(np.asarray(targets1[:, :5]), np.asarray(targets2[:, :5]), atol=1e-6)


def test_optimizer_surgery_wd_exemption_running_rms_and_self_healing():
    """Test 6: AdamW m_t-only reset, WD exemptions, forced-move masking, RunningRMS, & Self-Healing."""
    p_cfg = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    p_net = SovereignPolicyNet(config=p_cfg)
    params = init_policy_params(p_net, jax.random.PRNGKey(0))

    # 1. Weight decay mask exemptions
    wd_mask = make_weight_decay_mask(params)
    assert wd_mask["params"]["raw_beta_risk"] is False
    assert wd_mask["params"]["raw_beta_offense"] is False
    # Every 1D leaf (bias, LayerNorm scale) must be False, while 2D+ dense/conv kernels are True
    def _check_leaf(path, m_val, p_val):
        if p_val.ndim <= 1:
            assert m_val is False, f"1D param {path} should have WD=False"
    jax.tree_util.tree_map_with_path(_check_leaf, wd_mask, params)

    # 2. Surgical AdamW first-moment reset (m_t == 0, v_t & count preserved BIT-FOR-BIT!)
    tx_policy, _, _ = build_isolated_optimizers(RNaDConfig(), OracleConfig(), BeliefConfig())
    opt_state = tx_policy.init(params)
    fake_grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.1, params)
    _, opt_state = tx_policy.update(fake_grads, opt_state, params)
    _, opt_state = tx_policy.update(fake_grads, opt_state, params)

    def _find_adam_states(node):
        found = []
        if isinstance(node, optax.ScaleByAdamState):
            found.append(node)
        elif isinstance(node, (tuple, list)):
            for x in node:
                found.extend(_find_adam_states(x))
        elif isinstance(node, dict):
            for x in node.values():
                found.extend(_find_adam_states(x))
        return found

    before_adams = _find_adam_states(opt_state)
    assert len(before_adams) >= 1
    assert any(float(jnp.max(jnp.abs(x))) > 0.0 for x in jax.tree_util.tree_leaves(before_adams[0].mu))

    reset_state = reset_adam_first_moment_only(opt_state)
    after_adams = _find_adam_states(reset_state)
    for b_st, a_st in zip(before_adams, after_adams):
        # mu MUST be strictly 0
        for leaf in jax.tree_util.tree_leaves(a_st.mu):
            assert float(jnp.max(jnp.abs(leaf))) == 0.0
        # nu and count MUST be preserved bit-for-bit!
        for b_nu, a_nu in zip(jax.tree_util.tree_leaves(b_st.nu), jax.tree_util.tree_leaves(a_st.nu)):
            np.testing.assert_array_equal(np.asarray(b_nu), np.asarray(a_nu))
        np.testing.assert_array_equal(np.asarray(b_st.count), np.asarray(a_st.count))

    # 3. Forced-move (`num_legal == 1`) masking gives strictly 0 loss & 0 gradient!
    logits = jnp.zeros((4, 87), dtype=jnp.float32)
    mask_forced = jnp.zeros((4, 87), dtype=jnp.bool_).at[:, 5].set(True)
    actions = jnp.full((4,), 5, dtype=jnp.int32)
    loss_val, grads = jax.value_and_grad(
        lambda lg: compute_policy_loss_v6(
            logits=lg,
            actions=actions,
            old_log_probs=jnp.zeros((4,)),
            advantages_scaled=jnp.ones((4,)) * 10.0,
            log_rho=jnp.where(mask_forced, 0.0, -1e9),
            bc_logits=jnp.zeros((4, 87)),
            legal_mask=mask_forced,
            learning_seat_mask=jnp.ones((4,), dtype=jnp.bool_),
            cfg=RNaDConfig(),
            shield_lambda=0.05,
        )[0]
    )(logits)
    assert float(loss_val) == 0.0
    assert float(jnp.max(jnp.abs(grads))) == 0.0

    # 4. Running-RMS strictly preserves positive sign (NO mean subtraction!) & ignores inactive entries
    rms = init_running_rms()
    adv = jnp.array([2.0, 2.0, 1000.0, -1000.0], dtype=jnp.float32)
    active = jnp.array([True, True, False, False])
    rms2, scaled, rms_val = update_running_rms(rms, adv, active)
    # Inactive entries (1000, -1000) MUST NOT contaminate rms (~2.0), and active entries stay POSITIVE (~+1.0)!
    assert np.isclose(float(rms_val), 2.0, atol=1e-3)
    assert np.allclose(np.asarray(scaled[:2]), 1.0, atol=1e-3)
    assert np.all(np.asarray(scaled[2:]) == 0.0)

    # 5. Unbiased opponent utility dropout: NEVER fills with [0,0,0,0]!
    w_opp = jnp.broadcast_to(W_PHOENIX_7D[None, :], (3, 4))
    w_5d = build_opponent_utility_features(w_opp, jnp.zeros((3,)), rng_key=None, dropout_prob=0.0)
    np.testing.assert_allclose(np.asarray(w_5d[:, :4]), np.asarray(w_opp), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(w_5d[:, 4]), np.zeros((3,), dtype=np.float32))

    # 6. Self-Healing Circuit Breaker triggers rollback & halves LR on entropy collapse / NaN
    healer = SelfHealingManager(cfg=RNaDConfig())
    healer.save_checkpoint(0, {"dummy": jnp.array(1.0)})
    rep_ok = healer.evaluate_probes({"approx_kl": 0.01, "policy_entropy": 1.2, "oracle_birkhoff_err": 1e-5})
    assert rep_ok.healthy and rep_ok.action_taken == "none"

    rep_bad = healer.evaluate_probes({"approx_kl": 0.01, "policy_entropy": 0.10, "oracle_birkhoff_err": 1e-5})
    assert (not rep_bad.healthy) and rep_bad.action_taken == "rollback_and_halve_lr"
    assert np.isclose(healer.current_lr_scale, 0.5)
    assert healer.restore_latest() is not None

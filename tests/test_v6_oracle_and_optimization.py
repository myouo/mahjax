"""V6 Invariant Tests 5 & 6: Birkhoff Sinkhorn, Dual-Rate TD, Optimizer Surgery & Self-Healing."""

from __future__ import annotations

import pickle
import pytest

import jax
import jax.numpy as jnp
import numpy as np
import optax

import mahjax
from mahjong_rl.algo.gae import calculate_dual_rate_simplex_td_lambda
from mahjong_rl.algo.sovereign_v6 import (
    SelfHealingManager,
    build_isolated_optimizers,
    compute_birkhoff_error,
    compute_policy_loss_v6,
    init_running_rms,
    make_weight_decay_mask,
    reset_adam_first_moment_only,
    set_injected_lr,
    update_running_rms,
)
from mahjong_rl.algo.utility import W_PHOENIX_7D, W_PHOENIX_10D, build_opponent_utility_features, sample_meta_table_pool
from mahjong_rl.config import BeliefConfig, GRPConfig, ModelConfig, OracleConfig, RLConfig, RNaDConfig
from mahjong_rl.data.mjai_dataset import SYMMETRY_PERMS_36, build_tile_and_action_perm_36
from mahjong_rl.eval.mjai_bridge import make_v6_policy_actor
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRP_FEATURE_DIM, GRPNet, log_sinkhorn_4x4
from mahjong_rl.models.oracle import OracleCriticNet
from mahjong_rl.trainers.train_bc import init_policy_params
from mahjong_rl.trainers.train_belief import _init_belief_params
from mahjong_rl.trainers.train_rl import _desync_hanchan_envs, train_rl_v6


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

    # 7. Self-Healing Circuit Breaker uses rolling weighted window for p_riichi_legal & resets consecutive_heals
    rep_rii_single = healer.evaluate_probes(
        {"approx_kl": 0.01, "policy_entropy": 1.0, "oracle_birkhoff_err": 1e-5, "p_riichi_legal": 0.82, "n_riichi_legal": 49.0}
    )
    assert rep_rii_single.healthy
    assert healer.consecutive_heals == 0

    for _ in range(3):
        healer.evaluate_probes(
            {"approx_kl": 0.01, "policy_entropy": 1.0, "oracle_birkhoff_err": 1e-5, "p_riichi_legal": 0.90, "n_riichi_legal": 50.0}
        )
    rep_rii_hack = healer.evaluate_probes(
        {"approx_kl": 0.01, "policy_entropy": 1.0, "oracle_birkhoff_err": 1e-5, "p_riichi_legal": 0.90, "n_riichi_legal": 50.0}
    )
    assert (not rep_rii_hack.healthy) and any("p_riichi_legal_win" in r for r in rep_rii_hack.reasons)

    # max_snapshots honored on SelfHealingManager & consecutive_heals resets ONLY when healthy
    healer2 = SelfHealingManager(cfg=RNaDConfig(max_consecutive_heals=2), max_snapshots=2)
    for step_idx in range(4):
        healer2.save_checkpoint(step_idx, {"step": jnp.array(step_idx)})
    assert len(healer2.snapshots) == 2 and healer2.snapshots[0]["step"] == 2
    assert healer2.evaluate_probes({"approx_kl": 0.5}).action_taken == "rollback_and_halve_lr"
    assert healer2.consecutive_heals == 1
    assert healer2.evaluate_probes({"approx_kl": 0.001, "policy_entropy": 1.0}).healthy
    assert healer2.consecutive_heals == 0


def test_lambda_intra_one_eliminates_intra_round_bootstrap_bias():
    """Verify lambda_intra=1.0 bypasses intra-round V_oracle(s_{t+1}) bootstrapping completely."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(11), 3)
    B, T = 2, 10
    v_ora = log_sinkhorn_4x4(jax.random.normal(k1, (B, T, 4, 4)), num_iters=10)
    v_boot = log_sinkhorn_4x4(jax.random.normal(k2, (B, 4, 4)), num_iters=10)
    grp_ema = log_sinkhorn_4x4(jax.random.normal(k3, (B, T, 4, 4)), num_iters=10)
    perm_mat = jnp.eye(4, dtype=jnp.float32)
    term_4x4 = jnp.broadcast_to(perm_mat[None, None, :, :], (B, T, 4, 4))
    dones = jnp.zeros((B, T), dtype=jnp.bool_).at[:, 6].set(True)
    round_ended = jnp.zeros((B, T), dtype=jnp.bool_).at[:, 3].set(True).at[:, 6].set(True)

    targets_clean = calculate_dual_rate_simplex_td_lambda(
        v_ora, v_boot, grp_ema, term_4x4, dones, round_ended, lambda_intra=1.0, lambda_inter=0.70
    )
    # Wildly corrupt intermediate intra-round Oracle estimates at t=1..5 (leaving boundary steps t=3,6 any way)
    v_ora_corrupt = v_ora.at[:, 1:6].set(jnp.flip(perm_mat, axis=-1))
    targets_corrupt = calculate_dual_rate_simplex_td_lambda(
        v_ora_corrupt, v_boot, grp_ema, term_4x4, dones, round_ended, lambda_intra=1.0, lambda_inter=0.70
    )
    np.testing.assert_allclose(np.asarray(targets_clean[:, :7]), np.asarray(targets_corrupt[:, :7]), atol=1e-7)
    # Within round 0 (t=0..3), all targets with lambda_intra=1.0 MUST equal the round-boundary target at t=3!
    for t in range(3):
        np.testing.assert_allclose(np.asarray(targets_clean[:, t]), np.asarray(targets_clean[:, 3]), atol=1e-7)


def test_oracle_equivariance_with_opponent_tenpai_scalars():
    """Verify OracleCriticNet remains 36x-equivariant with nonzero opponent tenpai/live-wait scalars."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(19), 2)
    ora_net = OracleCriticNet(
        config=OracleConfig(num_blocks=1, channels=32, attn_heads=4, global_hidden_dim=32),
        river_dim=8,
    )
    planes = jax.random.uniform(k1, (2, 53, 34), minval=0.0, maxval=1.0)
    ora_planes = jnp.zeros((2, 26, 34), dtype=jnp.float32)
    # Populate wall_planes (15..18), opp_waits (20..22), and opp_nf_waits (23..25) with non-trivial patterns
    ora_planes = ora_planes.at[:, 15:18, 2].set(1.0).at[:, 15:17, 11].set(1.0).at[:, 15:19, 31].set(1.0)
    ora_planes = ora_planes.at[:, 20, 2].set(1.0).at[:, 21, 11].set(1.0).at[:, 22, 31].set(1.0)
    ora_planes = ora_planes.at[:, 23, 2].set(1.0).at[:, 25, 31].set(1.0)  # opp 1 furiten
    scalars = jax.random.normal(k2, (2, 56))
    w_table_ego = jnp.broadcast_to(W_PHOENIX_7D[None, None, :], (2, 4, 4))
    grp_logits = jax.random.normal(k1, (2, 4, 4))
    cp = jnp.zeros((2,), dtype=jnp.int32)
    river_feat = jax.random.normal(k2, (2, 34, 32))

    ora_params = ora_net.init(
        k1,
        planes=planes,
        oracle_planes=ora_planes,
        scalars=scalars,
        w_table_ego=w_table_ego,
        grp_logits_abs=grp_logits,
        current_player=cp,
        river_spatial_feat=river_feat,
    )
    base_out = jax.device_get(
        ora_net.apply(
            ora_params,
            planes=planes,
            oracle_planes=ora_planes,
            scalars=scalars,
            w_table_ego=w_table_ego,
            grp_logits_abs=grp_logits,
            current_player=cp,
            river_spatial_feat=river_feat,
        )
    )

    max_err = 0.0
    for sp, dp in SYMMETRY_PERMS_36:
        tile_map, _ = build_tile_and_action_perm_36(sp, dp)
        inv_tile = np.argsort(tile_map)
        perm_out = jax.device_get(
            ora_net.apply(
                ora_params,
                planes=planes[..., inv_tile],
                oracle_planes=ora_planes[..., inv_tile],
                scalars=scalars,
                w_table_ego=w_table_ego,
                grp_logits_abs=grp_logits,
                current_player=cp,
                river_spatial_feat=river_feat[:, inv_tile, :],
            )
        )
        max_err = max(max_err, float(np.max(np.abs(perm_out["probs_4x4"] - base_out["probs_4x4"]))))
    assert max_err < 1e-4, f"Oracle 36x equivariance error too high: {max_err}"


def test_freeze_belief_preserves_phase0_weights_bit_for_bit():
    """Verify freeze_belief=True keeps BeliefNet weights untouched while exposing new policy metrics."""
    small_model = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    small_belief = BeliefConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    small_grp = GRPConfig(hidden_dim=32, num_layers=2)
    small_oracle = OracleConfig(num_blocks=1, channels=32, attn_heads=4, global_hidden_dim=32)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(29), 3)
    p_net = SovereignPolicyNet(config=small_model)
    b_net = BeliefNet(config=small_belief)
    g_net = GRPNet(config=small_grp)

    p_init = init_policy_params(p_net, k1)
    b_init = _init_belief_params(b_net, k2)
    g_init = g_net.init(k3, jnp.zeros((2, GRP_FEATURE_DIM), dtype=jnp.float32))

    rl_cfg = RLConfig(
        round_mode="single",
        model=small_model,
        belief=small_belief,
        grp=small_grp,
        oracle=small_oracle,
        rnad=RNaDConfig(
            lambda_intra=1.0,
            freeze_belief=True,
            warmup_updates=1,
            outer_reset_steps=1,
            reset_warmup_steps=1,
            audit_interval=2,
        ),
        num_envs=4,
        num_steps=16,
        num_updates=3,
        update_epochs=1,
        minibatch_size=32,
        seed=0,
    )
    res = train_rl_v6(
        rl_cfg,
        policy_params_init=p_init,
        belief_params_init=b_init,
        belief_t_cal=1.0,
        grp_params_init=g_init,
        enable_self_healing=True,
        multi_gpu=False,
        verbose=False,
    )
    for orig, ema_w, fast_w in zip(
        jax.tree_util.tree_leaves(jax.device_get(b_init)),
        jax.tree_util.tree_leaves(jax.device_get(res.belief_ema_params)),
        jax.tree_util.tree_leaves(jax.device_get(res.belief_fast_params)),
    ):
        np.testing.assert_array_equal(np.asarray(orig), np.asarray(ema_w))
        np.testing.assert_array_equal(np.asarray(orig), np.asarray(fast_w))

    for row in res.history:
        assert row["belief_loss"] == 0.0
        assert np.isfinite(row["belief_pred_mean"])
        assert np.isfinite(row["p_riichi_legal"])
        assert np.isfinite(row["bc_kl"])
        assert np.isfinite(row["riichi_adv_gap"])


def test_set_injected_lr_zero_copy_preserves_adam_moments():
    """Verify set_injected_lr updates hyperparams['lr'] without copying mu/nu leaves."""
    p_cfg = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    p_net = SovereignPolicyNet(config=p_cfg)
    params = init_policy_params(p_net, jax.random.PRNGKey(3))
    tx_pi, _, _ = build_isolated_optimizers(RNaDConfig(policy_lr=1e-4), OracleConfig(), BeliefConfig())
    opt_st = tx_pi.init(params)

    new_st = set_injected_lr(opt_st, 2.5e-5, n_devices=1)
    assert np.isclose(float(new_st.hyperparams["lr"]), 2.5e-5)

    def _first_adam(node):
        if isinstance(node, optax.ScaleByAdamState):
            return node
        if isinstance(node, (tuple, list)):
            for x in node:
                r = _first_adam(x)
                if r is not None:
                    return r
        return None

    old_adam = _first_adam(opt_st)
    new_adam = _first_adam(new_st)
    assert old_adam is not None and new_adam is not None
    assert new_adam.mu is old_adam.mu
    assert new_adam.nu is old_adam.nu


def test_train_rl_phase2_lr_ramp_rms_reset_and_resume_shape_guard(tmp_path):
    """Verify Phase-2 LR ramp, shield_lambda warmup hold, and Oracle resume shape validation."""
    small_model = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    small_belief = BeliefConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    small_grp = GRPConfig(hidden_dim=32, num_layers=2)
    small_oracle = OracleConfig(num_blocks=1, channels=32, attn_heads=4, global_hidden_dim=32)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(41), 3)
    p_init = init_policy_params(SovereignPolicyNet(config=small_model), k1)
    b_init = _init_belief_params(BeliefNet(config=small_belief), k2)
    g_init = GRPNet(config=small_grp).init(k3, jnp.zeros((2, GRP_FEATURE_DIM), dtype=jnp.float32))

    rl_cfg = RLConfig(
        round_mode="single",
        model=small_model,
        belief=small_belief,
        grp=small_grp,
        oracle=small_oracle,
        rnad=RNaDConfig(
            policy_lr=4e-5,
            warmup_updates=1,
            outer_reset_steps=100,
            reset_warmup_steps=4,
            shield_lambda_start=0.05,
            shield_lambda_end=0.005,
            audit_interval=5,
        ),
        num_envs=4,
        num_steps=16,
        num_updates=3,
        update_epochs=1,
        minibatch_size=32,
        seed=0,
    )
    res = train_rl_v6(
        rl_cfg,
        policy_params_init=p_init,
        belief_params_init=b_init,
        grp_params_init=g_init,
        enable_self_healing=True,
        multi_gpu=False,
        verbose=False,
    )
    # upd 1 is Phase 2 entry (phase2_step = 0): LR must be ramped (< 4e-5) & shield_lambda at start (0.05)
    assert np.isclose(res.history[1]["eff_policy_lr"], 4e-5 * (0.1 + 0.9 * 1.0 / 4.0), rtol=1e-5)
    assert np.isclose(res.history[2]["eff_policy_lr"], 4e-5 * (0.1 + 0.9 * 2.0 / 4.0), rtol=1e-5)
    assert np.isclose(res.history[1]["shield_lambda"], 0.05, atol=1e-6)

    # Incompatible Policy OR Oracle checkpoint shape MUST raise ValueError cleanly before XLA compile
    bad_pol_ckpt = tmp_path / "bad_pol_resume.pkl"
    with open(bad_pol_ckpt, "wb") as f:
        pickle.dump({"completed_updates": 1, "params_pi": {"bad_leaf": np.zeros((2, 2))}}, f)
    with pytest.raises(ValueError, match="Incompatible Policy checkpoint shapes"):
        train_rl_v6(
            rl_cfg,
            policy_params_init=p_init,
            belief_params_init=b_init,
            grp_params_init=g_init,
            resume_path=bad_pol_ckpt,
            multi_gpu=False,
            verbose=False,
        )

    bad_ckpt = tmp_path / "bad_resume.pkl"
    with open(bad_ckpt, "wb") as f:
        pickle.dump({"completed_updates": 1, "params_oracle": {"bad_leaf": np.zeros((2, 2))}}, f)
    with pytest.raises(ValueError, match="Incompatible Oracle checkpoint shapes"):
        train_rl_v6(
            rl_cfg,
            policy_params_init=p_init,
            belief_params_init=b_init,
            grp_params_init=g_init,
            resume_path=bad_ckpt,
            multi_gpu=False,
            verbose=False,
        )

    # Host-only & bounded outer_snapshots (excludes untrained BC params) + hard abort stop
    res_snaps = train_rl_v6(
        RLConfig(
            round_mode="single",
            model=small_model,
            belief=small_belief,
            grp=small_grp,
            oracle=small_oracle,
            rnad=RNaDConfig(
                warmup_updates=1,
                outer_reset_steps=1,
                reset_warmup_steps=1,
                max_outer_snapshots=2,
                audit_interval=5,
            ),
            num_envs=4,
            num_steps=8,
            num_updates=5,
            update_epochs=1,
            minibatch_size=16,
            seed=1,
        ),
        policy_params_init=p_init,
        belief_params_init=b_init,
        grp_params_init=g_init,
        enable_self_healing=True,
        multi_gpu=False,
        verbose=False,
    )
    assert len(res_snaps.outer_snapshots) == 2
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree_util.tree_leaves(res_snaps.outer_snapshots[-1]))
    # Ensure initial untrained params_bc was NOT retained in outer_snapshots
    bc_diffs = [
        max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(jax.tree_util.tree_leaves(snap), jax.tree_util.tree_leaves(p_init))
        )
        for snap in res_snaps.outer_snapshots
    ]
    assert all(d > 0.0 for d in bc_diffs)

    # Self-healing abort MUST restore healthy snapshot & stop training loop early
    res_abort = train_rl_v6(
        RLConfig(
            round_mode="single",
            model=small_model,
            belief=small_belief,
            grp=small_grp,
            oracle=small_oracle,
            rnad=RNaDConfig(
                warmup_updates=1,
                outer_reset_steps=100,
                max_approx_kl=-1.0,
                max_consecutive_heals=1,
            ),
            num_envs=4,
            num_steps=8,
            num_updates=6,
            update_epochs=1,
            minibatch_size=16,
            seed=2,
        ),
        policy_params_init=p_init,
        belief_params_init=b_init,
        grp_params_init=g_init,
        enable_self_healing=True,
        multi_gpu=False,
        verbose=False,
    )
    assert len(res_abort.history) == 3  # upd 0 warmup, upd 1 rollback (heals=1), upd 2 abort (heals=2 > 1) -> stops!
    max_pi_drift = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(jax.tree_util.tree_leaves(res_abort.policy_params), jax.tree_util.tree_leaves(p_init))
    )
    assert max_pi_drift == 0.0


def test_pfsp_seat_rotation_and_bridge_unknown_prior():
    """Verify PFSP rotates learning seats across 0..3 and unknown-opponent bridge prior is 7d."""
    _, is_pfsp, learn_mask = sample_meta_table_pool(32, jax.random.PRNGKey(0))
    pfsp_rows = np.asarray(learn_mask)[np.asarray(is_pfsp)]
    assert pfsp_rows.shape[0] >= 8
    seat_counts = pfsp_rows.sum(axis=0)
    assert np.all(seat_counts >= 2)
    assert np.all(pfsp_rows.sum(axis=1) == 1)

    small_model = ModelConfig(channels=16, num_blocks=1, river_dim=8, attn_heads=2, global_hidden_dim=32)
    small_belief = BeliefConfig(channels=16, num_blocks=1, river_dim=8, attn_heads=2, global_hidden_dim=32)
    small_grp = GRPConfig(hidden_dim=32, num_layers=2, sinkhorn_iters=10)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
    pol_m = SovereignPolicyNet(config=small_model)
    bel_m = BeliefNet(config=small_belief)
    grp_m = GRPNet(config=small_grp)
    p_pol = init_policy_params(pol_m, k1)
    p_bel = _init_belief_params(bel_m, k2)
    p_grp = grp_m.init(k3, jnp.zeros((2, GRP_FEATURE_DIM), dtype=jnp.float32))

    captured_w_opp = []

    class _RecordingBelief:
        def apply(self, params, **kwargs):
            captured_w_opp.append(kwargs["w_opp_5d"])
            return bel_m.apply(params, **kwargs)

    actor_10d = make_v6_policy_actor(
        pol_m,
        p_pol,
        _RecordingBelief(),
        p_bel,
        grp_m,
        p_grp,
        w_rule=W_PHOENIX_10D,
        opp_rules_known=False,
        greedy=True,
    )
    st0 = mahjax.make("red_4p", round_mode="half", observe_type="dict", round_end="confirm").init(jax.random.PRNGKey(0))
    _ = actor_10d(st0, jax.random.PRNGKey(1))
    assert len(captured_w_opp) == 1
    w5 = np.asarray(captured_w_opp[0])
    np.testing.assert_allclose(w5[:, :4], np.broadcast_to(np.asarray(W_PHOENIX_7D)[None, :], (3, 4)), atol=1e-6)
    np.testing.assert_array_equal(w5[:, 4], np.zeros((3,), dtype=np.float32))


def test_desync_hanchan_envs_spreads_rounds():
    """Verify _desync_hanchan_envs spreads parallel hanchans across rounds/steps without terminals."""
    env = mahjax.make("red_4p", round_mode="half", observe_type="dict", round_end="auto")
    k_init, k_desync = jax.random.split(jax.random.PRNGKey(55))
    st0 = jax.jit(jax.vmap(env.init))(jax.random.split(k_init, 8))
    st_desync = _desync_hanchan_envs(env, st0, k_desync, num_envs=8, round_mode="half")
    rounds = np.asarray(jax.device_get(st_desync.round_state.round)).tolist()
    steps = np.asarray(jax.device_get(st_desync.step_count)).tolist()
    terms = np.asarray(jax.device_get(st_desync.terminated | st_desync.truncated))
    assert not np.any(terms)
    assert len(set(zip(rounds, steps))) >= 5
    assert max(rounds) >= 1

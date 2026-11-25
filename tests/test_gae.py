"""Unit tests for turn-based multi-player GAE, Tenhou rank rewards, and GRP PBRS."""

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjax.wrappers.auto_reset_wrapper import auto_reset
from mahjong_rl.algo.gae import (
    apply_grp_pbrs,
    calculate_turn_based_gae,
    compute_terminal_rank_rewards,
)
from mahjong_rl.features.obs_2d import clean_auto_reset_state


def test_single_episode_exact_mc_and_td0():
    """Verify exact MC returns (gamma=1, lambda=1) and TD(0) (lambda=0) for acting and non-acting seats."""
    players = jnp.array([[0, 1, 2, 0, 1]], dtype=jnp.int32)
    values = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=jnp.float32)
    dones = jnp.array([[False, False, False, False, True]], dtype=jnp.bool_)
    rewards = jnp.array(
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-40.0, 130.0, -40.0, -40.0],
        ]],
        dtype=jnp.float32,
    )

    adv_mc, tgt_mc, mask_mc = calculate_turn_based_gae(rewards, values, dones, players, gamma=1.0, gae_lambda=1.0)
    expected_mc_targets = [(0, -40.0), (1, 120.0), (2, -40.0), (0, -40.0), (1, 130.0)]
    for t, (p, expected_tgt) in enumerate(expected_mc_targets):
        assert bool(mask_mc[0, t, p])
        assert int(mask_mc[0, t].sum()) == 1
        np.testing.assert_allclose(float(tgt_mc[0, t, p]), expected_tgt, atol=1e-5)
        np.testing.assert_allclose(float(adv_mc[0, t, p]), expected_tgt - float(values[0, t]), atol=1e-5)

    adv_td, tgt_td, mask_td = calculate_turn_based_gae(rewards, values, dones, players, gamma=1.0, gae_lambda=0.0)
    expected_td_targets = [(0, 0.4), (1, -9.5), (2, -40.0), (0, -40.0), (1, 130.0)]
    for t, (p, expected_tgt) in enumerate(expected_td_targets):
        assert bool(mask_td[0, t, p])
        np.testing.assert_allclose(float(tgt_td[0, t, p]), expected_tgt, atol=1e-5)
        np.testing.assert_allclose(float(adv_td[0, t, p]), expected_tgt - float(values[0, t]), atol=1e-5)


def test_no_cross_episode_leakage_in_either_direction():
    """Episode 1 terminal steps must not see Episode 2; Episode 2 step 0 must see Episode 2's future."""
    players = jnp.array([[0, 1, 2, 0, 1, 0]], dtype=jnp.int32)
    values = jnp.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]], dtype=jnp.float32)
    dones = jnp.array([[False, False, True, False, False, True]], dtype=jnp.bool_)
    rewards = jnp.array(
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -80.0, 80.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [120.0, -40.0, -40.0, -40.0],
        ]],
        dtype=jnp.float32,
    )

    adv, tgt, mask = calculate_turn_based_gae(rewards, values, dones, players, gamma=0.9, gae_lambda=0.95)
    assert int(mask.sum()) == 6
    np.testing.assert_allclose(float(tgt[0, 0, 0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(float(tgt[0, 1, 1]), -80.0, atol=1e-5)
    np.testing.assert_allclose(float(tgt[0, 2, 2]), 80.0, atol=1e-5)

    td_5 = 120.0 - 30.0
    td_3 = 0.0 + 0.9 * 30.0 - 10.0
    expected_adv_3 = td_3 + 0.9 * 0.95 * td_5
    np.testing.assert_allclose(float(adv[0, 3, 0]), expected_adv_3, atol=1e-4)
    np.testing.assert_allclose(float(tgt[0, 4, 1]), -40.0, atol=1e-5)


def test_unterminated_tail_no_garbage_gae_leakage():
    """Invalid rollout-tail steps must not leak `-value` into earlier valid steps."""
    players = jnp.array([[0, 1, 0, 0]], dtype=jnp.int32)
    values = jnp.array([[0.1, 0.2, 0.5, 0.9]], dtype=jnp.float32)
    dones = jnp.array([[False, False, False, False]], dtype=jnp.bool_)
    rewards = jnp.array(
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0, 0.0],
        ]],
        dtype=jnp.float32,
    )

    gamma, lam = 0.9, 0.95
    adv, tgt, mask = calculate_turn_based_gae(rewards, values, dones, players, gamma=gamma, gae_lambda=lam)
    assert not bool(mask[0, 3, 0])
    assert not bool(mask[0, 1, 1])
    assert bool(mask[0, 2, 0]) and bool(mask[0, 0, 0])
    expected_adv_2 = 1.0 + gamma * 0.9 - 0.5
    np.testing.assert_allclose(float(adv[0, 2, 0]), expected_adv_2, atol=1e-5)
    np.testing.assert_allclose(float(adv[0, 0, 0]), (gamma * 0.5 - 0.1) + gamma * lam * expected_adv_2, atol=1e-5)


def test_grp_pbrs_telescopes_exactly_with_terminal_zero():
    """For gamma=1, PBRS shaping over a complete episode sums to -Phi(s_0) (policy-invariant)."""
    # 3-step episode ending at t=2
    raw_rew = jnp.array([[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [90.0 / 135.0, 45.0 / 135.0, 0.0, -1.0],
    ]], dtype=jnp.float32)
    phi_curr = jnp.array([[
        [0.1, -0.1, 0.0, 0.0],
        [0.3, 0.2, -0.1, -0.4],
        [0.5, 0.2, 0.0, -0.7],
    ]], dtype=jnp.float32)
    phi_next = jnp.array([[
        [0.3, 0.2, -0.1, -0.4],
        [0.5, 0.2, 0.0, -0.7],
        [99.0, 99.0, 99.0, 99.0],  # must be masked to 0 because done=True!
    ]], dtype=jnp.float32)
    dones = jnp.array([[False, False, True]], dtype=jnp.bool_)

    shaped = apply_grp_pbrs(raw_rew, phi_curr, phi_next, dones, gamma=1.0, pbrs_coef=1.0)
    # Sum of shaped rewards over episode == raw_terminal_reward - phi_curr[0]
    np.testing.assert_allclose(
        np.asarray(shaped[0].sum(axis=0)),
        np.asarray(raw_rew[0, 2] - phi_curr[0, 0]),
        atol=1e-5,
    )


def test_terminal_rank_rewards_tenhou_tiebreak():
    """Verify Tenhou rank points [90, 45, 0, -135] with init_wind tie-break."""
    env = mahjax.make("red_4p", round_mode="half")
    st = env.init(jax.random.PRNGKey(0))
    # Force a tie between seat 1 (init_wind=2) and seat 2 (init_wind=0)
    rs = st.round_state.replace(
        score=jnp.array([400, 250, 250, 100], dtype=jnp.int32),
        init_wind=jnp.array([1, 2, 0, 3], dtype=jnp.int8),
    )
    st = st.replace(round_state=rs)

    rew_done = compute_terminal_rank_rewards(st, jnp.bool_(True), (90.0, 45.0, 0.0, -135.0), 135.0)
    rew_live = compute_terminal_rank_rewards(st, jnp.bool_(False), (90.0, 45.0, 0.0, -135.0), 135.0)
    # Seat 0 -> 1st (+90/135), Seat 2 -> 2nd (+45/135, smaller init_wind), Seat 1 -> 3rd (0), Seat 3 -> 4th (-1)
    np.testing.assert_allclose(np.asarray(rew_done), [90.0 / 135.0, 0.0, 45.0 / 135.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(rew_live), [0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_auto_reset_rollout_integration_and_clean_first_obs():
    """Verify real MahJax auto_reset rollout produces clean first-step obs and valid GAE."""
    env = mahjax.make("no_red_4p", round_mode="single", observe_type="dict")
    step_fn = auto_reset(env.step, env.init)

    def rollout_one(state, rng):
        def body(carry, _):
            st, k = carry
            k, k_act, k_step = jax.random.split(k, 3)
            obs = env.observe(clean_auto_reset_state(st))
            cp = st.current_player.astype(jnp.int32)
            action = mahjax.rule_based_player(st, k_act)
            next_st = step_fn(st, action, k_step)
            done = (next_st.terminated | next_st.truncated).astype(jnp.bool_)
            rew = next_st.rewards.astype(jnp.float32)
            return (next_st, k), (rew, done, cp, obs["is_discard_node"], obs["last_draw"])

        _, outs = jax.lax.scan(body, (state, rng), None, length=120)
        return outs

    init_states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 4))
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    rewards, dones, players, is_discard_nodes, last_draws = jax.jit(jax.vmap(rollout_one))(init_states, keys)

    done_coords = np.argwhere(np.asarray(dones[:, :-1]))
    assert len(done_coords) > 0
    for b, t in done_coords:
        assert bool(is_discard_nodes[b, t + 1])
        assert int(last_draws[b, t + 1]) >= 0

    adv, tgt, mask = jax.jit(calculate_turn_based_gae)(
        rewards, jnp.zeros_like(players, dtype=jnp.float32), dones, players, gamma=1.0, gae_lambda=1.0
    )
    assert jnp.all(jnp.isfinite(adv)) and jnp.all(jnp.isfinite(tgt)) and int(mask.sum()) > 0

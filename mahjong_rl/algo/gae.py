"""Turn-based multi-player GAE, Tenhou terminal rank rewards, and GRP potential shaping."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from mahjax.mahjong_4p.constants import NUM_PLAYERS
from mahjax.mahjong_4p.state import State
from mahjong_rl.models.grp import scores_to_rank_matrix_4x4


def compute_terminal_rank_matrix_4x4(
    scores: jnp.ndarray,
    init_wind: jnp.ndarray,
    done: jnp.ndarray,
) -> jnp.ndarray:
    """Return exact `(4, 4)` 1-hot permutation matrix on terminal steps (uniform 0.25 otherwise)."""
    perm = scores_to_rank_matrix_4x4(scores, init_wind)
    uniform = jnp.full((NUM_PLAYERS, NUM_PLAYERS), 0.25, dtype=jnp.float32)
    return jnp.where(done, perm, uniform)


def calculate_dual_rate_simplex_td_lambda(
    v_oracle_4x4: jnp.ndarray,
    v_bootstrap_4x4: jnp.ndarray,
    grp_ema_next_4x4: jnp.ndarray,
    terminal_rank_4x4: jnp.ndarray,
    dones: jnp.ndarray,
    round_ended: jnp.ndarray,
    lambda_intra: float = 1.0,
    lambda_inter: float = 0.70,
) -> jnp.ndarray:
    """Birkhoff-closed Dual-Rate Simplex TD(lambda) targets `(B, T, 4, 4)`.

    Preserves exact Birkhoff polytope closure (`row_sum == 1`, `col_sum == 1`) at every step
    via convex combinations of doubly-stochastic matrices:
    - If `done[b, t]`: `target = terminal_rank_4x4[b, t]` (exact 1-hot permutation, zero cross-episode leak!)
    - Else if `round_ended[b, t]`: `target = (1 - lambda_inter) * GRP_EMA(s_{t+1}) + lambda_inter * next_target`
    - Else (intra-round):
      * when `lambda_intra >= 1.0` (default): `target = next_target` (exact round-horizon return, zero intra-round Oracle bootstrapping)
      * otherwise: `target = (1 - lambda_intra) * V_oracle(s_{t+1}) + lambda_intra * next_target`

    Args:
        v_oracle_4x4: `(B, T, 4, 4)` Oracle placement matrices at `s_t`.
        v_bootstrap_4x4: `(B, 4, 4)` Oracle placement matrix at `s_T` (post-rollout state).
        grp_ema_next_4x4: `(B, T, 4, 4)` `GRP_EMA` placement matrices at `s_{t+1}`.
        terminal_rank_4x4: `(B, T, 4, 4)` exact permutation matrices when `dones[b, t]` is True.
        dones: `(B, T)` bool, True on the LAST step of a hanchan.
        round_ended: `(B, T)` bool, True when step `t` ended a hand/round.
        lambda_intra: Intra-round TD(lambda) decay (default `1.0`).
        lambda_inter: Cross-round TD(lambda) blend with `GRP_EMA` (default `0.70`).

    Returns:
        targets_4x4: `(B, T, 4, 4)` float32 doubly-stochastic TD targets.
    """
    lam_in = float(lambda_intra)
    lam_out = float(lambda_inter)

    def single_env(
        v_seq: jnp.ndarray,
        v_boot: jnp.ndarray,
        grp_next_seq: jnp.ndarray,
        term_seq: jnp.ndarray,
        done_seq: jnp.ndarray,
        rend_seq: jnp.ndarray,
    ) -> jnp.ndarray:
        # Next-step Oracle estimate V_oracle(s_{t+1}) for t = 0..T-1
        v_next_seq = jnp.concatenate([v_seq[1:], v_boot[None, :, :]], axis=0)  # (T, 4, 4)

        def scan_fn(next_target, inputs):
            v_next, grp_next, term_mat, is_done, is_rend = inputs
            intra_t = next_target if lam_in >= 1.0 else (1.0 - lam_in) * v_next + lam_in * next_target
            inter_t = (1.0 - lam_out) * grp_next + lam_out * next_target
            target_t = jnp.where(is_done, term_mat, jnp.where(is_rend, inter_t, intra_t))
            return target_t, target_t

        _, targets = lax.scan(
            scan_fn,
            v_boot,
            (v_next_seq, grp_next_seq, term_seq, done_seq, rend_seq),
            reverse=True,
        )
        return targets

    return jax.vmap(single_env)(
        v_oracle_4x4,
        v_bootstrap_4x4,
        grp_ema_next_4x4,
        terminal_rank_4x4,
        dones,
        round_ended,
    )


def compute_round_outcome_targets(
    scores_curr: jnp.ndarray,         # (B, T, 4) int32/float32 absolute scores at s_t
    scores_post: jnp.ndarray,         # (B, T, 4) int32/float32 absolute scores right after step t
    round_ended: jnp.ndarray,         # (B, T) bool
    dones: jnp.ndarray,               # (B, T) bool
    step_winner_abs: jnp.ndarray,     # (B, T) int32 in {0..3} if TSUMO/RON else 4 (draw)
    current_players: jnp.ndarray,     # (B, T) int32 acting seat at s_t
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ego-frame Dual-Horizon auxiliary targets for `OracleCriticNet`.

    Returns:
        winner_ego: `(B, T)` int32 in `{0=me, 1=right, 2=across, 3=left, 4=draw}`
        score_delta_ego: `(B, T, 4)` float32 `(round_end_scores - scores_curr) / 100.0` in ego frame
        aux_valid: `(B, T)` bool, True iff the round terminated within the rollout window.
    """
    def single_env(sc_c, sc_p, rend, dn, win_abs, cp_seq):
        def scan_fn(carry, inputs):
            end_scores, end_win, has_end = carry
            sc_cur, sc_pst, is_rend, is_dn, w_abs, cp = inputs

            boundary = is_rend | is_dn
            end_scores = jnp.where(boundary, sc_pst.astype(jnp.float32), end_scores)
            end_win = jnp.where(boundary, w_abs.astype(jnp.int32), end_win)
            has_end = jnp.where(boundary, jnp.bool_(True), has_end)

            delta_abs = (end_scores - sc_cur.astype(jnp.float32)) / 100.0  # 10k-pt units
            order = (jnp.arange(NUM_PLAYERS, dtype=jnp.int32) + cp.astype(jnp.int32)) % NUM_PLAYERS
            delta_ego = jnp.where(has_end, delta_abs[order], 0.0)
            win_ego = jnp.where(
                end_win < NUM_PLAYERS,
                (end_win - cp.astype(jnp.int32)) % NUM_PLAYERS,
                jnp.int32(NUM_PLAYERS),
            )
            return (end_scores, end_win, has_end), (win_ego, delta_ego, has_end)

        init_carry = (
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
            jnp.int32(NUM_PLAYERS),
            jnp.bool_(False),
        )
        _, (w_ego, d_ego, valid) = lax.scan(
            scan_fn,
            init_carry,
            (sc_c, sc_p, rend, dn, win_abs, cp_seq),
            reverse=True,
        )
        return w_ego, d_ego, valid

    return jax.vmap(single_env)(
        scores_curr, scores_post, round_ended, dones, step_winner_abs, current_players
    )


def compute_terminal_rank_rewards(
    state: State,
    done: jnp.ndarray,
    rank_points: Tuple[float, float, float, float] = (90.0, 45.0, 0.0, -135.0),
    reward_scale: float = 135.0,
) -> jnp.ndarray:
    """Compute exact Tenhou placement rewards when a hanchan terminates (zeros otherwise).

    Ties are broken by `init_wind` (seat order at game start), matching Tenhou and MahJax.
    """
    scores = jnp.asarray(state.round_state.score)  # (4,)
    init_wind = jnp.asarray(state.round_state.init_wind)  # (4,)
    order = jnp.lexsort((init_wind, -scores))  # 0th element is 1st place
    pts_table = jnp.asarray(rank_points, dtype=jnp.float32) / float(reward_scale)
    rank_rew = jnp.zeros_like(pts_table).at[order].set(pts_table)
    return jnp.where(done, rank_rew, jnp.zeros_like(rank_rew))


def apply_grp_pbrs(
    rewards: jnp.ndarray,
    phi_curr: jnp.ndarray,
    phi_next: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 1.0,
    pbrs_coef: float = 1.0,
) -> jnp.ndarray:
    """Potential-Based Reward Shaping (PBRS) using GRP expected final rank points.

        r'_t = r_t + pbrs_coef * (gamma * (1 - done_t) * Phi(s_{t+1}) - Phi(s_t))

    At terminal steps (`done_t == True`), Phi(s_terminal) is strictly 0, preserving
    policy invariance while turning sparse hanchan payoffs into dense per-round signals.

    Args:
        rewards: (..., P) raw/normalized environment rewards.
        phi_curr: (..., P) GRP potential Phi(s_t) for all players.
        phi_next: (..., P) GRP potential Phi(s_{t+1}) for all players.
        dones: (...) bool, True on the last step of an episode.
        gamma: Discount factor.
        pbrs_coef: Scaling weight (0.0 disables shaping).
    """
    not_done = (~dones).astype(jnp.float32)[..., None]
    shaping = gamma * not_done * phi_next - phi_curr
    return rewards + float(pbrs_coef) * shaping


def calculate_turn_based_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    current_players: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) for turn-based multi-player games.

    Fixes both silent defects found in naive MahJax rollouts:
    1. Resets episode accumulators on `dones[b, t]` (the LAST step of an episode, from
       `next_state.terminated | next_state.truncated`), preventing cross-episode leakage.
    2. Stores masked `advantage` (0.0 on invalid rollout-tail steps) into `gae[player]`
       rather than raw `new_gae`, preventing dummy `next_value=0` noise from leaking
       into earlier valid steps of an unterminated episode.

    Args:
        rewards: (B, T, P) float32 rewards delivered at step t.
        values: (B, T) float32 value estimates V(s_t) for current_players[b, t].
        dones: (B, T) bool, True on the last step of an episode.
        current_players: (B, T) int32 acting seat at step t.
        gamma: Per-decision discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        advantages: (B, T, P) float32
        targets: (B, T, P) float32
        valid_mask: (B, T, P) bool
    """

    def single_env_gae(
        ep_rewards: jnp.ndarray,
        ep_values: jnp.ndarray,
        ep_dones: jnp.ndarray,
        ep_players: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        num_players = ep_rewards.shape[-1]

        def scan_fn(carry, inputs):
            gae, next_value, reward_accum, has_next_value, ep_ended = carry
            reward, value, done, player = inputs

            # Reset carry belonging to the NEXT episode before processing this terminal step
            gae = jnp.where(done, jnp.zeros_like(gae), gae)
            next_value = jnp.where(done, jnp.zeros_like(next_value), next_value)
            reward_accum = jnp.where(done, jnp.zeros_like(reward_accum), reward_accum)
            has_next_value = jnp.where(done, jnp.zeros_like(has_next_value), has_next_value)
            ep_ended = jnp.where(done, jnp.bool_(True), ep_ended)

            reward_accum = reward_accum + reward
            player_reward = reward_accum[player]
            reward_accum = reward_accum.at[player].set(0.0)

            is_valid = has_next_value[player] | ep_ended

            td_error = player_reward + gamma * next_value[player] - value
            new_gae = td_error + gamma * gae_lambda * gae[player]

            advantage = jnp.where(is_valid, new_gae, 0.0)
            target = jnp.where(is_valid, advantage + value, value)

            # Never propagate raw `new_gae` from an invalid tail step
            gae = gae.at[player].set(advantage)
            next_value = next_value.at[player].set(value)
            has_next_value = has_next_value.at[player].set(jnp.bool_(True))

            new_carry = (gae, next_value, reward_accum, has_next_value, ep_ended)
            output = (
                jnp.zeros(num_players, dtype=jnp.float32).at[player].set(advantage),
                jnp.zeros(num_players, dtype=jnp.float32).at[player].set(target),
                jnp.zeros(num_players, dtype=jnp.bool_).at[player].set(is_valid),
            )
            return new_carry, output

        init_carry = (
            jnp.zeros(num_players, dtype=jnp.float32),
            jnp.zeros(num_players, dtype=jnp.float32),
            jnp.zeros(num_players, dtype=jnp.float32),
            jnp.zeros(num_players, dtype=jnp.bool_),
            jnp.bool_(False),
        )
        _, (adv, targets, valid_mask) = lax.scan(
            scan_fn,
            init_carry,
            (ep_rewards, ep_values, ep_dones, ep_players),
            reverse=True,
        )
        return adv, targets, valid_mask

    return jax.vmap(single_env_gae)(rewards, values, dones, current_players)

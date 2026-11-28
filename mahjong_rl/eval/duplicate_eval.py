"""Fixed-wall Duplicate 1v3 evaluation protocol with paired 95% confidence intervals.

Each wall seed is played 4 times with the candidate agent rotated across all 4 seats
(0, 1, 2, 3) against 3 copies of the baseline agent. Using `round_end="confirm"` (with
DUMMY actions auto-played outside the policy) ensures every single hand of a full
hanchan is captured accurately in hora / deal-in / riichi / meld telemetry.

Performance notes (why this is fast on a GPU):
- All `num_walls * 4` seat rotations of one chunk are flattened into a SINGLE batched
  lane axis, so the policy/belief/GRP forwards see a real GPU-sized batch instead of
  `4` lanes at a time.
- The whole chunk runs inside one `jax.lax.while_loop` on device; it exits as soon as
  every lane has terminated (instead of always stepping `max_steps`), and no per-step
  value ever crosses back to the host.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import mahjax
from mahjax.common.action import Action
from mahjong_rl.algo.sovereign_v6 import shard_for_pmap
from mahjong_rl.config import EvalConfig


def _expand_wall_batch(x: jnp.ndarray, num_players: int) -> jnp.ndarray:
    """`(W, ...) -> (W * num_players, ...)`, repeating each wall once per seat rotation."""
    return jnp.repeat(x[:, None], num_players, axis=1).reshape(
        (x.shape[0] * num_players,) + x.shape[1:]
    )


def _play_duplicate_chunk_core(
    env,
    joint_step: Callable[[Any, jax.Array, jax.Array], Tuple[jnp.ndarray, jnp.ndarray]],
    rank_pts_arr: jnp.ndarray,
    init_keys: jax.Array,
    step_seeds: jax.Array,
    max_steps: int,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Play one chunk of fixed-wall duplicate games with a single batched while-loop.

    Args:
        joint_step: `(states, k_cand, k_base) -> (cand_actions, base_actions)` on the
            flattened `(walls * 4)` lane batch. `play_duplicate_chunk` wraps two
            independent actors; `play_duplicate_chunk_joint` lets one actor share
            feature encoding between both policies.
        init_keys: `(W,)` per-wall deck keys.
        step_seeds: `(W,)` per-wall action/step keys (shared by all 4 seat rotations).
        max_steps: hard safety cap; the loop normally exits when all lanes terminate.

    Returns:
        totals:     `{key: (W, 4) int32/float32}`
        rank_idx:   `(W, 4)` int32, 0 = 1st .. 3 = 4th
        pt:         `(W, 4)` float32 Tenhou placement points
        finished:   `(W, 4)` int32, 1 iff the lane terminated before `max_steps`
    """
    num_walls = init_keys.shape[0]
    num_players = int(env.num_players)
    lanes = num_walls * num_players

    init_states = jax.vmap(env.init)(init_keys)  # (W, ...)
    states = jax.tree_util.tree_map(lambda x: _expand_wall_batch(x, num_players), init_states)
    agent_seats = jnp.tile(jnp.arange(num_players, dtype=jnp.int32), (num_walls,))
    wall_seeds = jnp.repeat(step_seeds, num_players, axis=0)
    rngs = wall_seeds

    totals = {
        "hands": jnp.zeros((lanes,), dtype=jnp.int32),
        "won": jnp.zeros((lanes,), dtype=jnp.int32),
        "dealt_in": jnp.zeros((lanes,), dtype=jnp.int32),
        "riichi": jnp.zeros((lanes,), dtype=jnp.int32),
        "melded": jnp.zeros((lanes,), dtype=jnp.int32),
        "won_pts": jnp.zeros((lanes,), dtype=jnp.float32),
        "dealt_in_pts": jnp.zeros((lanes,), dtype=jnp.float32),
    }

    def cond(carry):
        states, _, step_idx, _ = carry
        done = states.terminated | states.truncated
        return jnp.any(~done) & (step_idx < int(max_steps))

    def body(carry):
        states, rngs, step_idx, totals = carry
        done = states.terminated | states.truncated
        step_keys = jax.vmap(lambda k: jax.random.split(k, 3))(rngs)  # (L, 3, 2)
        rngs, k1, k2 = step_keys[:, 0], step_keys[:, 1], step_keys[:, 2]
        # Deterministic kyoku-aligned wall key so all 4 duplicate rotations see identical
        # deals on identical kyokus regardless of turn-count drift!
        kyoku_id = states.round_state.round.astype(jnp.int32) * 64 + jnp.minimum(
            states.round_state.honba.astype(jnp.int32), 63
        )
        k_step = jax.vmap(jax.random.fold_in)(wall_seeds, kyoku_id)

        is_dummy = states.legal_action_mask[:, Action.DUMMY]
        cp = states.current_player.astype(jnp.int32)
        a_cand, a_base = joint_step(states, k1, k2)
        a_cand = a_cand.astype(jnp.int32)
        a_base = a_base.astype(jnp.int32)
        policy_act = jnp.where(cp == agent_seats, a_cand, a_base)
        action = jnp.where(is_dummy, jnp.int32(Action.DUMMY), policy_act).astype(jnp.int32)

        raw_next = jax.vmap(env.step)(states, action, k_step)
        # Freeze terminated lanes exactly like the per-lane `lax.cond` did.
        next_states = jax.tree_util.tree_map(
            lambda old, new: jnp.where(
                done.reshape((-1,) + (1,) * (old.ndim - 1)), old, new
            ),
            states,
            raw_next,
        )

        # Rising edge of terminated_round captures EVERY hand in confirm mode.
        hand_ended = (
            ~done
            & next_states.round_state.terminated_round
            & ~states.round_state.terminated_round
        )
        p = next_states.players
        won = p.has_won
        riichi = p.riichi
        melded = ~p.is_hand_concealed
        rewards = next_states.rewards.astype(jnp.float32)
        is_hora = won.any(axis=-1)  # (L,)
        is_ron = is_hora & ((rewards < 0.0).sum(axis=-1) == 1)  # (L,)
        dealt_in_mask = is_ron[:, None] & (rewards < 0.0)  # (L, 4)

        seat_idx = jnp.arange(lanes, dtype=jnp.int32)
        cand_won = hand_ended & won[seat_idx, agent_seats]
        cand_dealt_in = hand_ended & dealt_in_mask[seat_idx, agent_seats]
        cand_riichi = hand_ended & riichi[seat_idx, agent_seats]
        cand_melded = hand_ended & melded[seat_idx, agent_seats]
        cand_rewards = rewards[seat_idx, agent_seats]

        totals = {
            "hands": totals["hands"] + hand_ended.astype(jnp.int32),
            "won": totals["won"] + cand_won.astype(jnp.int32),
            "dealt_in": totals["dealt_in"] + cand_dealt_in.astype(jnp.int32),
            "riichi": totals["riichi"] + cand_riichi.astype(jnp.int32),
            "melded": totals["melded"] + cand_melded.astype(jnp.int32),
            "won_pts": totals["won_pts"] + jnp.where(cand_won, cand_rewards, 0.0),
            "dealt_in_pts": totals["dealt_in_pts"] + jnp.where(cand_dealt_in, -cand_rewards, 0.0),
        }
        return next_states, rngs, step_idx + jnp.int32(1), totals

    states, _, _, totals = lax.while_loop(
        cond, body, (states, rngs, jnp.int32(0), totals)
    )

    scores = states.round_state.score.astype(jnp.float32)  # (L, 4)
    init_wind = states.round_state.init_wind.astype(jnp.int32)  # (L, 4)
    seat_idx = jnp.arange(lanes, dtype=jnp.int32)
    my_score = scores[seat_idx, agent_seats]
    my_wind = init_wind[seat_idx, agent_seats]
    ahead = (scores > my_score[:, None]) | (
        (scores == my_score[:, None]) & (init_wind < my_wind[:, None])
    )
    rank_idx = ahead.sum(axis=1).astype(jnp.int32)  # 0=1st .. 3=4th
    pt = rank_pts_arr[rank_idx]
    finished = (states.terminated | states.truncated).astype(jnp.int32)

    shape = (num_walls, num_players)
    totals = {k: v.reshape(shape) for k, v in totals.items()}
    return totals, rank_idx.reshape(shape), pt.reshape(shape), finished.reshape(shape)


def play_duplicate_chunk(
    env,
    candidate_actor: Callable[[mahjax.State, jax.Array], jnp.ndarray],
    baseline_actor: Callable[[mahjax.State, jax.Array], jnp.ndarray],
    rank_pts_arr: jnp.ndarray,
    init_keys: jax.Array,
    step_seeds: jax.Array,
    max_steps: int,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One chunk with two independent `(state, key) -> action` actors."""

    def joint_step(states, k1, k2):
        return (
            jax.vmap(candidate_actor)(states, k1),
            jax.vmap(baseline_actor)(states, k2),
        )

    return _play_duplicate_chunk_core(
        env, joint_step, rank_pts_arr, init_keys, step_seeds, max_steps
    )


def play_duplicate_chunk_joint(
    env,
    joint_actor: Callable[[mahjax.State, jax.Array, jax.Array], Tuple[jnp.ndarray, jnp.ndarray]],
    rank_pts_arr: jnp.ndarray,
    init_keys: jax.Array,
    step_seeds: jax.Array,
    max_steps: int,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One chunk with `joint_actor(state, k_cand, k_base) -> (cand, base)`.

    Use this when the two policies can share an observation/feature encoding
    (e.g. candidate and baseline are the same V6 network with different weights).
    """

    def joint_step(states, k1, k2):
        return jax.vmap(joint_actor)(states, k1, k2)

    return _play_duplicate_chunk_core(
        env, joint_step, rank_pts_arr, init_keys, step_seeds, max_steps
    )


def evaluate_duplicate_1v3(
    candidate_actor: Callable[[mahjax.State, jax.Array], jnp.ndarray],
    baseline_actor: Callable[[mahjax.State, jax.Array], jnp.ndarray],
    cfg: EvalConfig = EvalConfig(),
    chunk_walls: int = 16,
) -> Dict[str, float]:
    """Run paired fixed-wall Duplicate 1v3 evaluation (4 * cfg.num_walls total games).

    Args:
        candidate_actor: `act(state, key) -> int32` scalar action for the candidate agent.
        baseline_actor: `act(state, key) -> int32` scalar action for the 3 baseline seats.
        cfg: EvalConfig specifying env_name, round_mode, num_walls, max_steps, rank_points.
        chunk_walls: walls played concurrently per launch (`4 * chunk_walls` batched lanes).

    Returns:
        Dictionary of metrics including paired Tenhou pt mean & 95% CI, 1st-4th rank rates,
        average rank, and per-hand hora / deal-in / riichi / meld rates.
    """
    # Use round_end="confirm" so every hand's round-end state is observable before resetting
    env = mahjax.make(
        cfg.env_name,
        round_mode=cfg.round_mode,
        observe_type="dict",
        round_end="confirm",
    )
    num_players = env.num_players
    # Use host NumPy constant so `jax.pmap` can replicate cleanly across all GPUs!
    rank_pts_arr = np.asarray(cfg.rank_points, dtype=np.float32)

    def run_chunk(init_keys, step_seeds):
        return play_duplicate_chunk(
            env,
            candidate_actor,
            baseline_actor,
            jnp.asarray(rank_pts_arr),
            init_keys,
            step_seeds,
            int(cfg.max_steps),
        )

    run_single_dev = jax.jit(run_chunk)
    n_devices = jax.local_device_count()
    run_pmap = jax.pmap(run_chunk) if n_devices > 1 else None

    rng = jax.random.PRNGKey(cfg.seed)
    k_init, k_step = jax.random.split(rng)
    init_keys = jax.random.split(k_init, cfg.num_walls)
    step_seeds = jax.random.split(k_step, cfg.num_walls)

    # Chunk walls across ALL local GPUs (`chunk_walls * n_devices` walls simultaneously!)
    per_dev_w = max(1, min(int(chunk_walls), int(cfg.num_walls)))
    step_w = per_dev_w * n_devices if run_pmap is not None else per_dev_w
    totals_list, ranks_list, pts_list, fin_list = [], [], [], []
    for start in range(0, cfg.num_walls, step_w):
        end = min(start + step_w, cfg.num_walls)
        cnt = end - start
        ik = init_keys[start:end]
        ss = step_seeds[start:end]
        if cnt < step_w:
            # Pad to a fixed shape so every launch reuses ONE compiled executable.
            pad = step_w - cnt
            ik = jnp.concatenate([ik, jnp.repeat(ik[-1:], pad, axis=0)], axis=0)
            ss = jnp.concatenate([ss, jnp.repeat(ss[-1:], pad, axis=0)], axis=0)
        if run_pmap is not None:
            ik_sh = shard_for_pmap(
                ik.reshape((n_devices, step_w // n_devices) + ik.shape[1:]), n_devices
            )
            ss_sh = shard_for_pmap(
                ss.reshape((n_devices, step_w // n_devices) + ss.shape[1:]), n_devices
            )
            t_c, r_c, p_c, f_c = jax.device_get(run_pmap(ik_sh, ss_sh))
            t_c = {k: np.asarray(v).reshape((step_w,) + np.asarray(v).shape[2:])[:cnt] for k, v in t_c.items()}
            r_c = np.asarray(r_c, dtype=np.int32).reshape((step_w, num_players))[:cnt]
            p_c = np.asarray(p_c, dtype=np.float64).reshape((step_w, num_players))[:cnt]
            f_c = np.asarray(f_c, dtype=np.float64).reshape((step_w, num_players))[:cnt]
        else:
            t_c, r_c, p_c, f_c = jax.device_get(run_single_dev(ik, ss))
            t_c = {k: np.asarray(v)[:cnt] for k, v in t_c.items()}
            r_c = np.asarray(r_c, dtype=np.int32)[:cnt]
            p_c = np.asarray(p_c, dtype=np.float64)[:cnt]
            f_c = np.asarray(f_c, dtype=np.float64)[:cnt]
        totals_list.append(t_c)
        ranks_list.append(r_c)
        pts_list.append(p_c)
        fin_list.append(f_c)

    totals = {k: np.concatenate([np.asarray(t[k]) for t in totals_list], axis=0) for k in totals_list[0]}
    rank_idxs_np = np.concatenate(ranks_list, axis=0)
    pts_np = np.concatenate(pts_list, axis=0)
    finished_np = np.concatenate(fin_list, axis=0)

    # Paired duplicate statistics: average across the 4 seat rotations per wall first!
    # Filter to fully-finished walls when at least 1 wall finished all 4 seats so timed-out
    # multi-renchan / sudden-death walls never pollute placement or Tenhou pt averages.
    wall_done = finished_np.all(axis=1)
    finished_wall_ratio = float(wall_done.mean())
    eval_mask = wall_done if bool(wall_done.any()) else np.ones((cfg.num_walls,), dtype=bool)
    wall_mean_pts = pts_np[eval_mask].mean(axis=1)  # (n_eval_walls,)
    pt_mean = float(wall_mean_pts.mean())
    n_eval_w = int(wall_mean_pts.shape[0])
    pt_se = float(wall_mean_pts.std(ddof=1) / np.sqrt(max(n_eval_w, 1))) if n_eval_w > 1 else 0.0
    pt_ci95 = 1.96 * pt_se
    eval_ranks_np = rank_idxs_np[eval_mask]

    total_games = int(rank_idxs_np.size)
    total_hands = float(np.asarray(totals["hands"]).sum())
    denom_hands = max(total_hands, 1.0)
    total_won = float(np.asarray(totals["won"]).sum())
    total_dealt = float(np.asarray(totals["dealt_in"]).sum())

    return {
        "total_games": float(total_games),
        "total_hands": total_hands,
        "finished_game_ratio": float(finished_np.mean()),
        "finished_wall_ratio": finished_wall_ratio,
        "tenhou_pt_mean": pt_mean,
        "tenhou_pt_ci95": pt_ci95,
        "avg_rank": float(eval_ranks_np.mean() + 1.0),
        "rank_1_rate": float((eval_ranks_np == 0).mean()),
        "rank_2_rate": float((eval_ranks_np == 1).mean()),
        "rank_3_rate": float((eval_ranks_np == 2).mean()),
        "rank_4_rate": float((eval_ranks_np == 3).mean()),
        "hora_rate": total_won / denom_hands,
        "deal_in_rate": total_dealt / denom_hands,
        "riichi_rate": float(np.asarray(totals["riichi"]).sum()) / denom_hands,
        "meld_rate": float(np.asarray(totals["melded"]).sum()) / denom_hands,
        "avg_win_pts": float(np.asarray(totals["won_pts"]).sum()) / max(total_won, 1.0),
        "avg_deal_in_pts": float(np.asarray(totals["dealt_in_pts"]).sum()) / max(total_dealt, 1.0),
    }

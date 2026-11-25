"""Fixed-wall Duplicate 1v3 evaluation protocol with paired 95% confidence intervals.

Each wall seed is played 4 times with the candidate agent rotated across all 4 seats
(0, 1, 2, 3) against 3 copies of the baseline agent. Using `round_end="confirm"` (with
DUMMY actions auto-played outside the policy) ensures every single hand of a full
hanchan is captured accurately in hora / deal-in / riichi / meld telemetry.
"""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import mahjax
from mahjax.common.action import Action
from mahjong_rl.config import EvalConfig


def evaluate_duplicate_1v3(
    candidate_actor: Callable[[mahjax.State, jax.Array], jnp.ndarray],
    baseline_actor: Callable[[mahjax.State, jax.Array], jnp.ndarray],
    cfg: EvalConfig = EvalConfig(),
    chunk_walls: int = 8,
) -> Dict[str, float]:
    """Run paired fixed-wall Duplicate 1v3 evaluation (4 * cfg.num_walls total games).

    Args:
        candidate_actor: `act(state, key) -> int32` scalar action for the candidate agent.
        baseline_actor: `act(state, key) -> int32` scalar action for the 3 baseline seats.
        cfg: EvalConfig specifying env_name, round_mode, num_walls, max_steps, rank_points.

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
    rank_pts_arr = jnp.asarray(cfg.rank_points, dtype=jnp.float32)

    def play_single_seat(init_state, agent_seat: jnp.ndarray, wall_step_seed: jax.Array):
        def step_body(carry, step_idx):
            state, rng = carry
            done = state.terminated | state.truncated
            rng, k1, k2 = jax.random.split(rng, 3)
            # Derive deterministic per-step wall key from wall_step_seed so all 4 duplicate
            # rotations draw from identical walls on identical round transitions
            k_step = jax.random.fold_in(wall_step_seed, step_idx)

            is_dummy = state.legal_action_mask[Action.DUMMY]
            cp = state.current_player.astype(jnp.int32)
            a_cand = candidate_actor(state, k1).astype(jnp.int32)
            a_base = baseline_actor(state, k2).astype(jnp.int32)
            policy_act = jnp.where(cp == agent_seat, a_cand, a_base)
            action = jnp.where(is_dummy, jnp.int32(Action.DUMMY), policy_act)

            next_state = lax.cond(
                done,
                lambda: state,
                lambda: env.step(state, action, k_step),
            )

            # Rising edge of terminated_round captures EVERY hand in confirm mode
            hand_ended = (
                ~done
                & next_state.round_state.terminated_round
                & ~state.round_state.terminated_round
            )
            p = next_state.players
            won = p.has_won
            riichi = p.riichi
            melded = ~p.is_hand_concealed
            rewards = next_state.rewards.astype(jnp.float32)
            is_hora = won.any()
            is_ron = is_hora & ((rewards < 0.0).sum() == 1)
            last_p = next_state.round_state.last_player.astype(jnp.int32)
            dealt_in_mask = jnp.zeros(num_players, dtype=jnp.bool_).at[last_p].set(is_ron)

            cand_won = hand_ended & won[agent_seat]
            cand_dealt_in = hand_ended & dealt_in_mask[agent_seat]
            cand_riichi = hand_ended & riichi[agent_seat]
            cand_melded = hand_ended & melded[agent_seat]

            ev = {
                "hands": hand_ended.astype(jnp.int32),
                "won": cand_won.astype(jnp.int32),
                "dealt_in": cand_dealt_in.astype(jnp.int32),
                "riichi": cand_riichi.astype(jnp.int32),
                "melded": cand_melded.astype(jnp.int32),
                "won_pts": jnp.where(cand_won, rewards[agent_seat], 0.0),
                "dealt_in_pts": jnp.where(cand_dealt_in, -rewards[agent_seat], 0.0),
            }
            return (next_state, rng), ev

        (final_state, _), events = lax.scan(
            step_body,
            (init_state, wall_step_seed),
            jnp.arange(cfg.max_steps, dtype=jnp.int32),
        )
        totals = jax.tree.map(lambda x: x.sum(axis=0), events)

        scores = final_state.round_state.score.astype(jnp.float32)
        init_wind = final_state.round_state.init_wind.astype(jnp.int32)
        my_score = scores[agent_seat]
        my_wind = init_wind[agent_seat]
        ahead = (scores > my_score) | ((scores == my_score) & (init_wind < my_wind))
        rank_idx = ahead.sum().astype(jnp.int32)  # 0=1st .. 3=4th
        pt = rank_pts_arr[rank_idx]
        finished = (final_state.terminated | final_state.truncated).astype(jnp.int32)
        return totals, rank_idx, pt, finished

    def play_duplicate_wall(init_key: jax.Array, wall_step_seed: jax.Array):
        init_state = env.init(init_key)
        seats = jnp.arange(num_players, dtype=jnp.int32)
        return jax.vmap(lambda s: play_single_seat(init_state, s, wall_step_seed))(seats)

    run_all = jax.jit(jax.vmap(play_duplicate_wall))

    rng = jax.random.PRNGKey(cfg.seed)
    k_init, k_step = jax.random.split(rng)
    init_keys = jax.random.split(k_init, cfg.num_walls)
    step_seeds = jax.random.split(k_step, cfg.num_walls)

    # Chunk walls to bound GPU VRAM cleanly (e.g., 8 walls = 32 concurrent games per chunk)
    step_w = max(1, min(int(chunk_walls), int(cfg.num_walls)))
    totals_list, ranks_list, pts_list, fin_list = [], [], [], []
    for start in range(0, cfg.num_walls, step_w):
        end = min(start + step_w, cfg.num_walls)
        t_c, r_c, p_c, f_c = jax.device_get(run_all(init_keys[start:end], step_seeds[start:end]))
        totals_list.append(t_c)
        ranks_list.append(np.asarray(r_c, dtype=np.int32))
        pts_list.append(np.asarray(p_c, dtype=np.float64))
        fin_list.append(np.asarray(f_c, dtype=np.float64))

    totals = {k: np.concatenate([np.asarray(t[k]) for t in totals_list], axis=0) for k in totals_list[0]}
    rank_idxs_np = np.concatenate(ranks_list, axis=0)
    pts_np = np.concatenate(pts_list, axis=0)
    finished_np = np.concatenate(fin_list, axis=0)

    # Paired duplicate statistics: average across the 4 seat rotations per wall first!
    wall_mean_pts = pts_np.mean(axis=1)  # (num_walls,)
    pt_mean = float(wall_mean_pts.mean())
    pt_se = float(wall_mean_pts.std(ddof=1) / np.sqrt(max(cfg.num_walls, 1))) if cfg.num_walls > 1 else 0.0
    pt_ci95 = 1.96 * pt_se

    total_games = int(rank_idxs_np.size)
    total_hands = float(np.asarray(totals["hands"]).sum())
    denom_hands = max(total_hands, 1.0)
    total_won = float(np.asarray(totals["won"]).sum())
    total_dealt = float(np.asarray(totals["dealt_in"]).sum())

    return {
        "total_games": float(total_games),
        "total_hands": total_hands,
        "finished_game_ratio": float(finished_np.mean()),
        "tenhou_pt_mean": pt_mean,
        "tenhou_pt_ci95": pt_ci95,
        "avg_rank": float(rank_idxs_np.mean() + 1.0),
        "rank_1_rate": float((rank_idxs_np == 0).mean()),
        "rank_2_rate": float((rank_idxs_np == 1).mean()),
        "rank_3_rate": float((rank_idxs_np == 2).mean()),
        "rank_4_rate": float((rank_idxs_np == 3).mean()),
        "hora_rate": total_won / denom_hands,
        "deal_in_rate": total_dealt / denom_hands,
        "riichi_rate": float(np.asarray(totals["riichi"]).sum()) / denom_hands,
        "meld_rate": float(np.asarray(totals["melded"]).sum()) / denom_hands,
        "avg_win_pts": float(np.asarray(totals["won_pts"]).sum()) / max(total_won, 1.0),
        "avg_deal_in_pts": float(np.asarray(totals["dealt_in_pts"]).sum()) / max(total_dealt, 1.0),
    }

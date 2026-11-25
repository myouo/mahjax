"""Sovereign-Mahjong V6 Sharded Offline Dataset Builder & Exact 36x Symmetry Permutations.

Streams complete-episode replays (from `.jsonl` mjai files or JAX policy rollouts) into
compressed `.npz` shards containing:
- Zero-leakage V6 observations (`planes`, `scalars`, `river_tokens`, `river_tile_ids`,
  `river_valid_mask`, `last_draw_type`, `is_last_draw_red`, `target_type`, `target_is_red`)
- Privileged Oracle planes (`oracle_planes (N, 26, 34)`)
- Exact sparse-compacted Belief ground-truth labels (`y_ron`, `y_score_bin`, `y_call_threat`, `y_tenpai`)
- Exact Birkhoff `(4, 4)` terminal rank permutation matrices (`final_rank_matrix_4x4`) + `grp_features (29,)`
- Compact `TableSnapshot` fields so Phase 0-C BC runs strictly isomorphic `Delta_U_ron/agari/ryuukyoku`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjax.common.action import Action
from mahjax.ui.mjai import decode_steps
from mahjax.ui.rules import Rules
from mahjong_rl.algo.legal_ron import (
    compute_sparse_belief_labels,
    extract_ron_snapshot,
    extract_table_snapshot,
)
from mahjong_rl.algo.utility import W_PHOENIX_7D, make_homogeneous_table
from mahjong_rl.features.obs_2d import clean_auto_reset_state
from mahjong_rl.features.obs_v6 import encode_obs_v6
from mahjong_rl.features.oracle import encode_oracle_planes
from mahjong_rl.models.grp import encode_grp_features, scores_to_rank_matrix_4x4

# All 6 permutations of 3 items (used for both suits m/p/s and dragons White/Green/Red)
PERMS_3: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

# Full 36-element symmetry group S_3^suit x S_3^sangen
SYMMETRY_PERMS_36: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = tuple(
    (sp, dp) for sp in PERMS_3 for dp in PERMS_3
)


def build_tile_and_action_perm_36(
    suit_perm: Tuple[int, int, int],
    dragon_perm: Tuple[int, int, int] = (0, 1, 2),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build exact index maps `(tile_map[34], action_map[87])` for `(suit_perm, dragon_perm)`."""
    tile_map = np.arange(34, dtype=np.int32)
    # 1. Permute number suits m=[0:9], p=[9:18], s=[18:27]
    for old_s, new_s in enumerate(suit_perm):
        tile_map[old_s * 9 : (old_s + 1) * 9] = np.arange(new_s * 9, (new_s + 1) * 9, dtype=np.int32)
    # 2. Winds [27:31] remain fixed
    # 3. Permute dragons White=31, Green=32, Red=33
    for old_d, new_d in enumerate(dragon_perm):
        tile_map[31 + old_d] = 31 + new_d

    action_map = np.arange(Action.NUM_ACTION, dtype=np.int32)
    # Normal discards 0..33 (covers suits + winds + dragons)
    action_map[:34] = tile_map
    # Red fives 34, 35, 36
    for old_s, new_s in enumerate(suit_perm):
        action_map[34 + old_s] = 34 + new_s
    # Self-kans 37..70 (exact tile 0..33 shifted by 37)
    action_map[37:71] = 37 + tile_map
    # Actions 71..86 (TSUMOGIRI, RIICHI, TSUMO, RON, PON..CHI_R_RED, PASS, KYUUSHU, DUMMY) are invariant!
    return tile_map, action_map


_TILE_MAPS_36: List[np.ndarray] = []
_INV_TILE_MAPS_36: List[np.ndarray] = []
_ACTION_MAPS_36: List[np.ndarray] = []
_INV_ACTION_MAPS_36: List[np.ndarray] = []
for _sp, _dp in SYMMETRY_PERMS_36:
    _tm, _am = build_tile_and_action_perm_36(_sp, _dp)
    _TILE_MAPS_36.append(_tm)
    _INV_TILE_MAPS_36.append(np.argsort(_tm))
    _ACTION_MAPS_36.append(_am)
    _INV_ACTION_MAPS_36.append(np.argsort(_am))


def permute_v6_batch(batch: Dict[str, np.ndarray], perm_idx: int) -> Dict[str, np.ndarray]:
    """Apply element `perm_idx in [0..35]` of `S_3^suit x S_3^sangen` to a V6 batch dictionary."""
    if perm_idx == 0:
        return batch
    tile_map = _TILE_MAPS_36[perm_idx]
    inv_tile = _INV_TILE_MAPS_36[perm_idx]
    act_map = _ACTION_MAPS_36[perm_idx]
    inv_act = _INV_ACTION_MAPS_36[perm_idx]

    out = dict(batch)
    for k in ("planes", "oracle_planes", "y_ron", "y_score_bin", "y_call_threat"):
        if k in out and out[k] is not None:
            out[k] = out[k][..., inv_tile]
    if "river_tile_ids" in out:
        rids = out["river_tile_ids"]
        out["river_tile_ids"] = np.where(rids >= 0, tile_map[np.clip(rids, 0, 33)], -1).astype(rids.dtype)
    for k in ("last_draw_type", "target_type"):
        if k in out:
            v = out[k]
            out[k] = np.where(v >= 0, tile_map[np.clip(v, 0, 33)], -1).astype(v.dtype)
    if "action" in out:
        out["action"] = act_map[out["action"]].astype(out["action"].dtype)
    if "action_mask" in out:
        out["action_mask"] = out["action_mask"][..., inv_act]
    return out


# Backward-compatible alias for 6 suit-only permutations
def permute_suits_sample(
    planes: np.ndarray,
    oracle_planes: Optional[np.ndarray],
    action: np.ndarray,
    action_mask: np.ndarray,
    perm_idx: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    b = {"planes": planes, "oracle_planes": oracle_planes, "action": action, "action_mask": action_mask}
    out = permute_v6_batch(b, perm_idx * 6)
    return out["planes"], out["oracle_planes"], out["action"], out["action_mask"]


def _empty_v6_buf() -> Dict[str, List[np.ndarray]]:
    return {
        "planes": [],
        "scalars": [],
        "river_tokens": [],
        "river_tile_ids": [],
        "river_valid_mask": [],
        "last_draw_type": [],
        "is_last_draw_red": [],
        "target_type": [],
        "target_is_red": [],
        "oracle_planes": [],
        "action": [],
        "action_mask": [],
        "current_player": [],
        "grp_features": [],
        "final_rank_matrix_4x4": [],
        "w_table_abs": [],
        "table_scores": [],
        "table_round_num": [],
        "table_round_limit": [],
        "table_honba": [],
        "table_kyotaku": [],
        "table_dealer": [],
        "table_init_wind": [],
        "table_wall_rem": [],
        "y_ron": [],
        "y_score_bin": [],
        "y_call_threat": [],
        "y_tenpai": [],
    }


def _save_shard(out_dir: Path, shard_idx: int, buf: Dict[str, List[np.ndarray]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_idx:05d}.npz"
    arrays = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
    np.savez_compressed(path, **arrays)
    return path


def _label_ron_snapshots_batched(snaps_pytree, k_max: int = 1024):
    """Compute exact sparse belief labels over a batch of `RonSnapshotV6` in chunks."""
    n = snaps_pytree.current_player.shape[0]
    chunk = 512
    ys_ron, ys_sb, ys_ct, ys_ten = [], [], [], []
    label_fn = jax.jit(lambda s: compute_sparse_belief_labels(s, k_max=k_max))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = jax.tree_util.tree_map(lambda x: jnp.asarray(x[start:end]), snaps_pytree)
        yr, ysb, yct, yt, _ = jax.device_get(label_fn(sub))
        ys_ron.append(np.asarray(yr, dtype=np.uint8))
        ys_sb.append(np.asarray(ysb, dtype=np.uint8))
        ys_ct.append(np.asarray(yct, dtype=np.uint8))
        ys_ten.append(np.asarray(yt, dtype=np.uint8))
    return (
        np.concatenate(ys_ron, axis=0),
        np.concatenate(ys_sb, axis=0),
        np.concatenate(ys_ct, axis=0),
        np.concatenate(ys_ten, axis=0),
    )


def build_shards_from_mjai_files(
    mjai_paths: Iterable[Path | str],
    out_dir: Path | str,
    env_name: Literal["red_4p", "no_red_4p"] = "red_4p",
    w_table_default: Optional[np.ndarray] = None,
    shard_size: int = 50_000,
    ron_k_max: int = 1024,
) -> List[Path]:
    """Replay MahJax-compatible `.jsonl` mjai logs and write V6 sharded `.npz` files."""
    out_path = Path(out_dir)
    rules = Rules(env_name)
    w_def = (
        np.asarray(make_homogeneous_table(W_PHOENIX_7D), dtype=np.float32)
        if w_table_default is None
        else np.asarray(w_table_default, dtype=np.float32)
    )

    encode_step = jax.jit(
        lambda s: (
            encode_obs_v6(s),
            encode_oracle_planes(s),
            encode_grp_features(s),
            extract_table_snapshot(s),
            extract_ron_snapshot(s),
        )
    )

    buf = _empty_v6_buf()
    buffered = 0
    shard_idx = 0
    written: List[Path] = []

    for fp in mjai_paths:
        events = [json.loads(line) for line in Path(fp).read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = []
        snaps = []
        last_state = None
        for state, action in decode_steps(rules, events):
            last_state = state
            if int(action) == int(Action.DUMMY):
                continue
            pub, ora, grp, tbl, rsnap = jax.device_get(encode_step(state))
            steps.append((pub, ora, grp, tbl, int(action), np.asarray(state.legal_action_mask, dtype=np.bool_), int(state.current_player)))
            snaps.append(rsnap)
        if not steps or last_state is None:
            continue

        rank_mat = np.asarray(
            scores_to_rank_matrix_4x4(last_state.round_state.score, last_state.round_state.init_wind),
            dtype=np.float32,
        )  # (4, 4)
        stacked_snaps = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *snaps)
        y_ron, y_sb, y_ct, y_ten = _label_ron_snapshots_batched(stacked_snaps, k_max=ron_k_max)

        n_g = len(steps)
        buf["planes"].append(np.stack([s[0]["planes"] for s in steps], axis=0).astype(np.float32))
        buf["scalars"].append(np.stack([s[0]["scalars"] for s in steps], axis=0).astype(np.float32))
        buf["river_tokens"].append(np.stack([s[0]["river_tokens"] for s in steps], axis=0).astype(np.float32))
        buf["river_tile_ids"].append(np.stack([s[0]["river_tile_ids"] for s in steps], axis=0).astype(np.int8))
        buf["river_valid_mask"].append(np.stack([s[0]["river_valid_mask"] for s in steps], axis=0).astype(np.bool_))
        buf["last_draw_type"].append(np.array([int(s[0]["last_draw_type"]) for s in steps], dtype=np.int8))
        buf["is_last_draw_red"].append(np.array([bool(s[0]["is_last_draw_red"]) for s in steps], dtype=np.bool_))
        buf["target_type"].append(np.array([int(s[0]["target_type"]) for s in steps], dtype=np.int8))
        buf["target_is_red"].append(np.array([bool(s[0]["target_is_red"]) for s in steps], dtype=np.bool_))
        buf["oracle_planes"].append(np.stack([s[1] for s in steps], axis=0).astype(np.float32))
        buf["grp_features"].append(np.stack([s[2] for s in steps], axis=0).astype(np.float32))
        buf["action"].append(np.array([s[4] for s in steps], dtype=np.int32))
        buf["action_mask"].append(np.stack([s[5] for s in steps], axis=0).astype(np.bool_))
        buf["current_player"].append(np.array([s[6] for s in steps], dtype=np.int8))
        buf["final_rank_matrix_4x4"].append(np.broadcast_to(rank_mat[None, :, :], (n_g, 4, 4)).copy())
        buf["w_table_abs"].append(np.broadcast_to(w_def[None, :, :], (n_g, 4, 4)).copy())
        buf["table_scores"].append(np.stack([s[3].scores for s in steps], axis=0).astype(np.int32))
        buf["table_round_num"].append(np.array([int(s[3].round_num) for s in steps], dtype=np.int8))
        buf["table_round_limit"].append(np.array([int(s[3].round_limit) for s in steps], dtype=np.int8))
        buf["table_honba"].append(np.array([int(s[3].honba) for s in steps], dtype=np.int32))
        buf["table_kyotaku"].append(np.array([int(s[3].kyotaku) for s in steps], dtype=np.int8))
        buf["table_dealer"].append(np.array([int(s[3].dealer) for s in steps], dtype=np.int8))
        buf["table_init_wind"].append(np.stack([s[3].init_wind for s in steps], axis=0).astype(np.int8))
        buf["table_wall_rem"].append(np.array([int(s[3].wall_rem) for s in steps], dtype=np.int32))
        buf["y_ron"].append(y_ron)
        buf["y_score_bin"].append(y_sb)
        buf["y_call_threat"].append(y_ct)
        buf["y_tenpai"].append(y_ten)
        buffered += n_g

        if buffered >= shard_size:
            written.append(_save_shard(out_path, shard_idx, buf))
            shard_idx += 1
            for k in buf:
                buf[k].clear()
            buffered = 0

    if buffered > 0:
        written.append(_save_shard(out_path, shard_idx, buf))
    return written


def build_shards_from_policy_rollout(
    out_dir: Path | str,
    num_games: int = 16,
    env_name: Literal["red_4p", "no_red_4p"] = "red_4p",
    round_mode: Literal["single", "east", "half"] = "half",
    max_steps_per_game: int = 1200,
    actor_fn: Optional[Callable] = None,
    w_table_default: Optional[np.ndarray] = None,
    shard_size: int = 50_000,
    ron_k_max: int = 1024,
    seed: int = 0,
) -> List[Path]:
    """Generate complete-episode V6 sharded datasets from a vectorized JAX policy."""
    out_path = Path(out_dir)
    env = mahjax.make(env_name, round_mode=round_mode, observe_type="dict", round_end="auto")
    policy = mahjax.rule_based_player if actor_fn is None else actor_fn
    w_def = (
        np.asarray(make_homogeneous_table(W_PHOENIX_7D), dtype=np.float32)
        if w_table_default is None
        else np.asarray(w_table_default, dtype=np.float32)
    )

    def play_one(init_key: jax.Array, play_key: jax.Array):
        init_state = env.init(init_key)

        def step_body(carry, _):
            state, rng = carry
            rng, k_act, k_step = jax.random.split(rng, 3)
            done_before = state.terminated | state.truncated
            clean_st = clean_auto_reset_state(state)
            pub = encode_obs_v6(clean_st)
            ora = encode_oracle_planes(clean_st)
            grp = encode_grp_features(clean_st)
            tbl = extract_table_snapshot(clean_st)
            rsnap = extract_ron_snapshot(clean_st)
            mask = clean_st.legal_action_mask.astype(jnp.bool_)
            cp = clean_st.current_player.astype(jnp.int8)
            act = policy(clean_st, k_act).astype(jnp.int32)

            next_state = jax.lax.cond(
                done_before,
                lambda: state,
                lambda: env.step(state, act, k_step),
            )
            return (next_state, rng), (~done_before, pub, ora, grp, tbl, rsnap, act, mask, cp)

        (final_state, _), seq = jax.lax.scan(step_body, (init_state, play_key), None, length=max_steps_per_game)
        rank_mat = scores_to_rank_matrix_4x4(final_state.round_state.score, final_state.round_state.init_wind)
        return seq, rank_mat

    play_batch = jax.jit(jax.vmap(play_one))

    rng = jax.random.PRNGKey(seed)
    chunk_games = min(num_games, 8)
    buf = _empty_v6_buf()
    buffered = 0
    shard_idx = 0
    written: List[Path] = []

    remaining = num_games
    while remaining > 0:
        b = min(remaining, chunk_games)
        rng, k_i, k_p = jax.random.split(rng, 3)
        seq, rank_mats = play_batch(jax.random.split(k_i, b), jax.random.split(k_p, b))
        valid, pub, ora, grp, tbl, rsnap, act, mask, cp = jax.device_get(seq)
        rank_mats_np = np.asarray(jax.device_get(rank_mats), dtype=np.float32)  # (B, 4, 4)

        for g in range(b):
            v_idx = np.nonzero(valid[g])[0]
            if v_idx.size == 0:
                continue
            g_snaps = jax.tree_util.tree_map(lambda x: x[g, v_idx], rsnap)
            y_ron, y_sb, y_ct, y_ten = _label_ron_snapshots_batched(g_snaps, k_max=ron_k_max)

            n_g = int(v_idx.size)
            buf["planes"].append(pub["planes"][g, v_idx].astype(np.float32))
            buf["scalars"].append(pub["scalars"][g, v_idx].astype(np.float32))
            buf["river_tokens"].append(pub["river_tokens"][g, v_idx].astype(np.float32))
            buf["river_tile_ids"].append(pub["river_tile_ids"][g, v_idx].astype(np.int8))
            buf["river_valid_mask"].append(pub["river_valid_mask"][g, v_idx].astype(np.bool_))
            buf["last_draw_type"].append(pub["last_draw_type"][g, v_idx].astype(np.int8))
            buf["is_last_draw_red"].append(pub["is_last_draw_red"][g, v_idx].astype(np.bool_))
            buf["target_type"].append(pub["target_type"][g, v_idx].astype(np.int8))
            buf["target_is_red"].append(pub["target_is_red"][g, v_idx].astype(np.bool_))
            buf["oracle_planes"].append(ora[g, v_idx].astype(np.float32))
            buf["grp_features"].append(grp[g, v_idx].astype(np.float32))
            buf["action"].append(act[g, v_idx].astype(np.int32))
            buf["action_mask"].append(mask[g, v_idx].astype(np.bool_))
            buf["current_player"].append(cp[g, v_idx].astype(np.int8))
            buf["final_rank_matrix_4x4"].append(np.broadcast_to(rank_mats_np[g][None, :, :], (n_g, 4, 4)).copy())
            buf["w_table_abs"].append(np.broadcast_to(w_def[None, :, :], (n_g, 4, 4)).copy())
            buf["table_scores"].append(tbl.scores[g, v_idx].astype(np.int32))
            buf["table_round_num"].append(tbl.round_num[g, v_idx].astype(np.int8))
            buf["table_round_limit"].append(tbl.round_limit[g, v_idx].astype(np.int8))
            buf["table_honba"].append(tbl.honba[g, v_idx].astype(np.int32))
            buf["table_kyotaku"].append(tbl.kyotaku[g, v_idx].astype(np.int8))
            buf["table_dealer"].append(tbl.dealer[g, v_idx].astype(np.int8))
            buf["table_init_wind"].append(tbl.init_wind[g, v_idx].astype(np.int8))
            buf["table_wall_rem"].append(tbl.wall_rem[g, v_idx].astype(np.int32))
            buf["y_ron"].append(y_ron)
            buf["y_score_bin"].append(y_sb)
            buf["y_call_threat"].append(y_ct)
            buf["y_tenpai"].append(y_ten)
            buffered += n_g

        remaining -= b
        if buffered >= shard_size:
            written.append(_save_shard(out_path, shard_idx, buf))
            shard_idx += 1
            for k in buf:
                buf[k].clear()
            buffered = 0

    if buffered > 0:
        written.append(_save_shard(out_path, shard_idx, buf))
    return written

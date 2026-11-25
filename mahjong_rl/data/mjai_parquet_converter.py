"""High-Throughput Zero-Loss External MJAI Parquet -> Sovereign-Mahjong V6 Shard Converter.

Architecture: Two-Stage Hybrid Pipeline ("CPU Exact Deck Reconstruction + GPU Batched Replay")
1. CPU Stage (pure Python, ~0.4ms/round):
   - Computes exact end-of-hanchan final scores & Birkhoff `(4, 4)` placement permutation matrix.
   - Reconstructs the EXACT 136-tile MahJax `deck` for every kyoku (`tehais` in `deck[84:136]`,
     live draws in `deck[83..14]`, rinshan draws in `deck[10..13]`, dora indicators in
     `deck[9,7,5,3,1]`, ura indicators in `deck[8,6,4,2,0]`, exact complement multiset fill).
   - Maps every MJAI action event directly to its exact MahJax `Action` id (`0..86`), sorting
     simultaneous double-ron `hora` events in clockwise head-bump order.
2. GPU Stage (`jax.jit(jax.vmap(lax.scan))` over 64-128 kyokus simultaneously):
   - Injects reconstructed `deck` via MahJax's own `_init_for_next_round_from_prepared`.
   - Advances exact `mahjax` `env.step`, automatically recovering EVERY `PASS` decision point
     and handling all 6 draw types (exhaustive + 5 abortive draws) with zero Python round-trips.
   - Drops any rare desynced kyoku (`desync_flag`) so emitted training shards are 100% exact.
   - Extracts `obs_v6`, `oracle_planes`, `grp_features`, `TableSnapshot`, and runs static-capacity
     `compute_sparse_belief_labels` on GPU.
   - Stores spatial planes & `river_tokens` in `float16` (bit-exact for `{0, 1, k/4, k/8}`!),
     reducing disk footprint by >2x with zero information loss.

CLI usage:
    python -m mahjong_rl.data.mjai_parquet_converter \
        --parquet-dir data/tenhou_2026_mjai \
        --out-dir data/v6_shards_2026 \
        --max-games 10000
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

import mahjax
from mahjax.common.action import Action
from mahjax.common.constants import RED_FIVE_TILE_TYPES, SENTINEL_TILE_ID, TILE_RANGE
from mahjax.common.hand import Hand
from mahjax.common.logic import (
    _calc_wind,
    _deal_hands,
    _dora_indicator_ix,
    _init_for_next_round_from_prepared,
    _make_state,
    _ura_dora_indicator_ix,
    v_can_win,
)
from mahjax.mahjong_4p.constants import (
    FIRST_DORA_INDICATOR_IX,
    FIRST_DRAW_IDX,
    MAX_DORA_INDICATORS,
    NUM_PHYSICAL_TILES,
    NUM_PLAYERS,
    RINSHAN_IX,
)
from mahjax.mahjong_4p.env import Mahjong4p
from mahjax.mahjong_4p.state import State
from mahjax.ui.mjai import mjai_to_tile
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

_WIND_TO_IDX = {"E": 0, "S": 1, "W": 2, "N": 3}
_RED_BASE_SET = frozenset(RED_FIVE_TILE_TYPES)  # {4, 13, 22}
_KAN_EVENT_TYPES = frozenset({"ankan", "kakan", "daiminkan"})
_ACTION_EVENT_TYPES = frozenset(
    {"dahai", "reach", "pon", "chi", "daiminkan", "ankan", "kakan", "hora", "ryukyoku"}
)

# Canonical 136-tile counts in red_4p (0..33 normal, 34=5mr, 35=5pr, 36=5sr)
_FULL_TILE_COUNTS = np.full((37,), 4, dtype=np.int32)
_FULL_TILE_COUNTS[4] = 3
_FULL_TILE_COUNTS[13] = 3
_FULL_TILE_COUNTS[22] = 3
_FULL_TILE_COUNTS[34:37] = 1


def _tile_type(tile_id: int) -> int:
    if tile_id >= 34:
        return RED_FIVE_TILE_TYPES[tile_id - 34]
    return tile_id


def _map_mjai_event_to_action(ev: dict) -> Optional[Tuple[int, int]]:
    """Map one MJAI action event to `(actor, mahjax_action_id)` (`actor = -1` for table-wide `ryukyoku`)."""
    kind = ev.get("type")
    if kind == "ryukyoku":
        return -1, int(Action.KYUUSHU)
    if kind not in _ACTION_EVENT_TYPES:
        return None

    actor = int(ev["actor"])
    if kind == "reach":
        return actor, int(Action.RIICHI)
    if kind == "dahai":
        if bool(ev.get("tsumogiri", False)):
            return actor, int(Action.TSUMOGIRI)
        return actor, int(mjai_to_tile(ev["pai"]))
    if kind == "hora":
        target = int(ev["target"])
        return actor, int(Action.TSUMO if actor == target else Action.RON)
    if kind == "daiminkan":
        return actor, int(Action.OPEN_KAN)
    if kind == "ankan":
        t = _tile_type(mjai_to_tile(ev["consumed"][0]))
        return actor, 37 + t
    if kind == "kakan":
        t = _tile_type(mjai_to_tile(ev["pai"]))
        return actor, 37 + t
    if kind == "pon":
        consumed = [mjai_to_tile(x) for x in ev["consumed"]]
        uses_red = any(x >= 34 for x in consumed)
        return actor, int(Action.PON_RED if uses_red else Action.PON)
    if kind == "chi":
        called = mjai_to_tile(ev["pai"])
        consumed = [mjai_to_tile(x) for x in ev["consumed"]]
        uses_red = any(x >= 34 for x in consumed)
        c_type = _tile_type(called)
        all_types = sorted([c_type, _tile_type(consumed[0]), _tile_type(consumed[1])])
        offset = all_types.index(c_type)
        if offset == 0:
            return actor, int(Action.CHI_L_RED if uses_red else Action.CHI_L)
        if offset == 1:
            return actor, int(Action.CHI_M_RED if uses_red else Action.CHI_M)
        return actor, int(Action.CHI_R_RED if uses_red else Action.CHI_R)
    return None


@dataclass
class ParsedRound:
    deck: np.ndarray                 # (136,) int8
    round_num: int                   # 0..11
    honba: int
    kyotaku: int
    dealer: int                      # 0..3
    scores: np.ndarray               # (4,) int32 (100-pt units)
    ev_actors: np.ndarray            # (L,) int8
    ev_actions: np.ndarray           # (L,) int32
    n_events: int
    final_rank_matrix_4x4: np.ndarray  # (4, 4) float32


def _compute_final_scores_from_kyokus(kyoku_slices: List[List[dict]]) -> np.ndarray:
    """Compute exact end-of-hanchan scores (in 100-pt units) from the last kyoku."""
    last = kyoku_slices[-1]
    sk = last[0]
    scores = np.array(sk["scores"], dtype=np.int64)
    kyotaku = int(sk.get("kyotaku", 0))
    won_any = False

    for ev in last[1:]:
        t = ev.get("type")
        if t == "reach_accepted":
            scores[int(ev["actor"])] -= 1000
            kyotaku += 1
        elif t == "hora":
            deltas = ev.get("deltas")
            if deltas is not None:
                scores += np.array(deltas, dtype=np.int64)
            kyotaku = 0
            won_any = True
        elif t == "ryukyoku":
            deltas = ev.get("deltas")
            if deltas is not None:
                scores += np.array(deltas, dtype=np.int64)

    # Tenhou rule: unclaimed riichi sticks at end of hanchan go to 1st place (tie-break by init_wind [0,1,2,3])
    if (not won_any) and kyotaku > 0:
        order = np.lexsort((np.arange(4), -scores))
        scores[int(order[0])] += kyotaku * 1000

    return (scores // 100).astype(np.int32)


def _reconstruct_deck_and_actions(
    kyoku_evs: List[dict],
    final_rank_mat: np.ndarray,
    max_events: int = 160,
) -> Optional[ParsedRound]:
    """Reconstruct exact 136-tile MahJax `deck` and ordered `(actor, action)` sequence for one kyoku."""
    sk = kyoku_evs[0]
    bakaze_idx = _WIND_TO_IDX.get(str(sk.get("bakaze", "E")), 0)
    kyoku_num = int(sk.get("kyoku", 1))
    round_num = bakaze_idx * 4 + (kyoku_num - 1)
    honba = int(sk.get("honba", 0))
    kyotaku = int(sk.get("kyotaku", 0))
    dealer = int(sk.get("oya", 0))
    scores_100 = (np.array(sk["scores"], dtype=np.int64) // 100).astype(np.int32)

    tehais = sk.get("tehais")
    if not tehais or len(tehais) != 4 or any(len(h) != 13 for h in tehais):
        return None

    # Track remaining tile multiset to guarantee a 100% physically valid 136-tile wall
    rem = _FULL_TILE_COUNTS.copy()
    deck = np.full((NUM_PHYSICAL_TILES,), -1, dtype=np.int8)

    def _place(idx: int, tile_id: int) -> bool:
        if idx < 0 or idx >= NUM_PHYSICAL_TILES or deck[idx] != -1:
            return False
        if rem[tile_id] <= 0:
            return False
        deck[idx] = tile_id
        rem[tile_id] -= 1
        return True

    # 1. Place 4 starting hands at deck[84:136] (matches `_deal_hands`)
    hand_start = NUM_PHYSICAL_TILES - 52  # 84
    for seat in range(4):
        for j, tile_str in enumerate(tehais[seat]):
            tid = mjai_to_tile(tile_str)
            if not _place(hand_start + seat * 13 + j, tid):
                return None

    # 2. Place initial dora indicator at deck[9]
    dora_count = 0
    if not _place(FIRST_DORA_INDICATOR_IX, mjai_to_tile(sk["dora_marker"])):
        return None
    dora_count = 1

    # 3. Single pass over kyoku events: collect draws, rinshan, dora, ura, and action stream
    live_ix = FIRST_DRAW_IDX  # 83, decrements down to 14
    rinshan_n = 0
    ura_markers: List[int] = []
    raw_actions: List[Tuple[int, int, Optional[int]]] = []  # (actor, action, hora_target_or_None)
    expect_rinshan = False

    for ev in kyoku_evs[1:]:
        t = ev.get("type")
        if t == "dora":
            if dora_count < MAX_DORA_INDICATORS:
                tid = mjai_to_tile(ev["dora_marker"])
                if not _place(FIRST_DORA_INDICATOR_IX - 2 * dora_count, tid):
                    return None
                dora_count += 1
        elif t == "tsumo":
            tid = mjai_to_tile(ev["pai"])
            if expect_rinshan:
                if rinshan_n >= 4 or not _place(RINSHAN_IX + rinshan_n, tid):
                    return None
                rinshan_n += 1
                expect_rinshan = False
            else:
                if live_ix < 14 or not _place(live_ix, tid):
                    return None
                live_ix -= 1
        elif t == "hora":
            expect_rinshan = False
            if not ura_markers and ev.get("ura_markers"):
                ura_markers = [mjai_to_tile(x) for x in ev["ura_markers"]]
            mapped = _map_mjai_event_to_action(ev)
            if mapped is not None:
                raw_actions.append((mapped[0], mapped[1], int(ev["target"])))
        elif t in _ACTION_EVENT_TYPES:
            expect_rinshan = t in _KAN_EVENT_TYPES
            mapped = _map_mjai_event_to_action(ev)
            if mapped is not None:
                raw_actions.append((mapped[0], mapped[1], None))
        elif t in ("end_kyoku", "end_game"):
            break

    # 4. Place revealed ura-dora indicators at deck[8, 6, 4, 2, 0]
    for k, utid in enumerate(ura_markers[:MAX_DORA_INDICATORS]):
        # Some logs may omit/repeat; place only if available in remaining pool
        if rem[utid] > 0:
            _place(FIRST_DORA_INDICATOR_IX - 2 * k - 1, utid)

    # 5. Fill all remaining unrevealed wall slots from the exact complement multiset
    leftover = np.repeat(np.arange(37, dtype=np.int8), rem)
    empty_slots = np.flatnonzero(deck == -1)
    if empty_slots.size != leftover.size:
        return None
    deck[empty_slots] = leftover

    # 6. Sort any consecutive multi-ron `hora` events in clockwise head-bump order `(actor - target) % 4`
    i = 0
    ordered_actors: List[int] = []
    ordered_actions: List[int] = []
    while i < len(raw_actions):
        if raw_actions[i][1] == int(Action.RON):
            j = i + 1
            while j < len(raw_actions) and raw_actions[j][1] == int(Action.RON):
                j += 1
            group = raw_actions[i:j]
            if len(group) > 1:
                group.sort(key=lambda item: (item[0] - (item[2] or 0)) % 4)
            for act_seat, act_id, _ in group:
                ordered_actors.append(act_seat)
                ordered_actions.append(act_id)
            i = j
        else:
            ordered_actors.append(raw_actions[i][0])
            ordered_actions.append(raw_actions[i][1])
            i += 1

    n_ev = len(ordered_actions)
    if n_ev == 0 or n_ev > max_events:
        return None

    ev_actors = np.full((max_events,), -1, dtype=np.int8)
    ev_actions = np.full((max_events,), -1, dtype=np.int32)
    ev_actors[:n_ev] = np.asarray(ordered_actors, dtype=np.int8)
    ev_actions[:n_ev] = np.asarray(ordered_actions, dtype=np.int32)

    return ParsedRound(
        deck=deck,
        round_num=round_num,
        honba=honba,
        kyotaku=kyotaku,
        dealer=dealer,
        scores=scores_100,
        ev_actors=ev_actors,
        ev_actions=ev_actions,
        n_events=n_ev,
        final_rank_matrix_4x4=final_rank_mat,
    )


def parse_mjai_game_to_rounds(
    events_str: str,
    max_events_per_round: int = 160,
) -> List[ParsedRound]:
    """Parse one MJAI JSONL game string into exact `ParsedRound` entries."""
    events = [json.loads(line) for line in events_str.splitlines() if line.strip()]
    kyoku_slices: List[List[dict]] = []
    cur: List[dict] = []
    for ev in events:
        t = ev.get("type")
        if t == "start_kyoku":
            if cur:
                kyoku_slices.append(cur)
            cur = [ev]
        elif cur:
            cur.append(ev)
            if t == "end_kyoku":
                kyoku_slices.append(cur)
                cur = []
    if cur:
        kyoku_slices.append(cur)
    if not kyoku_slices:
        return []

    final_scores = _compute_final_scores_from_kyokus(kyoku_slices)
    init_wind = np.arange(4, dtype=np.int32)
    # Exact (4, 4) Birkhoff permutation matrix in NumPy (zero JAX call on CPU!)
    ahead = (final_scores[None, :] > final_scores[:, None]) | (
        (final_scores[None, :] == final_scores[:, None]) & (init_wind[None, :] < init_wind[:, None])
    )
    ranks = ahead.sum(axis=1)
    rank_mat = np.eye(4, dtype=np.float32)[ranks]

    parsed: List[ParsedRound] = []
    for ks in kyoku_slices:
        pr = _reconstruct_deck_and_actions(ks, rank_mat, max_events=max_events_per_round)
        if pr is not None:
            parsed.append(pr)
    return parsed


def _init_round_state_from_deck(
    deck: jnp.ndarray,
    round_num: jnp.ndarray,
    honba: jnp.ndarray,
    kyotaku: jnp.ndarray,
    dealer: jnp.ndarray,
    scores: jnp.ndarray,
    round_limit: int = 7,
) -> State:
    """Initialize a bit-exact MahJax `State` directly from a reconstructed 136-tile `deck`."""
    base = _make_state(
        State,
        current_player=dealer.astype(jnp.int8),
        dealer=dealer.astype(jnp.int8),
        seat_wind=_calc_wind(dealer, NUM_PLAYERS),
        round=round_num.astype(jnp.int8),
        round_limit=jnp.int8(round_limit),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        order_points=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int32),
        honba=honba.astype(jnp.int8),
        kyotaku=kyotaku.astype(jnp.int8),
        score=scores.astype(jnp.int32),
    )
    init_hand_with_red = _deal_hands(deck, NUM_PLAYERS)
    init_hand = jax.vmap(Hand.to_34)(init_hand_with_red)
    face_down = jnp.full(MAX_DORA_INDICATORS, SENTINEL_TILE_ID, dtype=jnp.int8)
    dora_ind = face_down.at[0].set(deck[_dora_indicator_ix(base, 0)])
    ura_ind = face_down.at[0].set(deck[_ura_dora_indicator_ix(base, 0)])
    can_ron = v_can_win(init_hand, TILE_RANGE)
    prepared = (deck, dora_ind, ura_ind, init_hand_with_red, can_ron)
    return _init_for_next_round_from_prepared(base, prepared, Mahjong4p.RED)


def build_gpu_batch_replayer(
    max_steps_per_round: int = 192,
    ron_k_max: int = 2048,
    round_limit: int = 7,
):
    """Build a JIT-compiled GPU batched replayer + V6 feature & sparse-belief extractor."""
    # Use round_end="confirm" so MahJax never overwrites round-end state with a random next deal!
    env = mahjax.make("red_4p", round_mode="half", observe_type="dict", round_end="confirm")
    dummy_key = jax.random.PRNGKey(0)

    def replay_one_round(
        deck: jnp.ndarray,
        round_num: jnp.ndarray,
        honba: jnp.ndarray,
        kyotaku: jnp.ndarray,
        dealer: jnp.ndarray,
        scores: jnp.ndarray,
        ev_actors: jnp.ndarray,
        ev_actions: jnp.ndarray,
        n_events: jnp.ndarray,
    ):
        st0 = _init_round_state_from_deck(
            deck, round_num, honba, kyotaku, dealer, scores, round_limit=round_limit
        )
        max_ev = ev_actions.shape[0]

        def step_body(carry, _):
            st, cursor, done_round, desynced = carry
            cp = st.current_player.astype(jnp.int32)
            mask = st.legal_action_mask.astype(jnp.bool_)
            round_over = st.round_state.terminated_round | st.terminated | st.truncated
            active = (~done_round) & (~desynced) & (~round_over)

            has_ev = cursor < n_events
            safe_c = jnp.clip(cursor, 0, max_ev - 1)
            tgt_actor = ev_actors[safe_c].astype(jnp.int32)
            tgt_action = ev_actions[safe_c].astype(jnp.int32)
            safe_act = jnp.clip(tgt_action, 0, Action.NUM_ACTION - 1)

            # 1. Standard direct match: actor matches & recorded action is legal
            direct_match = has_ev & (tgt_actor == cp) & mask[safe_act]

            # 2. Edge case matching mahjax/ui/mjai.py:559: dahai with tsumogiri=False on unique drawn tile
            is_discard_act = (tgt_action >= 0) & (tgt_action < 37)
            alt_tsumo = (
                has_ev
                & (tgt_actor == cp)
                & is_discard_act
                & (~mask[safe_act])
                & (st.round_state.last_draw.astype(jnp.int32) == tgt_action)
                & mask[Action.TSUMOGIRI]
            )

            # 3. Abortive draw match: ryukyoku event (`tgt_actor == -1`, `tgt_action == KYUUSHU`)
            kyuushu_match = (
                has_ev
                & (tgt_actor < 0)
                & (tgt_action == Action.KYUUSHU)
                & mask[Action.KYUUSHU]
            )

            # 4. Implicit PASS recovery (MJAI omits PASS events!)
            can_pass = mask[Action.PASS]

            matched_ev = direct_match | alt_tsumo | kyuushu_match
            chosen_action = jnp.where(
                direct_match,
                safe_act,
                jnp.where(
                    alt_tsumo,
                    jnp.int32(Action.TSUMOGIRI),
                    jnp.where(
                        kyuushu_match,
                        jnp.int32(Action.KYUUSHU),
                        jnp.where(can_pass, jnp.int32(Action.PASS), jnp.int32(Action.DUMMY)),
                    ),
                ),
            )

            step_ok = active & (matched_ev | can_pass)
            step_desync = active & (~matched_ev) & (~can_pass)
            new_cursor = jnp.where(active & matched_ev, cursor + 1, cursor)

            # Encode V6 features BEFORE stepping (only kept where `step_ok`!)
            clean_st = clean_auto_reset_state(st)
            obs = encode_obs_v6(clean_st)
            ora = encode_oracle_planes(clean_st)
            grp_f = encode_grp_features(clean_st)
            tbl = extract_table_snapshot(clean_st)
            rsnap = extract_ron_snapshot(clean_st)

            next_st = jax.lax.cond(
                step_ok,
                lambda: env.step(st, chosen_action, dummy_key),
                lambda: st,
            )
            next_done = (
                done_round
                | round_over
                | (
                    step_ok
                    & (
                        next_st.round_state.terminated_round
                        | next_st.terminated
                        | (chosen_action == Action.KYUUSHU)
                    )
                )
            )
            next_desync = desynced | step_desync

            out = (step_ok, obs, ora, grp_f, tbl, rsnap, chosen_action, mask, cp.astype(jnp.int8))
            return (next_st, new_cursor, next_done, next_desync), out

        init_carry = (st0, jnp.int32(0), jnp.bool_(False), jnp.bool_(False))
        (_, final_cursor, final_done, final_desync), seq = jax.lax.scan(
            step_body, init_carry, None, length=max_steps_per_round
        )
        # Allow 1 unconsumed trailing exhaustive `ryukyoku` event (MahJax auto-terminates exhaustive draws)
        safe_fc = jnp.clip(final_cursor, 0, max_ev - 1)
        trailing_ryu = (final_cursor == n_events - 1) & (ev_actions[safe_fc] == Action.KYUUSHU)
        consumed_ok = (final_cursor >= n_events) | trailing_ryu
        round_clean = (~final_desync) & final_done & consumed_ok
        return seq, round_clean

    @jax.jit
    def replay_batch(
        decks, round_nums, honbas, kyotakus, dealers, scores, ev_actors, ev_actions, n_events
    ):
        seq, round_clean = jax.vmap(replay_one_round)(
            decks, round_nums, honbas, kyotakus, dealers, scores, ev_actors, ev_actions, n_events
        )
        step_ok, obs, ora, grp_f, tbl, rsnap, actions, masks, cps = seq
        # Mask out any desynced round completely!
        valid_mask = step_ok & round_clean[:, None]  # (B, T)

        # Compute sparse belief labels ONLY on active valid steps (zeroing inactive tail `can_win`!)
        b_sz, t_len = valid_mask.shape
        n_flat = b_sz * t_len
        flat_valid = valid_mask.reshape((n_flat,))
        flat_rsnap = jax.tree_util.tree_map(lambda x: x.reshape((n_flat,) + x.shape[2:]), rsnap)
        flat_rsnap = flat_rsnap._replace(can_win=flat_rsnap.can_win & flat_valid[:, None, None])
        y_ron, y_sb, y_ct, y_ten, overflow = compute_sparse_belief_labels(flat_rsnap, k_max=ron_k_max)
        y_ron = y_ron.reshape((b_sz, t_len, 3, 34)).astype(jnp.uint8)
        y_sb = y_sb.reshape((b_sz, t_len, 3, 34)).astype(jnp.uint8)
        y_ct = y_ct.reshape((b_sz, t_len, 3, 34)).astype(jnp.uint8)
        y_ten = y_ten.reshape((b_sz, t_len, 3)).astype(jnp.uint8)
        return valid_mask, round_clean, obs, ora, grp_f, tbl, actions, masks, cps, y_ron, y_sb, y_ct, y_ten, overflow

    return replay_batch


def _empty_shard_buf(include_oracle: bool = True) -> Dict[str, List[np.ndarray]]:
    buf: Dict[str, List[np.ndarray]] = {
        "planes": [],
        "scalars": [],
        "river_tokens": [],
        "river_tile_ids": [],
        "river_valid_mask": [],
        "last_draw_type": [],
        "is_last_draw_red": [],
        "target_type": [],
        "target_is_red": [],
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
    if include_oracle:
        buf["oracle_planes"] = []
    return buf


def _flush_shard_arrays(
    out_dir: Path,
    shard_idx: int,
    arrays: Dict[str, np.ndarray],
    compress: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_idx:05d}.npz"
    tmp_path = out_dir / f".shard_{shard_idx:05d}.tmp.npz"
    if compress:
        np.savez_compressed(tmp_path, **arrays)
    else:
        np.savez(tmp_path, **arrays)
    tmp_path.replace(path)
    return path


def convert_mjai_parquets_to_v6_shards(
    parquet_paths: Sequence[Path | str],
    out_dir: Path | str,
    *,
    max_games: Optional[int] = None,
    start_game: int = 0,
    hanchan_only: bool = True,
    round_batch_size: int = 64,
    max_steps_per_round: int = 192,
    max_events_per_round: int = 160,
    ron_k_max: int = 4096,
    shard_size: int = 50_000,
    storage_dtype: Literal["float16", "float32"] = "float16",
    include_oracle: bool = True,
    compress: bool = True,
    w_table_default: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Convert `hhim8826/tenhou-houou-mjai` Parquet files into ready-to-train V6 `.npz` shards.

    Spatial planes (`planes`, `oracle_planes`, `river_tokens`) default to `float16`, which is
    bit-exact for `{0, 1, k/4, k/8}` features and cuts disk usage by >2x with zero loss.
    """
    out_path = Path(out_dir)
    spatial_np_dtype = np.float16 if storage_dtype == "float16" else np.float32
    w_def = (
        np.asarray(make_homogeneous_table(W_PHOENIX_7D), dtype=np.float32)
        if w_table_default is None
        else np.asarray(w_table_default, dtype=np.float32)
    )

    replayer = build_gpu_batch_replayer(
        max_steps_per_round=max_steps_per_round,
        ron_k_max=ron_k_max,
        round_limit=7,
    )

    buf = _empty_shard_buf(include_oracle=include_oracle)
    buffered = 0
    shard_idx = 0
    written: List[Path] = []
    pending_writes: List[Future[Path]] = []
    writer_pool = ThreadPoolExecutor(max_workers=2)

    def _drain_oldest_write(drain_all: bool = False) -> None:
        while pending_writes and (drain_all or len(pending_writes) >= 2):
            fut = pending_writes.pop(0)
            written.append(fut.result())

    def _submit_shard() -> None:
        nonlocal buffered, shard_idx
        if buffered <= 0:
            return
        _drain_oldest_write(drain_all=False)
        arrays = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
        for k in buf:
            buf[k].clear()
        s_idx = shard_idx
        shard_idx += 1
        buffered = 0
        pending_writes.append(
            writer_pool.submit(_flush_shard_arrays, out_path, s_idx, arrays, compress)
        )
        if verbose:
            elapsed_now = max(time.perf_counter() - t0, 1e-6)
            tot_r = max(stats["rounds_clean"] + stats["rounds_desync"], 1)
            eta_str = ""
            if max_games is not None and stats["games_processed"] > 0:
                rem_g = max(0, max_games - stats["games_processed"])
                eta_s = rem_g / max(stats["games_processed"] / elapsed_now, 1e-6)
                eta_str = f" ETA={eta_s/60:.1f}m"
            print(
                f"[MJAI->V6] shard={s_idx:05d} games={stats['games_processed']}"
                f"{f'/{max_games}' if max_games else ''} "
                f"clean={stats['rounds_clean']}/{tot_r} ({100.0*stats['rounds_clean']/tot_r:.2f}%) "
                f"steps={stats['steps_written']:,} ({stats['steps_written']/elapsed_now:.0f} steps/s) "
                f"overflow={stats['ron_overflow_batches']}{eta_str}",
                flush=True,
            )

    pending_rounds: List[ParsedRound] = []
    stats = {
        "games_seen": 0,
        "games_processed": 0,
        "rounds_parsed": 0,
        "rounds_clean": 0,
        "rounds_desync": 0,
        "steps_written": 0,
        "pass_steps": 0,
        "ron_overflow_batches": 0,
    }
    t0 = time.perf_counter()

    def _run_gpu_batch(batch_rounds: List[ParsedRound]) -> None:
        nonlocal buffered, shard_idx
        actual_b = len(batch_rounds)
        if actual_b == 0:
            return
        # Pad tail batch to fixed `round_batch_size` to avoid XLA recompilation!
        padded = batch_rounds + [batch_rounds[-1]] * (round_batch_size - actual_b)

        decks = jnp.asarray(np.stack([r.deck for r in padded], axis=0), dtype=jnp.int8)
        r_nums = jnp.asarray([r.round_num for r in padded], dtype=jnp.int8)
        honbas = jnp.asarray([r.honba for r in padded], dtype=jnp.int8)
        kyotakus = jnp.asarray([r.kyotaku for r in padded], dtype=jnp.int8)
        dealers = jnp.asarray([r.dealer for r in padded], dtype=jnp.int8)
        scores = jnp.asarray(np.stack([r.scores for r in padded], axis=0), dtype=jnp.int32)
        ev_actr = jnp.asarray(np.stack([r.ev_actors for r in padded], axis=0), dtype=jnp.int8)
        ev_actn = jnp.asarray(np.stack([r.ev_actions for r in padded], axis=0), dtype=jnp.int32)
        n_evs = jnp.asarray([r.n_events for r in padded], dtype=jnp.int32)

        out = replayer(decks, r_nums, honbas, kyotakus, dealers, scores, ev_actr, ev_actn, n_evs)
        (
            valid_mask, round_clean, obs, ora, grp_f, tbl, actions, masks, cps,
            y_ron, y_sb, y_ct, y_ten, overflow,
        ) = jax.device_get(out)

        if bool(overflow):
            stats["ron_overflow_batches"] += 1

        # Slice back to `actual_b` (dropping padding replicas!)
        valid_mask = np.asarray(valid_mask[:actual_b], dtype=np.bool_)
        round_clean = np.asarray(round_clean[:actual_b], dtype=np.bool_)
        stats["rounds_clean"] += int(round_clean.sum())
        stats["rounds_desync"] += int((~round_clean).sum())

        r_idx, t_idx = np.nonzero(valid_mask)
        n_valid = int(r_idx.size)
        if n_valid == 0:
            return

        act_sel = np.asarray(actions[:actual_b], dtype=np.int32)[r_idx, t_idx]
        stats["steps_written"] += n_valid
        stats["pass_steps"] += int(np.sum(act_sel == int(Action.PASS)))

        rank_mats_b = np.stack([r.final_rank_matrix_4x4 for r in batch_rounds], axis=0)  # (actual_b, 4, 4)

        buf["planes"].append(np.asarray(obs["planes"][:actual_b])[r_idx, t_idx].astype(spatial_np_dtype))
        buf["scalars"].append(np.asarray(obs["scalars"][:actual_b])[r_idx, t_idx].astype(np.float32))
        buf["river_tokens"].append(np.asarray(obs["river_tokens"][:actual_b])[r_idx, t_idx].astype(spatial_np_dtype))
        buf["river_tile_ids"].append(np.asarray(obs["river_tile_ids"][:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["river_valid_mask"].append(np.asarray(obs["river_valid_mask"][:actual_b])[r_idx, t_idx].astype(np.bool_))
        buf["last_draw_type"].append(np.asarray(obs["last_draw_type"][:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["is_last_draw_red"].append(np.asarray(obs["is_last_draw_red"][:actual_b])[r_idx, t_idx].astype(np.bool_))
        buf["target_type"].append(np.asarray(obs["target_type"][:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["target_is_red"].append(np.asarray(obs["target_is_red"][:actual_b])[r_idx, t_idx].astype(np.bool_))
        if include_oracle:
            buf["oracle_planes"].append(np.asarray(ora[:actual_b])[r_idx, t_idx].astype(spatial_np_dtype))
        buf["action"].append(act_sel)
        buf["action_mask"].append(np.asarray(masks[:actual_b])[r_idx, t_idx].astype(np.bool_))
        buf["current_player"].append(np.asarray(cps[:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["grp_features"].append(np.asarray(grp_f[:actual_b])[r_idx, t_idx].astype(np.float32))
        buf["final_rank_matrix_4x4"].append(rank_mats_b[r_idx].astype(np.float32))
        buf["w_table_abs"].append(np.broadcast_to(w_def[None, :, :], (n_valid, 4, 4)).copy())
        buf["table_scores"].append(np.asarray(tbl.scores[:actual_b])[r_idx, t_idx].astype(np.int32))
        buf["table_round_num"].append(np.asarray(tbl.round_num[:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["table_round_limit"].append(np.asarray(tbl.round_limit[:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["table_honba"].append(np.asarray(tbl.honba[:actual_b])[r_idx, t_idx].astype(np.int32))
        buf["table_kyotaku"].append(np.asarray(tbl.kyotaku[:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["table_dealer"].append(np.asarray(tbl.dealer[:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["table_init_wind"].append(np.asarray(tbl.init_wind[:actual_b])[r_idx, t_idx].astype(np.int8))
        buf["table_wall_rem"].append(np.asarray(tbl.wall_rem[:actual_b])[r_idx, t_idx].astype(np.int32))
        buf["y_ron"].append(np.asarray(y_ron[:actual_b])[r_idx, t_idx].astype(np.uint8))
        buf["y_score_bin"].append(np.asarray(y_sb[:actual_b])[r_idx, t_idx].astype(np.uint8))
        buf["y_call_threat"].append(np.asarray(y_ct[:actual_b])[r_idx, t_idx].astype(np.uint8))
        buf["y_tenpai"].append(np.asarray(y_ten[:actual_b])[r_idx, t_idx].astype(np.uint8))

        buffered += n_valid
        if buffered >= shard_size:
            _submit_shard()

    try:
        stop = False
        for pq_path in parquet_paths:
            if stop:
                break
            pf = pq.ParquetFile(pq_path)
            for batch in pf.iter_batches(batch_size=256, columns=["kyoku_first", "aka_flag", "events"]):
                kfs = batch.column("kyoku_first").to_pylist()
                akas = batch.column("aka_flag").to_pylist()
                evs_col = batch.column("events").to_pylist()
                for kf, aka, ev_str in zip(kfs, akas, evs_col):
                    if hanchan_only and int(kf) != 0:
                        continue
                    if not bool(aka):
                        continue
                    stats["games_seen"] += 1
                    if stats["games_seen"] <= start_game:
                        continue
                    rounds = parse_mjai_game_to_rounds(ev_str, max_events_per_round=max_events_per_round)
                    stats["games_processed"] += 1
                    stats["rounds_parsed"] += len(rounds)
                    pending_rounds.extend(rounds)

                    while len(pending_rounds) >= round_batch_size:
                        _run_gpu_batch(pending_rounds[:round_batch_size])
                        pending_rounds = pending_rounds[round_batch_size:]

                    if max_games is not None and stats["games_processed"] >= max_games:
                        stop = True
                        break
                if stop:
                    break

        if pending_rounds:
            _run_gpu_batch(pending_rounds)
            pending_rounds.clear()

        if buffered > 0:
            _submit_shard()
        _drain_oldest_write(drain_all=True)
    finally:
        writer_pool.shutdown(wait=True)

    elapsed = max(time.perf_counter() - t0, 1e-6)
    total_rounds = max(stats["rounds_clean"] + stats["rounds_desync"], 1)
    result = {
        **stats,
        "start_game": start_game,
        "next_start_game": stats["games_seen"],
        "clean_round_rate": stats["rounds_clean"] / total_rounds,
        "elapsed_s": elapsed,
        "games_per_sec": stats["games_processed"] / elapsed,
        "steps_per_sec": stats["steps_written"] / elapsed,
        "shards_written": [str(p) for p in written],
        "total_shard_mb": sum(p.stat().st_size for p in written) / (1024 ** 2),
    }
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if verbose:
        print(
            f"[MJAI->V6] games={result['games_processed']} rounds={result['rounds_clean']}/{total_rounds} "
            f"({result['clean_round_rate']*100:.2f}% clean) steps={result['steps_written']:,} "
            f"(PASS={result['pass_steps']:,}) time={elapsed:.1f}s ({result['steps_per_sec']:.0f} steps/s) "
            f"shards={len(written)} ({result['total_shard_mb']:.1f} MB)"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Tenhou MJAI Parquet files to V6 .npz shards")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Directory with .parquet files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for V6 shards")
    parser.add_argument("--max-games", type=int, default=None, help="Optional cap on number of games")
    parser.add_argument("--start-game", type=int, default=0, help="Skip first N games (for resuming)")
    parser.add_argument("--round-batch-size", type=int, default=64, help="Parallel kyokus per GPU batch")
    parser.add_argument("--shard-size", type=int, default=50_000, help="Decision steps per output shard")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--no-compress", action="store_true", help="Write uncompressed .npz (faster I/O)")
    args = parser.parse_args()

    parquets = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No .parquet files found in {args.parquet_dir}")

    convert_mjai_parquets_to_v6_shards(
        parquets,
        args.out_dir,
        max_games=args.max_games,
        start_game=args.start_game,
        round_batch_size=args.round_batch_size,
        shard_size=args.shard_size,
        storage_dtype=args.dtype,
        compress=not args.no_compress,
        verbose=True,
    )


if __name__ == "__main__":
    main()

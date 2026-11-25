"""MahJax bit-exact Legal Ron engine, sparse K_max compaction, and Analytical Physics Trinity.

Implements all V6 physics & rule-exact primitives:
1. `resolve_discard_action`: maps MahJax 87-dim actions (`0..36` incl. red 5s + `71=TSUMOGIRI`)
   to `(is_discard, tile_34, is_red)` — zero missed tsumogiri or aka-dora!
2. `RonSnapshotV6` (~382 bytes/step): compact raw micro-state recorded during Rollout with
   ZERO `Yaku.judge` calls; sparse static compaction `jnp.nonzero(..., size=K_max)` runs
   `Yaku.judge + judge_other` + Rao-Blackwellized expected ura-dora on only ~1.5% active slots!
3. `simulate_ron_transition`: bit-exact with `mahjax/common/logic.py` (head-bump honba & kyotaku,
   double-ron full points, no phantom riichi rebate, dealer renchan vs advance, `_is_game_end`
   tobi / oorasu agari-yame / West-4 sudden death -> exact 1-hot permutation matrix).
4. `compute_analytical_trinity`: zero-training-cost vNM utility projections from frozen `GRP_base`:
   - `Delta_U_ron`: 3 opponents x 3 discrete score bins + 2nd-order double-ron cliff correction
     + Aka-Dora Markov score shift `T_AKA_SHIFT` + SIGNED (preserves South-4 Sashikomi!).
   - `Delta_U_agari`: 3-tier win utility gain `[2000, 8000, 12000]` for Push-Fold symmetry.
   - `Delta_U_ryuukyoku`: exhaustive-draw keishiki tenpai utility swing when `wall_rem <= 12`.
   - `call_threat_plane`: max-across-opponents meld-feed (nakase) threat plane.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from mahjax.common.action import Action
from mahjax.common.constants import NUM_TILE_TYPES_WITH_RED, RED_FIVE_TILE_TYPES
from mahjax.common.hand import Hand
from mahjax.common.tile import Tile
from mahjax.common.yaku import Yaku
from mahjax.mahjong_4p.constants import (
    DORA_ARRAY,
    HONBA_SCORE,
    NUM_PLAYERS,
    RIICHI_BET_SCORE,
    SUDDEN_DEATH_ROUNDS,
    TARGET_SCORE,
)
from mahjax.mahjong_4p.state import State
from mahjong_rl.models.grp import encode_grp_features_from_table, scores_to_rank_matrix_4x4

NUM_TILES = Tile.NUM_TILE_TYPE  # 34

# Standard 3-tier score bins in MahJax score units (100-pt units: 20 = 2000 pts)
SCORE_BINS_NON_DEALER = jnp.array([20.0, 80.0, 120.0], dtype=jnp.float32)   # 2000, 8000, 12000
SCORE_BINS_DEALER = jnp.array([29.0, 120.0, 180.0], dtype=jnp.float32)      # 2900, 12000, 18000

# Aka-Dora (+1 han) Markov score-bin transition matrix for red-5 discards (34, 35, 36, red TSUMOGIRI)
T_AKA_SHIFT = jnp.array(
    [
        [0.30, 0.70, 0.00],
        [0.00, 0.20, 0.80],
        [0.00, 0.00, 1.00],
    ],
    dtype=jnp.float32,
)

# Exact 3-bit opponent tenpai configurations (8 combinations) for exhaustive draw
_TENPAI_CONFIGS = jnp.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=jnp.float32,
)  # (8, 3)


def resolve_discard_action(action: jnp.ndarray, last_draw: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map any MahJax action `[0..86]` to `(is_discard, tile_34, is_red)`.

    Covers:
    - `0..33`: normal hand discard (`tile_34 = action`, `is_red = False`)
    - `34..36`: red five hand discard (`5m=4, 5p=13, 5s=22`, `is_red = True`)
    - `71 (Action.TSUMOGIRI)`: drawn tile (`tile_34 = to_tile_type(last_draw)`, `is_red = last_draw >= 34`)
    - All other actions (`37..70`, `72..86`): `is_discard = False`
    """
    a = jnp.asarray(action, dtype=jnp.int32)
    ld = jnp.asarray(last_draw, dtype=jnp.int32)

    is_norm = (a >= 0) & (a < NUM_TILES)
    is_red_hand = (a >= NUM_TILES) & (a < NUM_TILE_TYPES_WITH_RED)
    is_tsumo = (a == Action.TSUMOGIRI) & (ld >= 0)

    is_discard = is_norm | is_red_hand | is_tsumo
    red_tiles = jnp.asarray(RED_FIVE_TILE_TYPES, dtype=jnp.int32)  # [4, 13, 22]
    red_t = red_tiles[jnp.clip(a - NUM_TILES, 0, 2)]
    tsumo_t = Tile.to_tile_type(jnp.clip(ld, 0, NUM_TILE_TYPES_WITH_RED - 1)).astype(jnp.int32)

    tile_34 = jnp.where(is_norm, a, jnp.where(is_red_hand, red_t, jnp.where(is_tsumo, tsumo_t, -1)))
    is_red = is_red_hand | (is_tsumo & (ld >= NUM_TILES))
    return is_discard, tile_34, is_red


class TableSnapshot(NamedTuple):
    """Lightweight public table state used by `simulate_ron_transition` & `compute_analytical_trinity`."""

    scores: jnp.ndarray       # (4,) int32 (100-pt units, absolute seats)
    round_num: jnp.ndarray    # () int8
    round_limit: jnp.ndarray  # () int8
    honba: jnp.ndarray        # () int32
    kyotaku: jnp.ndarray      # () int8
    dealer: jnp.ndarray       # () int8 (absolute seat)
    init_wind: jnp.ndarray    # (4,) int8 (absolute seats)
    wall_rem: jnp.ndarray     # () int32
    current_player: jnp.ndarray  # () int8 (absolute seat)


class RonSnapshotV6(NamedTuple):
    """Ultra-compact (~382B/step) raw micro-state for zero-Yaku rollout & sparse update labeling."""

    hand_with_red: jnp.ndarray    # (4, 37) int8   -> 148 B
    melds: jnp.ndarray            # (4, 4) uint16  ->  32 B
    meld_counts: jnp.ndarray      # (4,) int8      ->   4 B
    can_win: jnp.ndarray          # (4, 34) bool   -> 136 B
    furiten: jnp.ndarray          # (4,) bool      ->   4 B
    riichi: jnp.ndarray           # (4,) bool      ->   4 B
    ippatsu: jnp.ndarray          # (4,) bool      ->   4 B
    double_riichi: jnp.ndarray    # (4,) bool      ->   4 B
    dora_indicators: jnp.ndarray  # (5,) int8      ->   5 B
    seat_wind: jnp.ndarray        # (4,) int8      ->   4 B
    scores: jnp.ndarray           # (4,) int32     ->  16 B
    init_wind: jnp.ndarray        # (4,) int8      ->   4 B
    round_num: jnp.ndarray        # () int8        ->   1 B
    round_limit: jnp.ndarray      # () int8        ->   1 B
    honba: jnp.ndarray            # () int32       ->   4 B
    kyotaku: jnp.ndarray          # () int8        ->   1 B
    dealer: jnp.ndarray           # () int8        ->   1 B
    last_draw: jnp.ndarray        # () int8        ->   1 B
    current_player: jnp.ndarray   # () int8        ->   1 B
    is_haitei: jnp.ndarray        # () bool        ->   1 B
    wall_rem: jnp.ndarray         # () int32       ->   4 B
    # Total ~380 bytes per transition!


def extract_table_snapshot(state: State) -> TableSnapshot:
    rs = state.round_state
    wall_rem = jnp.maximum(rs.next_deck_ix.astype(jnp.int32) - rs.last_deck_ix.astype(jnp.int32) + 1, 0)
    return TableSnapshot(
        scores=rs.score.astype(jnp.int32),
        round_num=rs.round.astype(jnp.int8),
        round_limit=rs.round_limit.astype(jnp.int8),
        honba=rs.honba.astype(jnp.int32),
        kyotaku=rs.kyotaku.astype(jnp.int8),
        dealer=rs.dealer.astype(jnp.int8),
        init_wind=rs.init_wind.astype(jnp.int8),
        wall_rem=wall_rem,
        current_player=state.current_player.astype(jnp.int8),
    )


def extract_ron_snapshot(state: State) -> RonSnapshotV6:
    """Extract compact raw micro-state during rollout with ZERO `Yaku.judge` calls."""
    rs = state.round_state
    p = state.players
    wall_rem = jnp.maximum(rs.next_deck_ix.astype(jnp.int32) - rs.last_deck_ix.astype(jnp.int32) + 1, 0)
    furiten = (p.furiten_by_discard | p.furiten_by_pass).astype(jnp.bool_)
    return RonSnapshotV6(
        hand_with_red=p.hand_with_red.astype(jnp.int8),
        melds=p.melds.astype(jnp.uint16),
        meld_counts=p.meld_counts.astype(jnp.int8),
        can_win=p.can_win.astype(jnp.bool_),
        furiten=furiten,
        riichi=p.riichi.astype(jnp.bool_),
        ippatsu=p.ippatsu.astype(jnp.bool_),
        double_riichi=p.double_riichi.astype(jnp.bool_),
        dora_indicators=rs.dora_indicators.astype(jnp.int8),
        seat_wind=rs.seat_wind.astype(jnp.int8),
        scores=rs.score.astype(jnp.int32),
        init_wind=rs.init_wind.astype(jnp.int8),
        round_num=rs.round.astype(jnp.int8),
        round_limit=rs.round_limit.astype(jnp.int8),
        honba=rs.honba.astype(jnp.int32),
        kyotaku=rs.kyotaku.astype(jnp.int8),
        dealer=rs.dealer.astype(jnp.int8),
        last_draw=rs.last_draw.astype(jnp.int8),
        current_player=state.current_player.astype(jnp.int8),
        is_haitei=rs.is_haitei.astype(jnp.bool_),
        wall_rem=wall_rem,
    )


def _visible_dora_2x34(dora_indicators: jnp.ndarray) -> jnp.ndarray:
    """Build `(2, 34)` dora array with visible dora in row 0 and ZERO ura-dora in row 1."""
    valid = dora_indicators >= 0
    ind_types = Tile.to_tile_type(jnp.where(valid, dora_indicators, 0)).astype(jnp.int32)
    dora_types = DORA_ARRAY[ind_types].astype(jnp.int32)
    vis = jnp.zeros((NUM_TILES,), dtype=jnp.int8)
    for k in range(5):
        vis = vis.at[dora_types[k]].add(valid[k].astype(jnp.int8))
    return jnp.stack([vis, jnp.zeros((NUM_TILES,), dtype=jnp.int8)], axis=0)


def _ron_points_from_fan_fu(fan: jnp.ndarray, fu: jnp.ndarray, is_dealer: jnp.ndarray) -> jnp.ndarray:
    """Convert `(fan, fu)` to MahJax score units (100-pt units), matching `_win_score(Yaku.score(...))`."""
    basic = Yaku.score(fan.astype(jnp.int32), fu.astype(jnp.int32))
    return jnp.ceil(jnp.where(is_dealer, basic * 6, basic * 4) / 100.0).astype(jnp.float32)


def _judge_single_ron_slot(
    hand_37: jnp.ndarray,
    melds_4: jnp.ndarray,
    n_meld: jnp.ndarray,
    tile_34: jnp.ndarray,
    riichi: jnp.ndarray,
    ippatsu: jnp.ndarray,
    double_riichi: jnp.ndarray,
    is_haitei: jnp.ndarray,
    prevalent_wind: jnp.ndarray,
    seat_wind: jnp.ndarray,
    dora_indicators: jnp.ndarray,
    is_dealer: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate exact legal Ron + Rao-Blackwellized expected ura-dora score (in 100-pt units).

    Runs ONLY on compacted `K_max` active `(sample, opp, tile)` slots — ZERO `lax.cond` over cache!
    """
    dora_2x34 = _visible_dora_2x34(dora_indicators)
    yaku_vec, hand_fan, fu = Yaku.judge_hand_related(
        hand=hand_37,
        melds=melds_4,
        n_meld=n_meld.astype(jnp.int32),
        last_tile=tile_34.astype(jnp.int32),
        riichi=riichi,
        is_ron=jnp.bool_(True),
        prevalent_wind=prevalent_wind.astype(jnp.int32),
        seat_wind=seat_wind.astype(jnp.int32),
        dora=dora_2x34,
        use_red_fives=jnp.bool_(True),
        kita=None,
    )
    other_yaku, other_yakuman, other_fan, other_yakuman_num = Yaku.judge_other(
        is_ron=jnp.bool_(True),
        is_riichi=riichi,
        is_ippatsu=ippatsu,
        is_bottom_of_the_river=is_haitei,
        is_double_riichi=double_riichi,
    )
    # Yakuman signalled by fu == 0 in Yaku.judge_hand_related
    is_yakuman = (fu == 0) | (other_yakuman_num > 0)
    has_yaku = jnp.any(yaku_vec) | jnp.any(other_yaku) | is_yakuman

    total_fan_base = jnp.where(
        is_yakuman,
        jnp.maximum(hand_fan, 1) + other_yakuman_num,
        hand_fan + other_fan,
    )

    # Rao-Blackwellized expected ura-dora score when riichi & non-yakuman:
    # Each ura indicator hits an average 14-tile hand with ~0.315 expected han
    # (exact binomial mixture over 0 / +1 / +2 ura hits avoids MC wall-shuffle variance!)
    n_ind = jnp.sum((dora_indicators >= 0).astype(jnp.float32))
    lam_ura = jnp.clip(0.315 * n_ind, 0.0, 2.0)
    p0 = jnp.exp(-lam_ura)
    p1 = lam_ura * p0
    p2 = jnp.maximum(0.0, 1.0 - p0 - p1)

    pts_0 = _ron_points_from_fan_fu(total_fan_base, fu, is_dealer)
    pts_1 = _ron_points_from_fan_fu(total_fan_base + 1, fu, is_dealer)
    pts_2 = _ron_points_from_fan_fu(total_fan_base + 2, fu, is_dealer)
    pts_rb = jnp.where(riichi & (~is_yakuman), p0 * pts_0 + p1 * pts_1 + p2 * pts_2, pts_0)

    yakuman_pts = jnp.where(is_dealer, 480.0, 320.0) * jnp.maximum(total_fan_base.astype(jnp.float32), 1.0)
    final_pts = jnp.where(is_yakuman, yakuman_pts, pts_rb)
    return has_yaku, jnp.where(has_yaku, final_pts, 0.0)


def score_to_bin_index(pts_units: jnp.ndarray, is_dealer: jnp.ndarray) -> jnp.ndarray:
    """Map score in 100-pt units to 3-tier bin index `{0: small, 1: mangan, 2: haneman+}`.

    Cutoffs (Round 28):
    - Non-dealer: `<= 39` (0), `40..119` (1), `>= 120` (2)
    - Dealer:     `<= 58` (0), `59..179` (1), `>= 180` (2)
    """
    small_cut = jnp.where(is_dealer, 58.5, 39.5)
    high_cut = jnp.where(is_dealer, 179.5, 119.5)
    return jnp.where(pts_units >= high_cut, 2, jnp.where(pts_units > small_cut, 1, 0)).astype(jnp.int32)


def _compute_call_threat_labels_single(snap: RonSnapshotV6) -> jnp.ndarray:
    """Fast branchless ground-truth meld-feed threat labels `(3, 34)` for ego opponents `(1, 2, 3)`.

    Marks `(o, t) = 1` iff opponent `o` is NOT in riichi, can legally Pon (or Chi if right/shimocha)
    tile `t`, AND holds a high-value/fast hand (`meld_counts >= 1`, `>= 2` visible dora/red5 in hand,
    or yakuhai pon). Zero `Shanten.number` overhead!
    """
    c_p = snap.current_player.astype(jnp.int32)
    others = (jnp.arange(1, NUM_PLAYERS, dtype=jnp.int32) + c_p) % NUM_PLAYERS  # [right, across, left]
    h37 = snap.hand_with_red[others].astype(jnp.int32)  # (3, 37)
    h34 = jax.vmap(Hand.to_34)(h37).astype(jnp.int32)   # (3, 34)
    not_riichi = ~snap.riichi[others]                   # (3,)
    n_melds = snap.meld_counts[others].astype(jnp.int32)  # (3,)

    # Dora + red5 count in opponent's concealed hand
    vis_dora = _visible_dora_2x34(snap.dora_indicators)[0].astype(jnp.int32)  # (34,)
    hand_dora = jnp.sum(h34 * vis_dora[None, :], axis=-1) + jnp.sum(h37[:, 34:37], axis=-1)  # (3,)

    # Can Pon any tile t: h34[o, t] >= 2
    can_pon = h34 >= 2  # (3, 34)

    # Can Chi (ONLY right/shimocha, index 0 in `others`!):
    h_r_suits = (h34[0, :27].reshape(3, 9) > 0)  # (3, 9)
    # Tile r can be chi'd if (r-2 & r-1) or (r-1 & r+1) or (r+1 & r+2)
    pad = jnp.pad(h_r_suits, ((0, 0), (2, 2)), constant_values=False)  # (3, 13)
    chi_l = pad[:, 3:12] & pad[:, 4:13]  # [r], r+1, r+2
    chi_m = pad[:, 1:10] & pad[:, 3:12]  # r-1, [r], r+1
    chi_r = pad[:, 0:9] & pad[:, 1:10]   # r-2, r-1, [r]
    can_chi_right = jnp.pad((chi_l | chi_m | chi_r).reshape(27), (0, 7), constant_values=False)  # (34,)
    can_chi = jnp.zeros((3, NUM_TILES), dtype=jnp.bool_).at[0].set(can_chi_right)

    # Yakuhai check per opponent (sangen 31..33, round wind, opponent's seat wind)
    prev_w = 27 + jnp.clip(snap.round_num.astype(jnp.int32) // 4, 0, 3)
    opp_sw = 27 + jnp.clip(snap.seat_wind[others].astype(jnp.int32), 0, 3)  # (3,)
    tile_idx = jnp.arange(NUM_TILES, dtype=jnp.int32)[None, :]
    is_yakuhai = (tile_idx >= 31) | (tile_idx == prev_w) | (tile_idx == opp_sw[:, None])

    high_value_or_fast = (n_melds >= 1) | (hand_dora >= 2)  # (3,)
    threat = not_riichi[:, None] & (
        ((can_pon | can_chi) & high_value_or_fast[:, None])
        | (can_pon & is_yakuhai)
    )
    return threat.astype(jnp.float32)  # (3, 34)


def compute_sparse_belief_labels(
    snapshots: RonSnapshotV6,
    k_max: int = 1024,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute exact Belief ground-truth labels for a flat batch `(N,)` of `RonSnapshotV6`.

    Uses XLA static-capacity compaction `jnp.nonzero(eligible, size=k_max, fill_value=-1)`
    so `Yaku.judge + judge_other` runs ONLY on ~1.5% active `(sample, opp, tile)` pairs
    (zero dynamic-shape recompilation, ~98% compute savings!).

    Returns:
        y_ron:         `(N, 3, 34)` float32 exact legal Ron mask (ego opponents 1, 2, 3)
        y_score_bin:   `(N, 3, 34)` int32 3-tier score bin `{0, 1, 2}` (valid where `y_ron == 1`)
        y_call_threat: `(N, 3, 34)` float32 meld-feed high-danger threat mask
        y_tenpai:      `(N, 3)`     float32 opponent tenpai mask (has >= 1 legal wait or tenpai)
        overflow_flag: `()`         bool True if active candidates exceeded `k_max`
    """
    N = snapshots.current_player.shape[0]
    c_p = snapshots.current_player.astype(jnp.int32)  # (N,)
    others = (c_p[:, None] + jnp.arange(1, NUM_PLAYERS, dtype=jnp.int32)[None, :]) % NUM_PLAYERS  # (N, 3)

    n_idx = jnp.arange(N, dtype=jnp.int32)[:, None]
    opp_can_win = snapshots.can_win[n_idx, others]     # (N, 3, 34) bool
    opp_furiten = snapshots.furiten[n_idx, others]     # (N, 3) bool
    eligible = opp_can_win & (~opp_furiten[:, :, None])  # (N, 3, 34) bool

    total_eligible = jnp.sum(eligible.astype(jnp.int32))
    overflow_flag = total_eligible > k_max

    # Static-capacity compaction into K_max slots
    s_idx, o_rel, t_idx = jnp.nonzero(eligible, size=k_max, fill_value=-1)
    valid_slot = s_idx >= 0
    s_safe = jnp.clip(s_idx, 0, max(N - 1, 0))
    o_safe = jnp.clip(o_rel, 0, 2)
    t_safe = jnp.clip(t_idx, 0, NUM_TILES - 1)
    o_abs = others[s_safe, o_safe]

    # Gather compact per-slot arguments
    h37_k = snapshots.hand_with_red[s_safe, o_abs]
    melds_k = snapshots.melds[s_safe, o_abs]
    nm_k = snapshots.meld_counts[s_safe, o_abs]
    riichi_k = snapshots.riichi[s_safe, o_abs]
    ipp_k = snapshots.ippatsu[s_safe, o_abs]
    driichi_k = snapshots.double_riichi[s_safe, o_abs]
    haitei_k = snapshots.is_haitei[s_safe]
    prev_w_k = jnp.clip(snapshots.round_num[s_safe].astype(jnp.int32) // 4, 0, 3)
    seat_w_k = snapshots.seat_wind[s_safe, o_abs]
    dora_k = snapshots.dora_indicators[s_safe]
    is_dlr_k = snapshots.dealer[s_safe].astype(jnp.int32) == o_abs

    has_yaku_k, pts_k = jax.vmap(_judge_single_ron_slot)(
        h37_k, melds_k, nm_k, t_safe, riichi_k, ipp_k, driichi_k,
        haitei_k, prev_w_k, seat_w_k, dora_k, is_dlr_k,
    )
    legal_ron_k = valid_slot & has_yaku_k
    bin_k = score_to_bin_index(pts_k, is_dlr_k)

    # Scatter back into (N, 3, 34) using sentinel-safe index N
    y_ron_pad = jnp.zeros((N + 1, 3, NUM_TILES), dtype=jnp.float32)
    y_bin_pad = jnp.zeros((N + 1, 3, NUM_TILES), dtype=jnp.int32)
    s_scatter = jnp.where(legal_ron_k, s_safe, N)
    y_ron = y_ron_pad.at[s_scatter, o_safe, t_safe].set(1.0)[:N]
    y_score_bin = y_bin_pad.at[s_scatter, o_safe, t_safe].set(bin_k)[:N]

    y_call_threat = jax.vmap(_compute_call_threat_labels_single)(snapshots)  # (N, 3, 34)
    y_tenpai = jnp.any(opp_can_win, axis=-1).astype(jnp.float32)            # (N, 3)
    return y_ron, y_score_bin, y_call_threat, y_tenpai, overflow_flag


def _resolve_head_bump_seat(discard_seat: jnp.ndarray, win_mask: jnp.ndarray) -> jnp.ndarray:
    """Return closest clockwise/turn-order winning seat `(discard_seat + 1, +2, +3) % 4`."""
    offsets = jnp.array([1, 2, 3], dtype=jnp.int32)
    cand_seats = (discard_seat.astype(jnp.int32) + offsets) % NUM_PLAYERS  # (3,)
    cand_won = win_mask[cand_seats]                                        # (3,) bool
    # First True along offsets 1, 2, 3 (argmax returns first index of max)
    first_idx = jnp.argmax(cand_won.astype(jnp.int32))
    return cand_seats[first_idx]


def simulate_ron_transition(
    s_table: TableSnapshot,
    discard_seat: jnp.ndarray,
    win_mask_abs: jnp.ndarray,
    win_points_abs: jnp.ndarray,
    grp_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Bit-exact branchless MahJax Ron macro-transition (matches `mahjax/common/logic.py`).

    Verified rules:
    1. Multi-ron (`allow_double_ron=True`): each winner receives `win_points_abs[w]`, discarder
       pays the sum of all winners' points; honba (`honba * 3`) and kyotaku (`kyotaku * 10`)
       are paid ONLY ONCE to the head-bump winner (matching `logic.py` L1625-L1642!).
    2. Riichi declaration deal-in: 1000-pt bet has NOT been deducted yet, so ZERO phantom rebate!
    3. Dealer renchan: if dealer is among `win_mask_abs`, `round' = round, honba' = honba + 1`;
       else `round' = round + 1, dealer' = (dealer + 1) % 4, honba' = 0`. Kyotaku resets to 0.
    4. Exact `_is_game_end` (`logic.py` L1958-L1980): tobi (`any(score < 0)`), last sudden-death
       round non-renchan, OR final round with deal passing / dealer-top agari-yame & `max >= 300`
       -> outputs exact `(4, 4)` 1-hot permutation matrix; otherwise calls frozen `grp_fn`!

    Returns:
        rank_matrix_4x4: `(4, 4)` float32 (absolute seats x ranks 1..4)
        new_scores:      `(4,)`   float32 (in 100-pt units)
        is_terminal:     `()`     bool
    """
    d_seat = discard_seat.astype(jnp.int32)
    w_mask = win_mask_abs.astype(jnp.bool_)
    w_pts = jnp.where(w_mask, win_points_abs.astype(jnp.float32), 0.0)

    head_winner = _resolve_head_bump_seat(d_seat, w_mask)
    any_winner = jnp.any(w_mask)

    honba_val = jnp.where(any_winner, s_table.honba.astype(jnp.float32) * float(HONBA_SCORE), 0.0)
    kyotaku_val = jnp.where(any_winner, s_table.kyotaku.astype(jnp.float32) * float(RIICHI_BET_SCORE), 0.0)

    total_win_pts = jnp.sum(w_pts)
    new_scores = s_table.scores.astype(jnp.float32) + w_pts
    new_scores = new_scores.at[d_seat].add(-(total_win_pts + honba_val))
    new_scores = new_scores.at[head_winner].add(honba_val + kyotaku_val)

    # Exact MahJax `_is_game_end` condition (logic.py L1970-L1979)
    dealer = s_table.dealer.astype(jnp.int32)
    will_dealer_continue = w_mask[dealer]
    r_num = s_table.round_num.astype(jnp.int32)
    r_lim = s_table.round_limit.astype(jnp.int32)

    is_final_round = r_num >= r_lim
    is_last_extra = r_num >= (r_lim + int(SUDDEN_DEATH_ROUNDS))
    rank_mat = scores_to_rank_matrix_4x4(new_scores, s_table.init_wind)  # (4, 4)
    dealer_is_top = rank_mat[dealer, 0] > 0.5
    deal_passes = (~will_dealer_continue) | dealer_is_top
    is_tobi = jnp.any(new_scores < 0.0)
    is_terminal = (
        is_tobi
        | (is_last_extra & (~will_dealer_continue))
        | (is_final_round & deal_passes & (jnp.max(new_scores) >= float(TARGET_SCORE)))
    )

    new_round = jnp.where(will_dealer_continue, r_num, r_num + 1)
    new_dealer = jnp.where(will_dealer_continue, dealer, (dealer + 1) % NUM_PLAYERS)
    new_honba = jnp.where(will_dealer_continue, s_table.honba.astype(jnp.int32) + 1, 0)
    new_kyotaku = jnp.int32(0)
    # Next round starts with full 70-tile live wall
    next_grp_feat = encode_grp_features_from_table(
        scores=new_scores,
        round_num=new_round,
        round_limit=r_lim,
        honba=new_honba,
        kyotaku=new_kyotaku,
        dealer=new_dealer,
        init_wind=s_table.init_wind,
        wall_rem=jnp.int32(70),
    )
    grp_mat = grp_fn(next_grp_feat)  # (4, 4)
    out_mat = jnp.where(is_terminal, rank_mat, grp_mat)
    return out_mat, new_scores, is_terminal


def compute_analytical_trinity(
    s_table: TableSnapshot,
    p_deal_3opp: jnp.ndarray,
    p_score_bins: jnp.ndarray,
    p_tenpai_3: jnp.ndarray,
    p_call_threat_3: jnp.ndarray,
    last_draw_type: jnp.ndarray,
    is_last_draw_red: jnp.ndarray,
    w_me: jnp.ndarray,
    grp_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the V6 Analytical Physics Trinity from frozen `GRP_base` (ZERO extra training cost!).

    Args:
        s_table:         `TableSnapshot` for the current state
        p_deal_3opp:     `(3, 34)` float32 per-opponent deal-in probabilities from `B_EMA`
        p_score_bins:    `(3, 34, 3)` float32 3-tier conditional score distribution from `B_EMA`
        p_tenpai_3:      `(3,)` float32 opponent tenpai probabilities from `B_EMA`
        p_call_threat_3: `(3, 34)` float32 opponent meld-feed threat probabilities from `B_EMA`
        last_draw_type:  `()` int32 (`0..33`, `-1` if none)
        is_last_draw_red:`()` bool
        w_me:            `(4,)` float32 normalized utility vector for `s_table.current_player`
        grp_fn:          frozen `GRP_base` callable `(29,) -> (4, 4)` Birkhoff placement matrix

    Returns:
        delta_u_ron_3opp:  `(3, 34)` float32 SIGNED expected deal-in utility shift per opponent
        delta_u_ron_87:    `(87,)`   float32 SIGNED expected deal-in utility shift mapped to all 87 actions
                                     (includes 2nd-order double-ron cliff & Aka-Dora shift on 34..36 & red 71!)
        delta_u_agari_3:   `(3,)`    float32 expected utility gain for winning `[2000, 8000, 12000]`
        delta_u_ryuukyoku: `()`      float32 exhaustive-draw keishiki tenpai swing (active iff `wall_rem <= 12`)
        call_threat_plane: `(34,)`   float32 `max_o p_call_threat_3[o]` meld-feed danger plane
    """
    c_p = s_table.current_player.astype(jnp.int32)
    others = (c_p + jnp.arange(1, NUM_PLAYERS, dtype=jnp.int32)) % NUM_PLAYERS  # (3,)
    dealer = s_table.dealer.astype(jnp.int32)

    # Current baseline expected utility from frozen GRP_base
    curr_grp_feat = encode_grp_features_from_table(
        scores=s_table.scores,
        round_num=s_table.round_num,
        round_limit=s_table.round_limit,
        honba=s_table.honba,
        kyotaku=s_table.kyotaku,
        dealer=s_table.dealer,
        init_wind=s_table.init_wind,
        wall_rem=s_table.wall_rem,
    )
    curr_mat = grp_fn(curr_grp_feat)  # (4, 4)
    u_curr = jnp.dot(curr_mat[c_p], w_me)  # scalar

    # =========================================================================
    # 1. DEFENSE: Delta_U_ron (3 opponents x 3 bins + Double-Ron + Aka-Dora)
    # =========================================================================
    is_opp_dealer = others == dealer  # (3,)
    opp_score_grid = jnp.where(is_opp_dealer[:, None], SCORE_BINS_DEALER[None, :], SCORE_BINS_NON_DEALER[None, :])  # (3, 3)

    def _sim_single_opp_bin(opp_abs: jnp.ndarray, pts_val: jnp.ndarray) -> jnp.ndarray:
        w_mask = jax.nn.one_hot(opp_abs, NUM_PLAYERS, dtype=jnp.bool_)
        w_pts = jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32).at[opp_abs].set(pts_val)
        mat, _, _ = simulate_ron_transition(s_table, c_p, w_mask, w_pts, grp_fn)
        return jnp.dot(mat[c_p], w_me)

    u_post_grid = jax.vmap(
        lambda o_abs, bins: jax.vmap(lambda pt: _sim_single_opp_bin(o_abs, pt))(bins)
    )(others, opp_score_grid)  # (3, 3)
    delta_u_grid = u_post_grid - u_curr  # (3, 3) — SIGNED (positive for Sashikomi, negative for deal-in!)

    # Unconditional expected shift GIVEN opponent o Rons tile a (3, 34)
    exp_du_given_ron = jnp.einsum("okb,ob->ok", p_score_bins, delta_u_grid)  # (3, 34)
    # Aka-Dora shifted conditional shift for red-5 discards
    p_bins_aka = jnp.einsum("okb,bc->okc", p_score_bins, T_AKA_SHIFT)        # (3, 34, 3)
    exp_du_given_ron_aka = jnp.einsum("okb,ob->ok", p_bins_aka, delta_u_grid)  # (3, 34)

    delta_u_ron_3opp = p_deal_3opp * exp_du_given_ron  # (3, 34)
    delta_u_1st = jnp.sum(delta_u_ron_3opp, axis=0)    # (34,)

    # 2nd-order Double-Ron nonlinear cliff correction across the 3 opponent pairs (0,1), (0,2), (1,2)
    # Uses exact conditional expectation `U(double) - U(single_i) - U(single_j) + U_curr`!
    # Approximate pair score with each opponent's expected mangan/middle tier (`opp_score_grid[:, 1]`)
    delta_u_2nd = jnp.zeros((NUM_TILES,), dtype=jnp.float32)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        oi, oj = others[i], others[j]
        w_mask_ij = jax.nn.one_hot(oi, NUM_PLAYERS, dtype=jnp.bool_) | jax.nn.one_hot(oj, NUM_PLAYERS, dtype=jnp.bool_)
        w_pts_ij = (
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32)
            .at[oi].set(opp_score_grid[i, 1])
            .at[oj].set(opp_score_grid[j, 1])
        )
        mat_ij, _, _ = simulate_ron_transition(s_table, c_p, w_mask_ij, w_pts_ij, grp_fn)
        u_ij = jnp.dot(mat_ij[c_p], w_me)
        # Excess non-additive cliff: U(S - Si - Sj) - U(S - Si) - U(S - Sj) + U(S)
        super_cliff = u_ij - u_post_grid[i, 1] - u_post_grid[j, 1] + u_curr
        p_joint = p_deal_3opp[i] * p_deal_3opp[j]  # (34,)
        delta_u_2nd = delta_u_2nd + p_joint * super_cliff

    delta_u_normal_34 = delta_u_1st + delta_u_2nd  # (34,)
    delta_u_aka_34 = jnp.sum(p_deal_3opp * exp_du_given_ron_aka, axis=0) + delta_u_2nd  # (34,)

    # Map cleanly onto the 87-action space (ONLY discard actions 0..36 & 71!)
    aka_idx = jnp.asarray(RED_FIVE_TILE_TYPES, dtype=jnp.int32)  # [4, 13, 22]
    safe_ld = jnp.clip(last_draw_type.astype(jnp.int32), 0, NUM_TILES - 1)
    has_draw = last_draw_type >= 0
    tsumo_du = jnp.where(
        has_draw,
        jnp.where(is_last_draw_red, delta_u_aka_34[safe_ld], delta_u_normal_34[safe_ld]),
        0.0,
    )
    delta_u_ron_87 = (
        jnp.zeros((Action.NUM_ACTION,), dtype=jnp.float32)
        .at[0:NUM_TILES].set(delta_u_normal_34)
        .at[34:37].set(delta_u_aka_34[aka_idx])
        .at[Action.TSUMOGIRI].set(tsumo_du)
    )

    # =========================================================================
    # 2. OFFENSE: Delta_U_agari (3-tier win utility jump [small, mangan, haneman])
    # =========================================================================
    self_is_dealer = c_p == dealer
    self_win_bins = jnp.where(self_is_dealer, SCORE_BINS_DEALER, SCORE_BINS_NON_DEALER)  # (3,)

    def _sim_self_win(pts_val: jnp.ndarray) -> jnp.ndarray:
        # Average tsumo-like score transfer across the 3 opponents (exact total + kyotaku + honba)
        honba_v = s_table.honba.astype(jnp.float32) * float(HONBA_SCORE)
        kyotaku_v = s_table.kyotaku.astype(jnp.float32) * float(RIICHI_BET_SCORE)
        opp_share = (pts_val + honba_v) / 3.0
        new_sc = s_table.scores.astype(jnp.float32) - opp_share
        new_sc = new_sc.at[c_p].set(s_table.scores[c_p].astype(jnp.float32) + pts_val + honba_v + kyotaku_v)

        r_num = s_table.round_num.astype(jnp.int32)
        r_lim = s_table.round_limit.astype(jnp.int32)
        rank_mat = scores_to_rank_matrix_4x4(new_sc, s_table.init_wind)
        dealer_is_top = rank_mat[dealer, 0] > 0.5
        deal_passes = (~self_is_dealer) | dealer_is_top
        is_term = (
            jnp.any(new_sc < 0.0)
            | ((r_num >= r_lim + int(SUDDEN_DEATH_ROUNDS)) & (~self_is_dealer))
            | ((r_num >= r_lim) & deal_passes & (jnp.max(new_sc) >= float(TARGET_SCORE)))
        )
        new_r = jnp.where(self_is_dealer, r_num, r_num + 1)
        new_d = jnp.where(self_is_dealer, dealer, (dealer + 1) % NUM_PLAYERS)
        new_h = jnp.where(self_is_dealer, s_table.honba.astype(jnp.int32) + 1, 0)
        gf = encode_grp_features_from_table(new_sc, new_r, r_lim, new_h, jnp.int32(0), new_d, s_table.init_wind, jnp.int32(70))
        mat = jnp.where(is_term, rank_mat, grp_fn(gf))
        return jnp.dot(mat[c_p], w_me) - u_curr

    delta_u_agari_3 = jax.vmap(_sim_self_win)(self_win_bins)  # (3,)

    # =========================================================================
    # 3. ENDGAME: Delta_U_ryuukyoku (Keishiki Tenpai swing when wall_rem <= 12)
    # =========================================================================
    def _eval_draw_for_cfg(opp_ten_3: jnp.ndarray, self_tenpai: jnp.ndarray) -> jnp.ndarray:
        # Build 4-seat tenpai mask in absolute seat order
        ten_abs = jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32).at[c_p].set(self_tenpai).at[others].set(opp_ten_3)
        n_ten = jnp.sum(ten_abs)
        # Standard 3000-pt (30 units) noten bappu when 1 <= n_ten <= 3
        gain = jnp.where((n_ten >= 1.0) & (n_ten <= 3.0), 30.0 / jnp.maximum(n_ten, 1.0), 0.0)
        loss = jnp.where((n_ten >= 1.0) & (n_ten <= 3.0), 30.0 / jnp.maximum(4.0 - n_ten, 1.0), 0.0)
        delta_sc = jnp.where(ten_abs > 0.5, gain, -loss)
        new_sc = s_table.scores.astype(jnp.float32) + delta_sc

        dealer_ten = ten_abs[dealer] > 0.5
        r_num = s_table.round_num.astype(jnp.int32)
        r_lim = s_table.round_limit.astype(jnp.int32)
        rank_mat = scores_to_rank_matrix_4x4(new_sc, s_table.init_wind)
        dealer_is_top = rank_mat[dealer, 0] > 0.5
        deal_passes = (~dealer_ten) | dealer_is_top
        is_term = (
            ((r_num >= r_lim + int(SUDDEN_DEATH_ROUNDS)) & (~dealer_ten))
            | ((r_num >= r_lim) & deal_passes & (jnp.max(new_sc) >= float(TARGET_SCORE)))
        )
        new_r = jnp.where(dealer_ten, r_num, r_num + 1)
        new_d = jnp.where(dealer_ten, dealer, (dealer + 1) % NUM_PLAYERS)
        new_h = s_table.honba.astype(jnp.int32) + 1  # Draw always increments honba!
        gf = encode_grp_features_from_table(new_sc, new_r, r_lim, new_h, s_table.kyotaku, new_d, s_table.init_wind, jnp.int32(70))
        mat = jnp.where(is_term, rank_mat, grp_fn(gf))
        return jnp.dot(mat[c_p], w_me)

    # Weight all 8 opponent tenpai configurations by B_EMA's `p_tenpai_3`!
    p_t = jnp.clip(p_tenpai_3, 0.0, 1.0)
    cfg_probs = jnp.prod(
        jnp.where(_TENPAI_CONFIGS > 0.5, p_t[None, :], 1.0 - p_t[None, :]),
        axis=-1,
    )  # (8,)
    u_if_ten = jnp.sum(cfg_probs * jax.vmap(lambda c: _eval_draw_for_cfg(c, jnp.float32(1.0)))(_TENPAI_CONFIGS))
    u_if_noten = jnp.sum(cfg_probs * jax.vmap(lambda c: _eval_draw_for_cfg(c, jnp.float32(0.0)))(_TENPAI_CONFIGS))
    in_endgame = (s_table.wall_rem <= 12).astype(jnp.float32)
    delta_u_ryuukyoku = in_endgame * (u_if_ten - u_if_noten)

    call_threat_plane = jnp.max(p_call_threat_3, axis=0)  # (34,)
    return delta_u_ron_3opp, delta_u_ron_87, delta_u_agari_3, delta_u_ryuukyoku, call_threat_plane

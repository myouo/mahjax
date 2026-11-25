"""Pure-JAX lightweight 2D feature planes (obs_lean: 20x34, obs_v2: 36x34) + global scalars.

Designed for high-throughput GPU training on MahJax 4-player environments (`red_4p` / `no_red_4p`):
- Always clears stale `terminated` / `truncated` flags left by `auto_reset` on step 0.
- Never reads stale `state.players.can_win` for the acting player (uses freshly computed
  `discard_shanten` and `discard_can_win` from `observe_dict`).
- Zero hidden information leakage: uses only the current player's private hand and public table state.
"""

from __future__ import annotations

from typing import Dict, Literal

import jax
import jax.numpy as jnp

from mahjax.common.action import Action
from mahjax.common.meld import EMPTY_MELD, Meld
from mahjax.common.shanten import Shanten
from mahjax.common.tile import River, Tile
from mahjax.mahjong_4p.constants import DORA_ARRAY, NUM_PLAYERS
from mahjax.mahjong_4p.observation import _observe_dict
from mahjax.mahjong_4p.state import State

NUM_TILES = Tile.NUM_TILE_TYPE  # 34
NUM_LEAN_PLANES = 20
NUM_V2_PLANES = 36
GLOBAL_SCALAR_DIM = 49

# Precomputed suit indices for fast vectorized suji calculation
_SUIT_BASES = jnp.array([0, 9, 18], dtype=jnp.int32)


def clean_auto_reset_state(state: State) -> State:
    """Clear stale terminated/truncated flags left by `auto_reset` before observing."""
    return state.replace(
        terminated=jnp.bool_(False),
        truncated=jnp.bool_(False),
    )


def _thermometer_4(counts: jnp.ndarray) -> jnp.ndarray:
    """Convert integer counts in [0, 4] of shape (..., 34) into (..., 4, 34) float32 planes."""
    thresholds = jnp.arange(1, 5, dtype=counts.dtype)[..., None]  # (4, 1)
    return (counts[..., None, :] >= thresholds).astype(jnp.float32)


def _scatter_add_34(indices: jnp.ndarray, weights: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    """Safe scatter-add into a (34,) float32 vector (guards negative sentinel indices)."""
    safe_idx = jnp.where(valid, jnp.clip(indices, 0, NUM_TILES - 1), 0)
    safe_w = jnp.where(valid, weights, 0.0)
    return jnp.zeros((NUM_TILES,), dtype=jnp.float32).at[safe_idx].add(safe_w)


def _meld_planes_per_seat(melds_rotated: jnp.ndarray) -> jnp.ndarray:
    """Encode each seat's melds (4, 4) uint16 -> (4, 34) normalized tile counts in [0, 1]."""

    def one_seat(seat_melds: jnp.ndarray) -> jnp.ndarray:
        valid = seat_melds != EMPTY_MELD
        action = Meld.action(seat_melds)
        target = Meld.target(seat_melds)  # tile type [0..33]
        is_chi = Meld.is_chi(seat_melds)
        is_kan = Meld.is_kan(seat_melds)

        counts = _scatter_add_34(
            target,
            jnp.where(is_kan, 4.0, 3.0),
            valid & ~is_chi,
        )
        chi_low = target - Meld._chi_index(action)
        for offset in range(3):
            counts = counts + _scatter_add_34(chi_low + offset, jnp.ones_like(target, dtype=jnp.float32), valid & is_chi)
        return jnp.clip(counts / 4.0, 0.0, 1.0)

    return jax.vmap(one_seat)(melds_rotated)  # (4, 34)


def _dora_count_plane(dora_indicators: jnp.ndarray) -> jnp.ndarray:
    """Map revealed dora indicators (5,) -> actual dora tile counts (34,) normalized by 4."""
    valid = dora_indicators >= 0
    ind_types = Tile.to_tile_type(jnp.where(valid, dora_indicators, 0))
    dora_types = DORA_ARRAY[ind_types].astype(jnp.int32)
    counts = _scatter_add_34(dora_types, jnp.ones_like(dora_types, dtype=jnp.float32), valid)
    return jnp.clip(counts / 4.0, 0.0, 1.0)


def _compute_genbutsu_planes(
    river_tiles: jnp.ndarray,
    action_history: jnp.ndarray,
    riichi_rotated: jnp.ndarray,
) -> jnp.ndarray:
    """Compute exact genbutsu (safe tile) masks (3, 34) for opponents (right, across, left).

    Includes:
    1. Every tile discarded by opponent `o` (even if called away).
    2. If opponent `o` is in riichi, every discard by ANY player after `o`'s RIICHI action.
    """
    # action_history: (3, 200) -> [rel_actor, action_or_tile, tsumogiri]
    hist_actor = action_history[0].astype(jnp.int32)
    hist_act = action_history[1].astype(jnp.int32)
    hist_tsumo = action_history[2].astype(jnp.int32)
    hist_len = hist_actor.shape[0]
    positions = jnp.arange(hist_len, dtype=jnp.int32)

    # Discards in action_history are identified by tsumogiri >= 0 (0 or 1); action stores tile [0..36]
    is_discard_event = (hist_act >= 0) & (hist_tsumo >= 0)
    discard_tile_types = Tile.to_tile_type(jnp.where(is_discard_event, hist_act, 0))

    def one_opp(opp_idx: jnp.ndarray) -> jnp.ndarray:
        # 1. Own river discards
        opp_river = river_tiles[opp_idx]  # (24,)
        valid_r = opp_river >= 0
        r_types = Tile.to_tile_type(jnp.where(valid_r, opp_river, 0))
        own_discards = _scatter_add_34(r_types, jnp.ones_like(r_types, dtype=jnp.float32), valid_r)

        # 2. All table discards at/after opp's RIICHI declaration
        is_opp_riichi_evt = (hist_actor == opp_idx) & (hist_act == Action.RIICHI)
        riichi_pos = jnp.where(is_opp_riichi_evt, positions, hist_len).min()
        # Only count post-riichi table discards if opp is actually in riichi
        post_riichi_discard = riichi_rotated[opp_idx] & is_discard_event & (positions > riichi_pos)
        post_discards = _scatter_add_34(
            discard_tile_types,
            jnp.ones_like(discard_tile_types, dtype=jnp.float32),
            post_riichi_discard,
        )
        return ((own_discards + post_discards) > 0).astype(jnp.float32)

    return jax.vmap(one_opp)(jnp.array([1, 2, 3], dtype=jnp.int32))  # (3, 34)


def _compute_suji_planes(genbutsu: jnp.ndarray) -> jnp.ndarray:
    """Derive suji safety planes (3, 34) from opponent genbutsu (3, 34).

    For each suit (0..8, 9..17, 18..26):
    - 1 (idx 0) is suji if 4 (idx 3) is genbutsu (0.5)
    - 2 (idx 1) is suji if 5 (idx 4) is genbutsu (0.5)
    - 3 (idx 2) is suji if 6 (idx 5) is genbutsu (0.5)
    - 4 (idx 3) gets 0.5 per end (1 and 7), so 1.0 if double-suji
    - 5 (idx 4) gets 0.5 per end (2 and 8), so 1.0 if double-suji
    - 6 (idx 5) gets 0.5 per end (3 and 9), so 1.0 if double-suji
    - 7 (idx 6) is suji if 4 (idx 3) is genbutsu (0.5)
    - 8 (idx 7) is suji if 5 (idx 4) is genbutsu (0.5)
    - 9 (idx 8) is suji if 6 (idx 5) is genbutsu (0.5)
    """

    def one_opp(g: jnp.ndarray) -> jnp.ndarray:
        suits = g[:27].reshape(3, 9)  # (3, 9)
        suji = jnp.zeros((3, 9), dtype=jnp.float32)
        suji = suji.at[:, 0].set(0.5 * suits[:, 3])
        suji = suji.at[:, 1].set(0.5 * suits[:, 4])
        suji = suji.at[:, 2].set(0.5 * suits[:, 5])
        suji = suji.at[:, 3].set(0.5 * (suits[:, 0] + suits[:, 6]))
        suji = suji.at[:, 4].set(0.5 * (suits[:, 1] + suits[:, 7]))
        suji = suji.at[:, 5].set(0.5 * (suits[:, 2] + suits[:, 8]))
        suji = suji.at[:, 6].set(0.5 * suits[:, 3])
        suji = suji.at[:, 7].set(0.5 * suits[:, 4])
        suji = suji.at[:, 8].set(0.5 * suits[:, 5])
        return jnp.concatenate([suji.reshape(27), jnp.zeros((7,), dtype=jnp.float32)], axis=0)

    return jax.vmap(one_opp)(genbutsu)  # (3, 34)


def _compute_real_time_rank_onehot(state: State, c_p: jnp.ndarray) -> jnp.ndarray:
    """Compute observer's current placement one-hot (4,) using Tenhou tie-break (init_wind)."""
    scores = state.round_state.score  # (4,)
    init_wind = state.round_state.init_wind  # (4,)
    # Higher score ranks ahead; on equal score smaller init_wind ranks ahead
    my_score = scores[c_p]
    my_wind = init_wind[c_p]
    ahead = (scores > my_score) | ((scores == my_score) & (init_wind < my_wind))
    rank_idx = ahead.sum().astype(jnp.int32)  # 0=1st, 1=2nd, 2=3rd, 3=4th
    return jax.nn.one_hot(rank_idx, NUM_PLAYERS, dtype=jnp.float32)


def encode_global_scalars(state: State, obs_dict: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Encode compact table/rank context into a (GLOBAL_SCALAR_DIM,) float32 vector."""
    c_p = state.current_player
    scores = obs_dict["scores"].astype(jnp.float32)  # (4,) rotated
    scores_norm = jnp.clip((scores - 250.0) / 250.0, -3.0, 3.0)
    score_diffs = jnp.clip((scores[1:] - scores[0]) / 250.0, -3.0, 3.0)
    my_rank_oh = _compute_real_time_rank_onehot(state, c_p)

    round_val = obs_dict["round"].astype(jnp.float32)
    round_limit = jnp.maximum(obs_dict["round_limit"].astype(jnp.float32), 1.0)
    round_progress = jnp.clip(round_val / (round_limit + 1.0), 0.0, 2.0)[None]
    is_sudden_death = (round_val > round_limit).astype(jnp.float32)[None]
    honba_norm = jnp.clip(obs_dict["honba"].astype(jnp.float32) / 8.0, 0.0, 2.0)[None]
    kyotaku_norm = jnp.clip(obs_dict["kyotaku"].astype(jnp.float32) / 5.0, 0.0, 2.0)[None]
    wall_norm = (obs_dict["wall_remaining"].astype(jnp.float32) / 70.0)[None]

    prev_wind_oh = jax.nn.one_hot(jnp.clip(obs_dict["prevalent_wind"].astype(jnp.int32), 0, 2), 3, dtype=jnp.float32)
    seat_wind_oh = jax.nn.one_hot(jnp.clip(obs_dict["seat_wind"].astype(jnp.int32), 0, 3), 4, dtype=jnp.float32)

    riichi = obs_dict["riichi"].astype(jnp.float32)
    ippatsu = obs_dict["ippatsu"].astype(jnp.float32)
    concealed = obs_dict["is_hand_concealed"].astype(jnp.float32)

    shanten_norm = ((obs_dict["shanten_count"].astype(jnp.float32) + 1.0) / 7.0)[None]
    furiten = obs_dict["furiten"].astype(jnp.float32)[None]
    is_discard = obs_dict["is_discard_node"].astype(jnp.float32)[None]
    has_target = (obs_dict["target"] >= 0).astype(jnp.float32)[None]
    last_player_oh = jax.nn.one_hot(obs_dict["last_player"].astype(jnp.int32) + 1, NUM_PLAYERS + 1, dtype=jnp.float32)

    mask = state.legal_action_mask
    can_call_flags = jnp.array(
        [
            mask[Action.RIICHI],
            mask[Action.TSUMO],
            mask[Action.RON],
            mask[Action.PON] | mask[Action.PON_RED] | mask[Action.OPEN_KAN] | mask[37:71].any(),
            mask[Action.CHI_L : Action.CHI_R_RED + 1].any(),
        ],
        dtype=jnp.float32,
    )

    return jnp.concatenate(
        [
            scores_norm,        # 4
            score_diffs,        # 3
            my_rank_oh,         # 4
            round_progress,     # 1
            is_sudden_death,    # 1
            honba_norm,         # 1
            kyotaku_norm,       # 1
            wall_norm,          # 1
            prev_wind_oh,       # 3
            seat_wind_oh,       # 4
            riichi,             # 4
            ippatsu,            # 4
            concealed,          # 4
            shanten_norm,       # 1
            furiten,            # 1
            is_discard,         # 1
            has_target,         # 1
            last_player_oh,     # 5
            can_call_flags,     # 5
        ],
        axis=0,
    )


def _encode_base_components(state: State):
    """Shared public + private encoding used by both `obs_lean` and `obs_v2`."""
    clean_state = clean_auto_reset_state(state)
    obs = _observe_dict(clean_state)

    c_p = clean_state.current_player
    order = (jnp.arange(NUM_PLAYERS) + c_p) % NUM_PLAYERS

    hand_34 = obs["hand"].astype(jnp.int32)
    hand_planes = _thermometer_4(hand_34)  # (4, 34)
    red_plane = obs["is_red"].astype(jnp.float32)[None, :]  # (1, 34)

    last_draw = obs["last_draw"].astype(jnp.int32)
    last_draw_type = Tile.to_tile_type(jnp.where(last_draw >= 0, last_draw, 0))
    last_draw_plane = jnp.where(
        last_draw >= 0,
        jax.nn.one_hot(last_draw_type, NUM_TILES, dtype=jnp.float32),
        jnp.zeros((NUM_TILES,), dtype=jnp.float32),
    )[None, :]  # (1, 34)

    tiles_seen = obs["tiles_seen"].astype(jnp.int32)
    seen_planes = _thermometer_4(tiles_seen)  # (4, 34)
    dora_plane = _dora_count_plane(obs["dora_indicators"])[None, :]  # (1, 34)

    melds_rot = clean_state.players.melds[order]
    meld_planes = _meld_planes_per_seat(melds_rot)  # (4, 34)

    river_decoded = River.decode_river(clean_state.players.river[order])  # (6, 4, 24)
    river_tiles = river_decoded[0]  # (4, 24)
    genbutsu_planes = _compute_genbutsu_planes(river_tiles, obs["action_history"], obs["riichi"])  # (3, 34)

    # Shanten-optimal discard indicator (freshly computed, unaffected by stale can_win)
    ds = obs["discard_shanten"].astype(jnp.int32)  # (34, 3)
    ds_min = ds.min(axis=-1)  # (34,)
    best_ds = ds_min.min()
    is_opt_discard = (
        obs["is_discard_node"]
        & (hand_34 > 0)
        & (ds_min == best_ds)
        & (best_ds < int(Shanten.NOT_IN_HAND))
    ).astype(jnp.float32)[None, :]  # (1, 34)

    # Live remaining wait count per discard (using discard_can_win * remaining unseen copies)
    dcw = (obs["discard_can_win"] > 0).astype(jnp.float32)  # (34, 34)
    remaining_copies = jnp.clip(4.0 - tiles_seen.astype(jnp.float32), 0.0, 4.0)  # (34,)
    live_waits = jnp.clip((dcw @ remaining_copies) / 16.0, 0.0, 1.0)[None, :]  # (1, 34)

    lean_planes = jnp.concatenate(
        [
            hand_planes,      # 4
            red_plane,        # 1
            last_draw_plane,  # 1
            seen_planes,      # 4
            dora_plane,       # 1
            meld_planes,      # 4
            genbutsu_planes,  # 3
            is_opt_discard,   # 1
            live_waits,       # 1
        ],
        axis=0,
    )  # (20, 34)

    scalars = encode_global_scalars(clean_state, obs)
    return clean_state, obs, lean_planes, scalars, river_decoded, genbutsu_planes


def encode_obs_lean(state: State) -> Dict[str, jnp.ndarray]:
    """Encode 20-plane lightweight 2D observation + global scalars."""
    _, _, lean_planes, scalars, _, _ = _encode_base_components(state)
    return {"planes": lean_planes, "scalars": scalars}


def encode_obs_v2(state: State) -> Dict[str, jnp.ndarray]:
    """Encode 36-plane full tactical 2D observation + global scalars."""
    clean_state, obs, lean_planes, scalars, river_decoded, genbutsu_planes = _encode_base_components(state)

    river_tiles = river_decoded[0]       # (4, 24)
    river_riichi = river_decoded[1]      # (4, 24)
    river_tsumo = river_decoded[3]       # (4, 24)
    recency_weights = 0.2 + 0.8 * (jnp.arange(river_tiles.shape[1], dtype=jnp.float32) / 23.0)

    # 1. Per-seat recency-weighted river discard map (4, 34)
    def seat_river_recency(tiles_row: jnp.ndarray) -> jnp.ndarray:
        valid = tiles_row >= 0
        types = Tile.to_tile_type(jnp.where(valid, tiles_row, 0))
        return jnp.clip(_scatter_add_34(types, recency_weights, valid), 0.0, 1.5)

    river_recency_planes = jax.vmap(seat_river_recency)(river_tiles)  # (4, 34)

    # 2. Opponent tedashi (hand-cut) & riichi declaration tile map (3, 34)
    def opp_tedashi_map(opp_idx: jnp.ndarray) -> jnp.ndarray:
        tiles_row = river_tiles[opp_idx]
        valid = tiles_row >= 0
        types = Tile.to_tile_type(jnp.where(valid, tiles_row, 0))
        is_tedashi = (river_tsumo[opp_idx] == 0).astype(jnp.float32)
        is_r_decl = river_riichi[opp_idx].astype(jnp.float32)
        weights = jnp.clip(0.5 * is_tedashi * recency_weights + 0.5 * is_r_decl, 0.0, 1.0)
        return jnp.clip(_scatter_add_34(types, weights, valid), 0.0, 1.5)

    opp_tedashi_planes = jax.vmap(opp_tedashi_map)(jnp.array([1, 2, 3], dtype=jnp.int32))  # (3, 34)

    # 3. 3-decomposition discard shanten normalized (3, 34)
    ds = obs["discard_shanten"].astype(jnp.float32)  # (34, 3)
    ds_norm = jnp.stack(
        [
            jnp.clip(ds[:, 0] / 8.0, 0.0, 1.0),
            jnp.clip(ds[:, 1] / 13.0, 0.0, 1.0),
            jnp.clip(ds[:, 2] / 13.0, 0.0, 1.0),
        ],
        axis=0,
    )  # (3, 34)

    # 4. Pending call target tile one-hot (1, 34)
    target = obs["target"].astype(jnp.int32)
    target_type = Tile.to_tile_type(jnp.where(target >= 0, target, 0))
    target_plane = jnp.where(
        target >= 0,
        jax.nn.one_hot(target_type, NUM_TILES, dtype=jnp.float32),
        jnp.zeros((NUM_TILES,), dtype=jnp.float32),
    )[None, :]  # (1, 34)

    # 5. Opponent suji safety planes (3, 34)
    suji_planes = _compute_suji_planes(genbutsu_planes)  # (3, 34)

    # 6. Kabe / dead-tile barrier indicator (1, 34): 1.0 if 4 seen (no-chance), 0.5 if 3 seen (one-chance)
    tiles_seen = obs["tiles_seen"].astype(jnp.int32)
    kabe_plane = jnp.where(tiles_seen >= 4, 1.0, jnp.where(tiles_seen == 3, 0.5, 0.0)).astype(jnp.float32)[None, :]

    # 7. Legal discard mask over 34 tile types (1, 34)
    mask = clean_state.legal_action_mask
    legal_34 = mask[:NUM_TILES].astype(jnp.bool_)
    legal_34 = legal_34.at[4].set(legal_34[4] | mask[Tile.RED_FIVE["m"]])
    legal_34 = legal_34.at[13].set(legal_34[13] | mask[Tile.RED_FIVE["p"]])
    legal_34 = legal_34.at[22].set(legal_34[22] | mask[Tile.RED_FIVE["s"]])
    # Also fold TSUMOGIRI into last_draw's tile type if legal
    last_draw = obs["last_draw"].astype(jnp.int32)
    last_draw_type = Tile.to_tile_type(jnp.where(last_draw >= 0, last_draw, 0))
    legal_34 = jnp.where(
        mask[Action.TSUMOGIRI] & (last_draw >= 0),
        legal_34.at[last_draw_type].set(True),
        legal_34,
    )
    legal_discard_plane = legal_34.astype(jnp.float32)[None, :]  # (1, 34)

    v2_planes = jnp.concatenate(
        [
            lean_planes,           # 20
            river_recency_planes,  # 4
            opp_tedashi_planes,    # 3
            ds_norm,               # 3
            target_plane,          # 1
            suji_planes,           # 3
            kabe_plane,            # 1
            legal_discard_plane,   # 1
        ],
        axis=0,
    )  # (36, 34)
    return {"planes": v2_planes, "scalars": scalars}


def encode_obs_2d(state: State, mode: Literal["lean", "v2"] = "v2") -> Dict[str, jnp.ndarray]:
    """Dispatch between `obs_lean` (20x34) and `obs_v2` (36x34)."""
    if mode == "lean":
        return encode_obs_lean(state)
    if mode == "v2":
        return encode_obs_v2(state)
    raise ValueError(f"Unsupported obs_mode: {mode!r}. Expected 'lean' or 'v2'.")

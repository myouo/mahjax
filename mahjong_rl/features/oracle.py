"""Oracle privileged feature encoder (26x34) for Birkhoff History-State Oracle Critic.

HIDDEN INFORMATION — strictly for Oracle Critic training, NEVER fed into Belief or Policy nets.
Includes opponents' concealed hands, red fives, remaining live wall, ura-dora (isolated here!),
opponents' true wait masks, and opponents' exact furiten flags.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from mahjax.common.constants import NUM_TILE_TYPES_WITH_RED, RED_FIVE_TILE_TYPES
from mahjax.common.tile import Tile
from mahjax.mahjong_4p.constants import NUM_PLAYERS
from mahjax.mahjong_4p.state import State
from mahjong_rl.features.obs_2d import (
    NUM_TILES,
    _dora_count_plane,
    _scatter_add_34,
    _thermometer_4,
    clean_auto_reset_state,
)
from mahjong_rl.features.obs_v6 import encode_obs_v6

NUM_ORACLE_PLANES = 26


def encode_oracle_planes(state: State) -> jnp.ndarray:
    """Encode 26x34 privileged planes in the current player's ego-centric seat frame.

    Channels (26 total):
    - 0..11  (12): Opponents' concealed hands (right, across, left), 4 thermometer planes each.
    - 12..14 (3):  Opponents' red fives (5m/5p/5s).
    - 15..18 (4):  Remaining live wall tile counts (thermometer 1..4).
    - 19     (1):  Ura-dora actual dora tile counts (strictly isolated in Oracle!).
    - 20..22 (3):  Opponents' true winning tile wait masks (`state.players.can_win[others]`).
    - 23..25 (3):  Opponents' furiten-filtered legal wait masks (`can_win & ~furiten`).
    """
    clean_state = clean_auto_reset_state(state)
    c_p = clean_state.current_player.astype(jnp.int32)
    others = (jnp.arange(1, NUM_PLAYERS, dtype=jnp.int32) + c_p) % NUM_PLAYERS  # (3,)

    # 1. Opponents' concealed hands (3, 4, 34) -> (12, 34)
    others_hand = clean_state.players.hand[others].astype(jnp.int32)
    others_hand_planes = _thermometer_4(others_hand).reshape(12, NUM_TILES)

    # 2. Opponents' red fives (3, 34)
    others_red_flags = (
        clean_state.players.hand_with_red[others][:, NUM_TILES:NUM_TILE_TYPES_WITH_RED] > 0
    ).astype(jnp.float32)
    others_is_red = (
        jnp.zeros((3, NUM_TILES), dtype=jnp.float32)
        .at[:, jnp.asarray(RED_FIVE_TILE_TYPES)]
        .set(others_red_flags)
    )

    # 3. Remaining live wall counts (4, 34)
    deck = clean_state.round_state.deck.astype(jnp.int32)
    idxs = jnp.arange(deck.shape[0], dtype=jnp.int32)
    live_mask = (idxs >= clean_state.round_state.last_deck_ix.astype(jnp.int32)) & (
        idxs <= clean_state.round_state.next_deck_ix.astype(jnp.int32)
    )
    deck_types = Tile.to_tile_type(jnp.where(live_mask, deck, 0))
    live_counts = _scatter_add_34(
        deck_types, jnp.ones_like(deck_types, dtype=jnp.float32), live_mask
    ).astype(jnp.int32)
    wall_planes = _thermometer_4(live_counts)

    # 4. Ura-dora actual dora counts (1, 34) — NEVER exposed in obs_v6!
    ura_plane = _dora_count_plane(clean_state.round_state.ura_dora_indicators)[None, :]

    # 5. Opponents' true wait tables (3, 34) & non-furiten wait tables (3, 34)
    others_waits = clean_state.players.can_win[others].astype(jnp.bool_)
    others_furiten = (
        clean_state.players.furiten_by_discard[others]
        | clean_state.players.furiten_by_pass[others]
    ).astype(jnp.bool_)
    others_non_furiten_waits = (others_waits & ~others_furiten[:, None]).astype(jnp.float32)

    return jnp.concatenate(
        [
            others_hand_planes,        # 12
            others_is_red,             # 3
            wall_planes,               # 4
            ura_plane,                 # 1
            others_waits.astype(jnp.float32),  # 3
            others_non_furiten_waits,  # 3
        ],
        axis=0,
    )  # (26, 34)


def encode_full_v6_with_oracle(state: State) -> Dict[str, jnp.ndarray]:
    """Encode public V6 observation plus privileged Oracle planes."""
    out = encode_obs_v6(state)
    out["oracle_planes"] = encode_oracle_planes(state)
    return out

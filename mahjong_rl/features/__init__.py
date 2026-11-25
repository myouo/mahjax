"""Sovereign-Mahjong V6 observation & Oracle feature encoders."""

from mahjong_rl.features.obs_2d import (
    GLOBAL_SCALAR_DIM,
    NUM_LEAN_PLANES,
    NUM_TILES,
    NUM_V2_PLANES,
    clean_auto_reset_state,
    encode_obs_2d,
)
from mahjong_rl.features.obs_v6 import (
    GLOBAL_SCALAR_DIM_V6,
    NUM_OBS_V6_PLANES,
    NUM_POLICY_INPUT_PLANES,
    NUM_PUBLIC_PLANES_V6,
    NUM_TUO_PLANES,
    RIVER_LEN,
    RIVER_TOKEN_DIM,
    compute_topological_ukeire_planes,
    encode_obs_v6,
)
from mahjong_rl.features.oracle import (
    NUM_ORACLE_PLANES,
    encode_full_v6_with_oracle,
    encode_oracle_planes,
)

__all__ = [
    "GLOBAL_SCALAR_DIM",
    "GLOBAL_SCALAR_DIM_V6",
    "NUM_LEAN_PLANES",
    "NUM_OBS_V6_PLANES",
    "NUM_ORACLE_PLANES",
    "NUM_POLICY_INPUT_PLANES",
    "NUM_PUBLIC_PLANES_V6",
    "NUM_TILES",
    "NUM_TUO_PLANES",
    "NUM_V2_PLANES",
    "RIVER_LEN",
    "RIVER_TOKEN_DIM",
    "clean_auto_reset_state",
    "compute_topological_ukeire_planes",
    "encode_full_v6_with_oracle",
    "encode_obs_2d",
    "encode_obs_v6",
    "encode_oracle_planes",
]

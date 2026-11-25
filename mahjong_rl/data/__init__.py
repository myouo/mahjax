"""Streaming sharded dataset pipeline (mjai replay ingestion + 36x symmetry augmentation)."""

from mahjong_rl.data.hf_sync import (
    clean_tmp_and_cache,
    download_checkpoints_from_hf,
    download_v6_shards_from_hf,
    inspect_kaggle_resources,
    inspect_v6_shards,
    resolve_hf_token,
    upload_checkpoints_to_hf,
    upload_v6_shards_to_hf,
    verify_hf_auth,
)
from mahjong_rl.data.mjai_dataset import (
    PERMS_3,
    SYMMETRY_PERMS_36,
    build_shards_from_mjai_files,
    build_shards_from_policy_rollout,
    build_tile_and_action_perm_36,
    permute_suits_sample,
    permute_v6_batch,
)
from mahjong_rl.data.mjai_parquet_converter import (
    ParsedRound,
    build_gpu_batch_replayer,
    convert_mjai_parquets_to_v6_shards,
    parse_mjai_game_to_rounds,
)
from mahjong_rl.data.shard_loader import ShardBatchLoader

__all__ = [
    "PERMS_3",
    "ParsedRound",
    "SYMMETRY_PERMS_36",
    "ShardBatchLoader",
    "build_gpu_batch_replayer",
    "build_shards_from_mjai_files",
    "build_shards_from_policy_rollout",
    "build_tile_and_action_perm_36",
    "clean_tmp_and_cache",
    "convert_mjai_parquets_to_v6_shards",
    "download_checkpoints_from_hf",
    "download_v6_shards_from_hf",
    "inspect_kaggle_resources",
    "inspect_v6_shards",
    "parse_mjai_game_to_rounds",
    "permute_suits_sample",
    "permute_v6_batch",
    "resolve_hf_token",
    "upload_checkpoints_to_hf",
    "upload_v6_shards_to_hf",
    "verify_hf_auth",
]

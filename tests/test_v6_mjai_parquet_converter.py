"""Unit tests for the Two-Stage External MJAI Parquet -> V6 Shard Converter."""

from __future__ import annotations

from pathlib import Path
import pytest

from mahjong_rl.data.mjai_parquet_converter import (
    convert_mjai_parquets_to_v6_shards,
    parse_mjai_game_to_rounds,
)
from mahjong_rl.data.shard_loader import ShardBatchLoader

_SAMPLE_PARQUET = Path("/home/myo/Desktop/mahjax/data/tenhou_2026_mjai/tenhou-new-00000.parquet")


@pytest.mark.skipif(not _SAMPLE_PARQUET.exists(), reason="2026 Tenhou MJAI parquet not downloaded")
def test_mjai_parquet_converter_zero_desync_and_loader_roundtrip(tmp_path):
    """Verify exact 100% clean GPU replay, zero ron overflows, and float32 ShardBatchLoader roundtrip."""
    res = convert_mjai_parquets_to_v6_shards(
        [_SAMPLE_PARQUET],
        out_dir=tmp_path / "shards",
        max_games=5,
        round_batch_size=16,
        ron_k_max=2048,
        shard_size=20_000,
        storage_dtype="float16",
        compress=True,
        verbose=False,
    )
    assert res["games_processed"] == 5
    assert res["rounds_clean"] > 0
    assert res["rounds_desync"] == 0
    assert res["ron_overflow_batches"] == 0
    assert res["pass_steps"] > 0
    assert len(res["shards_written"]) >= 1

    loader = ShardBatchLoader(res["shards_written"], batch_size=32, shuffle=False, suit_augment=True)
    batch = next(iter(loader))
    assert batch["planes"].shape == (32, 53, 34) and batch["planes"].dtype == "float32"
    assert batch["oracle_planes"].shape == (32, 26, 34) and batch["oracle_planes"].dtype == "float32"
    assert batch["river_tokens"].shape == (32, 4, 24, 23) and batch["river_tokens"].dtype == "float32"
    assert batch["action_mask"].shape == (32, 87)
    assert batch["final_rank_matrix_4x4"].shape == (32, 4, 4)
    assert batch["y_ron"].shape == (32, 3, 34)

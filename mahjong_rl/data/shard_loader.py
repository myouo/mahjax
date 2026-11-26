"""Streaming shard loader for V6 Belief, GRP, and BC training with optional 36x augmentation."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from mahjong_rl.data.mjai_dataset import permute_v6_batch


class ShardBatchLoader:
    """Iterates over V6 `.npz` shards one at a time to keep host RAM bounded.

    When `prefetch > 0` (default `2`), runs `.npz` decompression, `float16 -> float32` upcast,
    and 36x `suit_augment` in a lightweight background thread so GPU training never stalls on CPU I/O.
    """

    def __init__(
        self,
        shard_paths: Sequence[Path | str],
        batch_size: int = 512,
        shuffle: bool = True,
        suit_augment: bool = False,
        drop_last: bool = False,
        keys: Optional[Sequence[str]] = None,
        seed: int = 0,
        prefetch: int = 2,
    ):
        self.shard_paths: List[Path] = [Path(p) for p in shard_paths]
        if not self.shard_paths:
            raise ValueError("ShardBatchLoader requires at least one shard path")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.suit_augment = bool(suit_augment)
        self.drop_last = bool(drop_last)
        self.keys: Optional[Tuple[str, ...]] = tuple(keys) if keys is not None else None
        self.prefetch = max(0, int(prefetch))
        self._rng = np.random.default_rng(seed)

    def _iter_sync(self) -> Iterator[Dict[str, np.ndarray]]:
        order = np.arange(len(self.shard_paths))
        if self.shuffle:
            self._rng.shuffle(order)

        for shard_i in order:
            with np.load(self.shard_paths[int(shard_i)]) as data:
                load_keys = [k for k in self.keys if k in data.files] if self.keys is not None else data.files
                arrays = {k: data[k] for k in load_keys}

            first_key = "action" if "action" in arrays else next(iter(arrays))
            n = arrays[first_key].shape[0]
            idxs = np.arange(n)
            if self.shuffle:
                self._rng.shuffle(idxs)

            for start in range(0, n, self.batch_size):
                batch_idx = idxs[start : start + self.batch_size]
                if batch_idx.size == 0:
                    continue
                if self.drop_last and n >= self.batch_size and batch_idx.size < self.batch_size:
                    continue
                batch = {k: v[batch_idx] for k, v in arrays.items()}
                # Upcast lossless float16 storage planes back to float32 on batch yield
                for fk in ("planes", "oracle_planes", "river_tokens"):
                    if fk in batch and batch[fk].dtype == np.float16:
                        batch[fk] = batch[fk].astype(np.float32)
                if self.suit_augment:
                    # Sample uniformly from all 36 elements of S_3^suit x S_3^sangen
                    perm_idx = int(self._rng.integers(0, 36))
                    batch = permute_v6_batch(batch, perm_idx=perm_idx)
                yield batch

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.prefetch <= 0:
            yield from self._iter_sync()
            return

        q: "queue.Queue[Optional[Dict[str, np.ndarray]]]" = queue.Queue(maxsize=self.prefetch)
        stop_evt = threading.Event()
        err: List[BaseException] = []

        def _worker() -> None:
            try:
                for batch in self._iter_sync():
                    while not stop_evt.is_set():
                        try:
                            q.put(batch, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if stop_evt.is_set():
                        break
            except BaseException as exc:
                err.append(exc)
            finally:
                while True:
                    try:
                        q.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        if stop_evt.is_set():
                            break

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
            if err:
                raise err[0]
        finally:
            stop_evt.set()
            # Drain at most one element to unblock producer if waiting on q.put
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            th.join(timeout=2.0)

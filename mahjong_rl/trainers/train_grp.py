"""Phase 0-B: Train Birkhoff Doubly-Stochastic (4, 4) Global Reward Predictor (`GRP_base`).

Trains `GRPNet` with 4-seat placement cross-entropy on complete-hanchan shards.
Reports Brier score, ECE, and Birkhoff polytope row/col conservation error.
After Phase 0-B, `GRP_base` is FROZEN as the DAG Fire-Wall for all input-side analytical priors.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mahjong_rl.algo.sovereign_v6 import (
    compute_birkhoff_error,
    compute_brier_4rank,
    compute_ece_4rank,
    make_weight_decay_mask,
)
from mahjong_rl.config import GRPConfig
from mahjong_rl.data.shard_loader import ShardBatchLoader
from mahjong_rl.models.grp import GRP_FEATURE_DIM, GRPNet


def train_grp(
    shard_paths: Sequence[Path | str],
    cfg: GRPConfig = GRPConfig(),
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[GRPNet, Any, List[Dict[str, float]]]:
    """Train `GRPNet` to predict Birkhoff `(4, 4)` final hanchan placement distributions."""
    model = GRPNet(config=cfg)
    rng = jax.random.PRNGKey(seed)
    dummy_x = jnp.zeros((2, GRP_FEATURE_DIM), dtype=jnp.float32)
    params = model.init(rng, dummy_x)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.lr, weight_decay=1e-4, mask=make_weight_decay_mask),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, grp_feat, target_4x4):
        def loss_fn(p):
            probs = model.apply(p, grp_feat)  # (B, 4, 4)
            log_p = jnp.log(jnp.clip(probs, 1e-7, 1.0))
            ce = -jnp.sum(target_4x4 * log_p, axis=-1)  # (B, 4)
            loss = jnp.mean(ce)
            return loss, probs

        (loss, probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        birkhoff_err = compute_birkhoff_error(probs)
        brier = jnp.mean(jnp.sum((probs - target_4x4) ** 2, axis=-1))
        return params, opt_state, loss, brier, birkhoff_err, probs

    if verbose:
        print(
            f"[Phase 0-B GRP] shards={len(shard_paths)} | bs={cfg.batch_size} | epochs={cfg.num_epochs}",
            flush=True,
        )
    t_all = time.perf_counter()
    history: List[Dict[str, float]] = []
    for epoch in range(cfg.num_epochs):
        t_ep = time.perf_counter()
        loader = ShardBatchLoader(
            shard_paths,
            batch_size=cfg.batch_size,
            shuffle=True,
            suit_augment=False,
            drop_last=True,
            keys=("grp_features", "final_rank_matrix_4x4"),
            seed=seed + epoch,
        )
        loss_acc = jnp.array(0.0, dtype=jnp.float32)
        brier_acc = jnp.array(0.0, dtype=jnp.float32)
        max_b_err = jnp.array(0.0, dtype=jnp.float32)
        n_batches = 0
        last_probs = None
        last_tgt = None
        for batch in loader:
            grp_f = jnp.asarray(batch["grp_features"], dtype=jnp.float32)
            tgt = jnp.asarray(batch["final_rank_matrix_4x4"], dtype=jnp.float32)
            params, opt_state, loss, brier, b_err, probs = train_step(params, opt_state, grp_f, tgt)
            loss_acc = loss_acc + loss
            brier_acc = brier_acc + brier
            max_b_err = jnp.maximum(max_b_err, b_err)
            last_probs = probs
            last_tgt = batch["final_rank_matrix_4x4"]
            n_batches += 1

        ece = (
            compute_ece_4rank(np.asarray(jax.device_get(last_probs)), last_tgt)
            if last_probs is not None
            else 0.0
        )
        denom = max(n_batches, 1)
        row = {
            "epoch": float(epoch),
            "grp_ce": float(loss_acc) / denom,
            "grp_brier": float(brier_acc) / denom,
            "grp_ece": float(ece),
            "grp_birkhoff_err": float(max_b_err),
        }
        history.append(row)
        if verbose:
            ep_s = time.perf_counter() - t_ep
            rem_s = ((time.perf_counter() - t_all) / (epoch + 1)) * (cfg.num_epochs - epoch - 1)
            print(
                f"[Phase 0-B GRP] ep {epoch + 1}/{cfg.num_epochs} ({n_batches} batches, {ep_s:.1f}s) | "
                f"ce={row['grp_ce']:.4f} brier={row['grp_brier']:.4f} ece={row['grp_ece']:.4f} "
                f"birkhoff={row['grp_birkhoff_err']:.2e} | ETA={rem_s/60:.1f}m",
                flush=True,
            )

    return model, params, history

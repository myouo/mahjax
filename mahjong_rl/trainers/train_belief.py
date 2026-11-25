"""Phase 0-A: Offline Oracle-Supervised Belief Model (`B_fast`) + Post-Training Temperature Scaling.

Supports automatic multi-GPU Data-Parallel training (`jax.pmap` + `lax.pmean` across all
`jax.local_device_count()` GPUs, e.g., Kaggle T4x2) with zero config changes!
Trains the 36x Equivariant `BeliefNet` with strictly UNWEIGHTED standard BCE + prior log-odds
bias init (`-4.184`) and 20% unbiased opponent utility dropout (`W_PHOENIX_7D, is_known=0`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mahjong_rl.algo.sovereign_v6 import (
    compute_belief_loss_v6,
    compute_binary_roc_auc,
    fit_belief_temperature,
    make_weight_decay_mask,
    replicate_for_pmap,
    shard_for_pmap,
)
from mahjong_rl.algo.utility import build_opponent_utility_features, rotate_utility_matrix_ego
from mahjong_rl.config import BeliefConfig
from mahjong_rl.data.shard_loader import ShardBatchLoader
from mahjong_rl.features.obs_v6 import (
    GLOBAL_SCALAR_DIM_V6,
    NUM_OBS_V6_PLANES,
    RIVER_LEN,
    RIVER_TOKEN_DIM,
)
from mahjong_rl.models.belief import BeliefNet


def _init_belief_params(model: BeliefNet, rng: jax.Array) -> Any:
    return model.init(
        rng,
        planes=jnp.zeros((2, NUM_OBS_V6_PLANES, 34), dtype=jnp.float32),
        scalars=jnp.zeros((2, GLOBAL_SCALAR_DIM_V6), dtype=jnp.float32),
        river_tokens=jnp.zeros((2, 4, RIVER_LEN, RIVER_TOKEN_DIM), dtype=jnp.float32),
        river_tile_ids=jnp.full((2, 4, RIVER_LEN), -1, dtype=jnp.int32),
        river_valid_mask=jnp.zeros((2, 4, RIVER_LEN), dtype=jnp.bool_),
        w_opp_5d=jnp.zeros((2, 3, 5), dtype=jnp.float32),
    )


def _extract_w_opp_5d(w_table_abs, cp, rng_key, dropout_prob: float) -> jnp.ndarray:
    w_ego_4x4 = jax.vmap(rotate_utility_matrix_ego)(w_table_abs, cp)
    w_opp = w_ego_4x4[:, 1:, :]
    is_known = jnp.ones((w_opp.shape[0], 3), dtype=jnp.float32)
    return build_opponent_utility_features(w_opp, is_known, rng_key=rng_key, dropout_prob=dropout_prob)


def _replicate(tree: Any, n_devices: int) -> Any:
    return replicate_for_pmap(tree, n_devices)


def _unreplicate(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _shard_batch(batch_np: Dict[str, np.ndarray], keys: Sequence[str], n_devices: int) -> Dict[str, Any]:
    out = {}
    for k in keys:
        a = batch_np[k]
        per_dev = a.shape[0] // n_devices
        out[k] = a[: per_dev * n_devices].reshape((n_devices, per_dev) + a.shape[1:])
    return shard_for_pmap(out, n_devices)


def train_belief(
    shard_paths: Sequence[Path | str],
    cfg: BeliefConfig = BeliefConfig(),
    val_shard_paths: Sequence[Path | str] | None = None,
    suit_augment: bool = True,
    multi_gpu: bool = True,
) -> Tuple[BeliefNet, Any, float, List[Dict[str, float]]]:
    """Train Phase 0-A `BeliefNet` (auto multi-GPU `pmap` if >1 GPU) and return `(model, params, T_cal, history)`."""
    model = BeliefNet(config=cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_init = jax.random.split(rng)
    params = _init_belief_params(model, k_init)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay, mask=make_weight_decay_mask),
    )
    opt_state = optimizer.init(params)

    n_devices = jax.local_device_count() if multi_gpu else 1
    use_pmap = n_devices > 1 and (cfg.batch_size % n_devices == 0)

    def _loss_and_grads(p, batch, step_key):
        w_opp_5d = _extract_w_opp_5d(batch["w_table_abs"], batch["current_player"], step_key, cfg.utility_dropout)

        def loss_fn(p_in):
            out = model.apply(
                p_in,
                planes=batch["planes"],
                scalars=batch["scalars"],
                river_tokens=batch["river_tokens"],
                river_tile_ids=batch["river_tile_ids"],
                river_valid_mask=batch["river_valid_mask"],
                w_opp_5d=w_opp_5d,
            )
            return compute_belief_loss_v6(
                out,
                y_ron=batch["y_ron"],
                y_score_bin=batch["y_score_bin"],
                y_call_threat=batch["y_call_threat"],
                y_tenpai=batch["y_tenpai"],
                cfg=cfg,
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        return grads, metrics

    if use_pmap:
        def _pmap_step(p, o, batch, step_key):
            grads, metrics = _loss_and_grads(p, batch, step_key)
            grads = jax.lax.pmean(grads, axis_name="devices")
            metrics = jax.lax.pmean(metrics, axis_name="devices")
            updates, o = optimizer.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return p, o, metrics

        train_step_pmap = jax.pmap(_pmap_step, axis_name="devices")
        params_rep = _replicate(params, n_devices)
        opt_rep = _replicate(opt_state, n_devices)
    else:
        @jax.jit
        def train_step_jit(p, o, batch, step_key):
            grads, metrics = _loss_and_grads(p, batch, step_key)
            updates, o = optimizer.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return p, o, metrics

    @jax.jit
    def eval_logits_step(p, batch):
        w_opp_5d = _extract_w_opp_5d(batch["w_table_abs"], batch["current_player"], None, 0.0)
        out = model.apply(
            p,
            planes=batch["planes"],
            scalars=batch["scalars"],
            river_tokens=batch["river_tokens"],
            river_tile_ids=batch["river_tile_ids"],
            river_valid_mask=batch["river_valid_mask"],
            w_opp_5d=w_opp_5d,
        )
        return out["deal_logits"], out["p_deal"]

    bel_keys = (
        "planes",
        "scalars",
        "river_tokens",
        "river_tile_ids",
        "river_valid_mask",
        "w_table_abs",
        "current_player",
        "y_ron",
        "y_score_bin",
        "y_call_threat",
        "y_tenpai",
    )
    history: List[Dict[str, float]] = []
    for epoch in range(cfg.num_epochs):
        loader = ShardBatchLoader(
            shard_paths,
            batch_size=cfg.batch_size,
            shuffle=True,
            suit_augment=suit_augment,
            drop_last=True,
            keys=bel_keys,
            seed=cfg.seed + epoch,
        )
        ep_acc: Dict[str, jnp.ndarray] | None = None
        n_steps = 0
        for batch_np in loader:
            rng, k_step = jax.random.split(rng)
            if use_pmap:
                b_sharded = _shard_batch(batch_np, bel_keys, n_devices)
                dev_keys = shard_for_pmap(jax.random.split(k_step, n_devices), n_devices)
                params_rep, opt_rep, metrics = train_step_pmap(params_rep, opt_rep, b_sharded, dev_keys)
                m_single = jax.tree_util.tree_map(lambda x: x[0], metrics)
            else:
                batch = {k: jnp.asarray(batch_np[k]) for k in bel_keys}
                params, opt_state, m_single = train_step_jit(params, opt_state, batch, k_step)

            if ep_acc is None:
                ep_acc = m_single
            else:
                ep_acc = jax.tree_util.tree_map(lambda a, b: a + b, ep_acc, m_single)
            n_steps += 1

        summary = {"epoch": float(epoch)}
        if ep_acc is not None and n_steps > 0:
            ep_host = jax.device_get(ep_acc)
            for mk, mv in ep_host.items():
                summary[mk] = float(mv) / n_steps
        history.append(summary)

    if use_pmap:
        params = jax.device_get(_unreplicate(params_rep))
        params = jax.device_put(params, jax.local_devices()[0])

    # Post-training Temperature Scaling calibration on held-out validation shards
    cal_shards = val_shard_paths if val_shard_paths else shard_paths
    cal_keys = (
        "planes",
        "scalars",
        "river_tokens",
        "river_tile_ids",
        "river_valid_mask",
        "w_table_abs",
        "current_player",
        "y_ron",
    )
    cal_loader = ShardBatchLoader(
        cal_shards,
        batch_size=cfg.batch_size,
        shuffle=False,
        suit_augment=False,
        drop_last=True,
        keys=cal_keys,
    )
    all_logits, all_labels = [], []
    for batch_np in cal_loader:
        batch = {k: jnp.asarray(batch_np[k]) for k in cal_keys if k != "y_ron"}
        d_logits, _ = eval_logits_step(params, batch)
        all_logits.append(np.asarray(jax.device_get(d_logits), dtype=np.float32))
        all_labels.append(np.asarray(batch_np["y_ron"], dtype=np.float32))

    if all_logits:
        cat_logits = np.concatenate(all_logits, axis=0)
        cat_labels = np.concatenate(all_labels, axis=0)
        t_cal = fit_belief_temperature(cat_logits, cat_labels)
        cal_probs = 1.0 / (1.0 + np.exp(-np.clip(cat_logits / t_cal, -30.0, 30.0)))
        val_auc = compute_binary_roc_auc(cat_labels, cal_probs)
        if history:
            history[-1]["val_t_cal"] = float(t_cal)
            history[-1]["val_roc_auc"] = float(val_auc)
    else:
        t_cal = 1.0

    return model, params, t_cal, history

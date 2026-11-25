"""Phase 3: Offline Behavioral-Space Self-Distillation (`SD-TAP`).

Distills the behavioral ensemble of the last `K` outer-epoch R-NaD snapshots:
    p_ens(a | s) = (1 / K) * sum_{k=1}^K pi_k(a | s)
into a SINGLE deployable `SovereignPolicyNet` (`params_deploy`) by minimizing exact
forward KL `KL(p_ens || pi_deploy)`, achieving Nash-equilibrium time-averaging without
weight-space mode collapse and with ZERO extra deployment latency!
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mahjong_rl.algo.sovereign_v6 import (
    make_weight_decay_mask,
    masked_categorical_log_probs,
    masked_categorical_probs,
    masked_kl_divergence,
    masked_mean,
)
from mahjong_rl.config import ModelConfig
from mahjong_rl.data.shard_loader import ShardBatchLoader
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRPNet
from mahjong_rl.trainers.train_bc import _compute_batch_trinity


def distill_sdtap(
    shard_paths: Sequence[Path | str],
    snapshot_params_list: Sequence[Any],
    belief_model: BeliefNet,
    belief_ema_params: Any,
    grp_model: GRPNet,
    grp_base_params: Any,
    model_cfg: ModelConfig = ModelConfig(),
    *,
    belief_t_cal: float = 1.0,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    num_epochs: int = 2,
    seed: int = 0,
) -> Tuple[SovereignPolicyNet, Any, List[Dict[str, float]]]:
    """Distill the behavioral average of `snapshot_params_list` into a single `SovereignPolicyNet`."""
    if not snapshot_params_list:
        raise ValueError("distill_sdtap requires at least one policy snapshot")

    model = SovereignPolicyNet(config=model_cfg)
    # Warm-start student from the latest outer snapshot for instant convergence
    params_deploy = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), snapshot_params_list[-1])
    stacked_snaps = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *snapshot_params_list)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay, mask=make_weight_decay_mask),
    )
    opt_state = optimizer.init(params_deploy)

    @jax.jit
    def distill_step(p_student, o_state, batch):
        w_me, (du_ron_3opp, du_ron_87, du_agari_3, du_ryu, call_threat) = _compute_batch_trinity(
            belief_model,
            jax.lax.stop_gradient(belief_ema_params),
            belief_t_cal,
            grp_model,
            jax.lax.stop_gradient(grp_base_params),
            batch,
            rng_key=None,
            utility_dropout=0.0,
        )
        mask = batch["action_mask"].astype(jnp.bool_)
        active = jnp.sum(mask.astype(jnp.int32), axis=-1) >= 2

        pol_in = dict(
            planes=batch["planes"],
            delta_u_ron_3opp=jax.lax.stop_gradient(du_ron_3opp),
            call_threat_plane=jax.lax.stop_gradient(call_threat),
            scalars=batch["scalars"],
            river_tokens=batch["river_tokens"],
            river_tile_ids=batch["river_tile_ids"],
            river_valid_mask=batch["river_valid_mask"],
            last_draw_type=batch["last_draw_type"],
            is_last_draw_red=batch["is_last_draw_red"],
            target_type=batch["target_type"],
            target_is_red=batch["target_is_red"],
            w_ego=w_me,
            delta_u_agari_3=jax.lax.stop_gradient(du_agari_3),
            delta_u_ryuukyoku=jax.lax.stop_gradient(du_ryu),
            delta_u_ron_87=jax.lax.stop_gradient(du_ron_87),
            legal_action_mask=mask,
        )

        # Compute exact behavioral ensemble average p_ens = (1/K) sum_k softmax(logits_k)
        def eval_one_snap(p_snap):
            lg = model.apply(p_snap, **pol_in)
            return masked_categorical_probs(lg, mask)

        snap_probs = jax.vmap(eval_one_snap)(stacked_snaps)  # (K, B, 87)
        p_ens = jax.lax.stop_gradient(jnp.mean(snap_probs, axis=0))
        p_ens = jnp.where(mask, jnp.maximum(p_ens, 1e-12), 0.0)
        p_ens = p_ens / jnp.maximum(jnp.sum(p_ens, axis=-1, keepdims=True), 1e-12)
        log_p_ens = jnp.where(mask, jnp.log(jnp.maximum(p_ens, 1e-12)), -1e9)

        def loss_fn(p):
            s_logits = model.apply(p, **pol_in)
            log_s = masked_categorical_log_probs(s_logits, mask)
            kl = masked_kl_divergence(p_ens, log_p_ens, log_s, mask)
            loss = masked_mean(kl, active)
            return loss, {"sdtap_kl": loss}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p_student)
        updates, o_state = optimizer.update(grads, o_state, p_student)
        p_student = optax.apply_updates(p_student, updates)
        return p_student, o_state, metrics

    req_keys = (
        "planes",
        "scalars",
        "river_tokens",
        "river_tile_ids",
        "river_valid_mask",
        "last_draw_type",
        "is_last_draw_red",
        "target_type",
        "target_is_red",
        "action_mask",
        "current_player",
        "w_table_abs",
        "table_scores",
        "table_round_num",
        "table_round_limit",
        "table_honba",
        "table_kyotaku",
        "table_dealer",
        "table_init_wind",
        "table_wall_rem",
    )
    history: List[Dict[str, float]] = []
    for epoch in range(num_epochs):
        loader = ShardBatchLoader(
            shard_paths,
            batch_size=batch_size,
            shuffle=True,
            suit_augment=False,
            drop_last=True,
            keys=req_keys,
            seed=seed + epoch,
        )
        kls = []
        for batch_np in loader:
            batch = {k: jnp.asarray(batch_np[k]) for k in req_keys}
            params_deploy, opt_state, metrics = distill_step(params_deploy, opt_state, batch)
            kls.append(float(metrics["sdtap_kl"]))
        history.append({"epoch": float(epoch), "sdtap_kl": float(np.mean(kls)) if kls else 0.0})

    return model, params_deploy, history

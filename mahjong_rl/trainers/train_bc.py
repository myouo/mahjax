"""Phase 0-C: Strictly Isomorphic Behavior Cloning of `SovereignPolicyNet`.

Architecture is 100% IDENTICAL between Phase 0-C BC and Phase 2 RL:
- Frozen `B_EMA` (with calibrated `T_cal`) + frozen `GRP_base` compute exact analytical
  `Delta_U_ron`, `Delta_U_agari`, `Delta_U_ryuukyoku`, and `call_threat_plane` on-the-fly.
- `beta_risk` and `beta_offense` are trained end-to-end from step 0 with zero weight decay!
- Masked cross-entropy strictly skips forced moves (`num_legal <= 1`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mahjong_rl.algo.legal_ron import TableSnapshot, compute_analytical_trinity
from mahjong_rl.algo.sovereign_v6 import (
    make_weight_decay_mask,
    masked_categorical_log_probs,
    masked_mean,
)
from mahjong_rl.algo.utility import (
    W_PHOENIX_7D,
    build_opponent_utility_features,
    rotate_utility_matrix_ego,
)
from mahjong_rl.config import BCConfig
from mahjong_rl.data.shard_loader import ShardBatchLoader
from mahjong_rl.features.obs_v6 import (
    GLOBAL_SCALAR_DIM_V6,
    NUM_OBS_V6_PLANES,
    RIVER_LEN,
    RIVER_TOKEN_DIM,
)
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRPNet


def init_policy_params(model: SovereignPolicyNet, rng: jax.Array) -> Any:
    """Initialize `SovereignPolicyNet` with canonical V6 input shapes."""
    return model.init(
        rng,
        planes=jnp.zeros((2, NUM_OBS_V6_PLANES, 34), dtype=jnp.float32),
        delta_u_ron_3opp=jnp.zeros((2, 3, 34), dtype=jnp.float32),
        call_threat_plane=jnp.zeros((2, 34), dtype=jnp.float32),
        scalars=jnp.zeros((2, GLOBAL_SCALAR_DIM_V6), dtype=jnp.float32),
        river_tokens=jnp.zeros((2, 4, RIVER_LEN, RIVER_TOKEN_DIM), dtype=jnp.float32),
        river_tile_ids=jnp.full((2, 4, RIVER_LEN), -1, dtype=jnp.int32),
        river_valid_mask=jnp.zeros((2, 4, RIVER_LEN), dtype=jnp.bool_),
        last_draw_type=jnp.full((2,), -1, dtype=jnp.int32),
        is_last_draw_red=jnp.zeros((2,), dtype=jnp.bool_),
        target_type=jnp.full((2,), -1, dtype=jnp.int32),
        target_is_red=jnp.zeros((2,), dtype=jnp.bool_),
        w_ego=jnp.broadcast_to(W_PHOENIX_7D[None, :], (2, 4)),
        delta_u_agari_3=jnp.zeros((2, 3), dtype=jnp.float32),
        delta_u_ryuukyoku=jnp.zeros((2,), dtype=jnp.float32),
        delta_u_ron_87=jnp.zeros((2, 87), dtype=jnp.float32),
        legal_action_mask=jnp.ones((2, 87), dtype=jnp.bool_),
    )


def _compute_batch_trinity(
    belief_model: BeliefNet,
    belief_params: Any,
    belief_t_cal: float,
    grp_model: GRPNet,
    grp_params: Any,
    batch: Dict[str, jnp.ndarray],
    rng_key: Optional[jax.Array],
    utility_dropout: float,
):
    w_ego_4x4 = jax.vmap(rotate_utility_matrix_ego)(batch["w_table_abs"], batch["current_player"])
    w_me = w_ego_4x4[:, 0, :]     # (B, 4)
    w_opp = w_ego_4x4[:, 1:, :]   # (B, 3, 4)
    is_known = jnp.ones((w_opp.shape[0], 3), dtype=jnp.float32)
    w_opp_5d = build_opponent_utility_features(w_opp, is_known, rng_key=rng_key, dropout_prob=utility_dropout)

    b_out = belief_model.apply(
        belief_params,
        planes=batch["planes"],
        scalars=batch["scalars"],
        river_tokens=batch["river_tokens"],
        river_tile_ids=batch["river_tile_ids"],
        river_valid_mask=batch["river_valid_mask"],
        w_opp_5d=w_opp_5d,
        temperature_cal=belief_t_cal,
    )
    grp_fn = lambda gf: grp_model.apply(grp_params, gf)

    def _one_trinity(
        sc, rn, rl, hb, ky, dl, iw, wr, cp,
        p_d, p_sb, p_ct, p_t, wm, ld, ld_red
    ):
        tbl = TableSnapshot(
            scores=sc,
            round_num=rn,
            round_limit=rl,
            honba=hb,
            kyotaku=ky,
            dealer=dl,
            init_wind=iw,
            wall_rem=wr,
            current_player=cp,
        )
        return compute_analytical_trinity(
            s_table=tbl,
            p_deal_3opp=p_d,
            p_score_bins=p_sb,
            p_call_threat_3=p_ct,
            p_tenpai_3=p_t,
            w_me=wm,
            last_draw_type=ld,
            is_last_draw_red=ld_red,
            grp_fn=grp_fn,
        )

    trinity = jax.vmap(_one_trinity)(
        batch["table_scores"],
        batch["table_round_num"],
        batch["table_round_limit"],
        batch["table_honba"],
        batch["table_kyotaku"],
        batch["table_dealer"],
        batch["table_init_wind"],
        batch["table_wall_rem"],
        batch["current_player"],
        b_out["p_deal"],
        b_out["p_score_bin"],
        b_out["p_call_threat"],
        b_out["p_tenpai"],
        w_me,
        batch["last_draw_type"],
        batch["is_last_draw_red"],
    )
    return w_me, trinity


def train_bc(
    shard_paths: Sequence[Path | str],
    belief_model: BeliefNet,
    belief_params: Any,
    grp_model: GRPNet,
    grp_params: Any,
    cfg: BCConfig = BCConfig(),
    belief_t_cal: float = 1.0,
) -> Tuple[SovereignPolicyNet, Any, List[Dict[str, float]]]:
    """Train Phase 0-C strictly isomorphic `SovereignPolicyNet` via Behavior Cloning."""
    model = SovereignPolicyNet(config=cfg.model)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_init = jax.random.split(rng)
    params = init_policy_params(model, k_init)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay, mask=make_weight_decay_mask),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch, step_key):
        w_me, (du_ron_3opp, du_ron_87, du_agari_3, du_ryu, call_threat) = _compute_batch_trinity(
            belief_model,
            jax.lax.stop_gradient(belief_params),
            belief_t_cal,
            grp_model,
            jax.lax.stop_gradient(grp_params),
            batch,
            rng_key=step_key,
            utility_dropout=cfg.utility_dropout,
        )
        mask = batch["action_mask"].astype(jnp.bool_)
        actions = batch["action"].astype(jnp.int32)
        active = jnp.sum(mask.astype(jnp.int32), axis=-1) >= 2

        def loss_fn(p):
            logits, aux = model.apply(
                p,
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
                return_aux=True,
            )
            log_p = masked_categorical_log_probs(logits, mask)
            idx = jnp.arange(actions.shape[0], dtype=jnp.int32)
            nll = -log_p[idx, actions]
            loss = masked_mean(nll, active)
            preds = jnp.argmax(jnp.where(mask, logits, -1e9), axis=-1)
            acc = masked_mean((preds == actions).astype(jnp.float32), active)
            return loss, {
                "bc_loss": loss,
                "bc_acc": acc,
                "beta_risk": aux["beta_risk"],
                "beta_offense": aux["beta_offense"],
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    history: List[Dict[str, float]] = []
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
        "action",
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
    for epoch in range(cfg.num_epochs):
        loader = ShardBatchLoader(
            shard_paths,
            batch_size=cfg.batch_size,
            shuffle=True,
            suit_augment=cfg.suit_augment,
            drop_last=True,
            keys=req_keys,
            seed=cfg.seed + epoch,
        )
        ep_metrics: Dict[str, List[float]] = {}
        for batch_np in loader:
            rng, k_step = jax.random.split(rng)
            batch = {k: jnp.asarray(batch_np[k]) for k in req_keys}
            params, opt_state, metrics = train_step(params, opt_state, batch, k_step)
            for mk, mv in metrics.items():
                ep_metrics.setdefault(mk, []).append(float(mv))
        summary = {"epoch": float(epoch)}
        for mk, vals in ep_metrics.items():
            summary[mk] = float(np.mean(vals)) if vals else 0.0
        history.append(summary)

    return model, params, history

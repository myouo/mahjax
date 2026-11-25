"""P1 Birkhoff Doubly-Stochastic History-State Dual-Horizon Oracle Critic (`OracleCriticNet`).

Key architectural guarantees:
1. History-State Completeness: fuses `obs_v6` (53x34) + `oracle_planes` (26x34, incl. ura-dora &
   hidden hands/wall) + `stop_grad(river_spatial_feat)` reused at zero compute cost from `B_EMA`.
2. Zero-Sum Birkhoff Manifold: predicts ego-frame residual `delta_logits` (zero-initialized!)
   added to temperature-scaled `stop_grad(grp_logits_ego) / tau_prior`, projected via 3-step
   differentiable `log_sinkhorn_4x4`, and un-rotated back to absolute seat order `(4, 4)`.
3. Dual-Horizon Auxiliary Heads (training-only, zero inference cost):
   - `round_winner_logits (5,)`: ego-frame `[me, right, across, left, exhaustive_draw]`
   - `round_score_delta (4,)`: ego-frame expected score delta in 10k-pt units (`/ 100.0`).
"""

from __future__ import annotations

from typing import Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from mahjax.mahjong_4p.constants import NUM_PLAYERS
from mahjong_rl.config import OracleConfig
from mahjong_rl.models.equivariant import EquivariantMahjongBackbone
from mahjong_rl.models.grp import log_sinkhorn_4x4


def _rotate_4x4_abs_to_ego(mat_abs: jnp.ndarray, current_player: jnp.ndarray) -> jnp.ndarray:
    """Rotate `(B, 4, 4)` seat rows from absolute `[0,1,2,3]` to ego `[me, right, across, left]`."""
    def _one(m: jnp.ndarray, cp: jnp.ndarray) -> jnp.ndarray:
        order = (jnp.arange(NUM_PLAYERS, dtype=jnp.int32) + cp.astype(jnp.int32)) % NUM_PLAYERS
        return m[order]
    return jax.vmap(_one)(mat_abs, current_player)


def _rotate_4x4_ego_to_abs(mat_ego: jnp.ndarray, current_player: jnp.ndarray) -> jnp.ndarray:
    """Inverse rotation from ego `[me, right, across, left]` back to absolute `[0,1,2,3]`."""
    def _one(m: jnp.ndarray, cp: jnp.ndarray) -> jnp.ndarray:
        inv_order = (jnp.arange(NUM_PLAYERS, dtype=jnp.int32) - cp.astype(jnp.int32)) % NUM_PLAYERS
        return m[inv_order]
    return jax.vmap(_one)(mat_ego, current_player)


class OracleCriticNet(nn.Module):
    """Birkhoff Doubly-Stochastic History-State Dual-Horizon Oracle Critic."""

    config: OracleConfig = OracleConfig()
    tau_prior: float = 1.5
    river_dim: int = 24  # Fallback only if `river_spatial_feat` is not passed

    @nn.compact
    def __call__(
        self,
        planes: jnp.ndarray,                             # (..., 53, 34)
        oracle_planes: jnp.ndarray,                      # (..., 26, 34)
        scalars: jnp.ndarray,                            # (..., 56)
        w_table_ego: jnp.ndarray,                        # (..., 4, 4)
        grp_logits_abs: jnp.ndarray,                     # (..., 4, 4) from frozen GRP_base
        current_player: jnp.ndarray,                     # (...,) int32 absolute seat
        river_spatial_feat: Optional[jnp.ndarray] = None,  # (..., 34, 4*D_r) from B_EMA
        river_tokens: Optional[jnp.ndarray] = None,
        river_tile_ids: Optional[jnp.ndarray] = None,
        river_valid_mask: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        squeezed = planes.ndim == 2
        if squeezed:
            planes = planes[None, ...]
            oracle_planes = oracle_planes[None, ...]
            scalars = scalars[None, ...]
            w_table_ego = w_table_ego[None, ...]
            grp_logits_abs = grp_logits_abs[None, ...]
            current_player = current_player[None, ...]
            if river_spatial_feat is not None:
                river_spatial_feat = river_spatial_feat[None, ...]
            if river_tokens is not None:
                river_tokens = river_tokens[None, ...]
                river_tile_ids = river_tile_ids[None, ...]
                river_valid_mask = river_valid_mask[None, ...]

        bsz = planes.shape[0]
        full_planes = jnp.concatenate(
            [planes.astype(jnp.float32), oracle_planes.astype(jnp.float32)],
            axis=1,
        )  # (B, 79, 34)

        w_flat = w_table_ego.astype(jnp.float32).reshape((bsz, NUM_PLAYERS * NUM_PLAYERS))
        cond_vec = jnp.concatenate([scalars.astype(jnp.float32), w_flat], axis=-1)  # (B, 72)

        # Strictly stop-gradient on reused river spatial features so Belief & Oracle remain isolated!
        stopped_river = (
            jax.lax.stop_gradient(river_spatial_feat)
            if river_spatial_feat is not None
            else None
        )

        _, g, _ = EquivariantMahjongBackbone(
            num_blocks=self.config.num_blocks,
            channels=self.config.channels,
            river_dim=self.river_dim,
            attn_heads=self.config.attn_heads,
            se_reduction=4,
            global_hidden_dim=self.config.global_hidden_dim,
        )(
            planes=full_planes,
            cond_vec=cond_vec,
            river_tokens=river_tokens,
            river_tile_ids=river_tile_ids,
            river_valid_mask=river_valid_mask,
            precomputed_river_feat=stopped_river,
        )

        h = nn.gelu(nn.Dense(self.config.global_hidden_dim)(g))
        h = nn.LayerNorm()(h)

        # Zero-initialized residual over frozen GRP prior logits -> smooth Phase 1 warmup!
        delta_logits_ego = nn.Dense(
            NUM_PLAYERS * NUM_PLAYERS,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="birkhoff_delta_head",
        )(h).reshape((bsz, NUM_PLAYERS, NUM_PLAYERS))

        grp_ego = _rotate_4x4_abs_to_ego(
            jax.lax.stop_gradient(grp_logits_abs.astype(jnp.float32)),
            current_player,
        )
        combined_ego = grp_ego / float(self.tau_prior) + delta_logits_ego
        probs_ego = log_sinkhorn_4x4(combined_ego, num_iters=self.config.sinkhorn_iters)
        probs_abs = _rotate_4x4_ego_to_abs(probs_ego, current_player)

        # Dual-Horizon training-only auxiliary heads (ego frame)
        aux_h = nn.gelu(nn.Dense(self.config.global_hidden_dim // 2)(g))
        round_winner_logits = nn.Dense(NUM_PLAYERS + 1, name="round_winner_head")(aux_h)  # (B, 5)
        round_score_delta = nn.Dense(NUM_PLAYERS, name="round_score_delta_head")(aux_h)   # (B, 4)

        out = {
            "probs_4x4": probs_abs,                    # (B, 4, 4) absolute seat order
            "probs_ego_4x4": probs_ego,                # (B, 4, 4) ego seat order
            "logits_ego_4x4": combined_ego,
            "round_winner_logits": round_winner_logits,  # (B, 5) ego [0..3, draw]
            "round_score_delta": round_score_delta,      # (B, 4) ego score delta / 100
        }
        if squeezed:
            out = {k: v[0] for k, v in out.items()}
        return out

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

from mahjax.common.shanten import Shanten
from mahjax.mahjong_4p.constants import NUM_PLAYERS
from mahjong_rl.config import OracleConfig
from mahjong_rl.features.obs_v6 import compute_topological_ukeire_planes
from mahjong_rl.models.equivariant import EquivariantMahjongBackbone
from mahjong_rl.models.grp import log_sinkhorn_4x4


def _exact_shanten_4(hands_b4_34: jnp.ndarray) -> jnp.ndarray:
    """Compute exact 4-player shanten `(B, 4)` in ego order using cached MahJax Shanten lookup."""
    def _one(h: jnp.ndarray) -> jnp.ndarray:
        hi = jnp.clip(jnp.round(h), 0.0, 4.0).astype(jnp.int32)
        return jnp.where(jnp.sum(hi) >= 13, Shanten.number(hi), Shanten.normal(hi) - 1).astype(jnp.float32)

    return jax.vmap(jax.vmap(_one))(hands_b4_34)


def _rotate_4x4_abs_to_ego(mat_abs: jnp.ndarray, current_player: jnp.ndarray) -> jnp.ndarray:
    """Rotate `(B, 4, 4)` seat rows from absolute `[0,1,2,3]` to ego `[me, right, across, left]`."""
    def _one(m: jnp.ndarray, cp: jnp.ndarray) -> jnp.ndarray:
        order = (jnp.arange(NUM_PLAYERS, dtype=jnp.int32) + cp.astype(jnp.int32)) % NUM_PLAYERS
        return m[order]
    return jax.vmap(_one)(mat_abs, current_player)


def _rotate_ego_to_abs(arr_ego: jnp.ndarray, current_player: jnp.ndarray) -> jnp.ndarray:
    """Inverse rotation along seat axis 1 from ego `[me, right, across, left]` back to `[0,1,2,3]`."""
    def _one(m: jnp.ndarray, cp: jnp.ndarray) -> jnp.ndarray:
        inv_order = (jnp.arange(NUM_PLAYERS, dtype=jnp.int32) - cp.astype(jnp.int32)) % NUM_PLAYERS
        return m[inv_order]
    return jax.vmap(_one)(arr_ego, current_player)


class OracleCriticNet(nn.Module):
    """Birkhoff Doubly-Stochastic History-State Dual-Horizon Oracle Critic."""

    config: OracleConfig = OracleConfig()
    tau_prior: float = 1.0
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
        pub_f = planes.astype(jnp.float32)
        ora_f = oracle_planes.astype(jnp.float32)
        full_planes = jnp.concatenate([pub_f, ora_f], axis=1)  # (B, 79, 34)

        # Symmetric 36x-equivariant opponent tenpai / furiten / live-wait scalars derived over tile axis
        live_wall_cnt = jnp.sum(ora_f[:, 15:19, :], axis=1)             # (B, 34) in [0..4]
        opp_waits = (ora_f[:, 20:23, :] > 0.5).astype(jnp.float32)      # (B, 3, 34)
        opp_nf_waits = (ora_f[:, 23:26, :] > 0.5).astype(jnp.float32)   # (B, 3, 34)
        opp_is_tenpai = (jnp.sum(opp_waits, axis=-1) > 0.5).astype(jnp.float32)
        opp_has_nf = (jnp.sum(opp_nf_waits, axis=-1) > 0.5).astype(jnp.float32)
        opp_is_furiten = opp_is_tenpai * (1.0 - opp_has_nf)
        opp_wait_types = jnp.clip(jnp.sum(opp_waits, axis=-1) / 9.0, 0.0, 2.0)
        opp_live_waits = jnp.clip(jnp.sum(opp_waits * live_wall_cnt[:, None, :], axis=-1) / 16.0, 0.0, 2.0)
        opp_nf_live_waits = jnp.clip(jnp.sum(opp_nf_waits * live_wall_cnt[:, None, :], axis=-1) / 16.0, 0.0, 2.0)
        opp_tenpai_scalars = jnp.concatenate(
            [opp_is_tenpai, opp_is_furiten, opp_wait_types, opp_live_waits, opp_nf_live_waits],
            axis=-1,
        )  # (B, 15)

        # Symmetric 36x-invariant ego hand/tenpai/dora scalars & opponent dora/ura/red scalars
        vis_dora_plane = pub_f[:, 11, :]
        ura_dora_plane = ora_f[:, 19, :]
        my_hand_cnt = jnp.sum(pub_f[:, 0:4, :], axis=1)                 # (B, 34)
        my_vis_dora = jnp.clip(jnp.sum(my_hand_cnt * vis_dora_plane, axis=-1), 0.0, 2.0)
        my_ura_dora = jnp.clip(jnp.sum(my_hand_cnt * ura_dora_plane, axis=-1), 0.0, 2.0)
        my_red_cnt = jnp.clip(jnp.sum(pub_f[:, 4, :], axis=-1) / 3.0, 0.0, 1.5)
        my_max_live_waits = jnp.max(pub_f[:, 39, :], axis=-1)
        my_opt_live_waits = (
            jnp.sum(pub_f[:, 39, :] * pub_f[:, 38, :], axis=-1)
            / jnp.maximum(jnp.sum(pub_f[:, 38, :], axis=-1), 1.0)
        )
        my_min_shanten = jnp.min(pub_f[:, 35, :], axis=-1)
        my_is_tenpai = (my_max_live_waits > 1e-4).astype(jnp.float32)
        ego_hand_scalars = jnp.stack(
            [
                my_vis_dora,
                my_ura_dora,
                my_red_cnt,
                my_max_live_waits,
                my_opt_live_waits,
                my_min_shanten,
                my_is_tenpai,
            ],
            axis=-1,
        )  # (B, 7)

        opp_hand_cnt = jnp.sum(ora_f[:, 0:12, :].reshape((bsz, 3, 4, 34)), axis=2)  # (B, 3, 34)
        opp_vis_dora = jnp.clip(jnp.sum(opp_hand_cnt * vis_dora_plane[:, None, :], axis=-1), 0.0, 2.0)
        opp_ura_dora = jnp.clip(jnp.sum(opp_hand_cnt * ura_dora_plane[:, None, :], axis=-1), 0.0, 2.0)
        opp_red_cnt = jnp.clip(jnp.sum(ora_f[:, 12:15, :], axis=-1) / 3.0, 0.0, 1.5)
        opp_dora_scalars = jnp.concatenate([opp_vis_dora, opp_ura_dora, opp_red_cnt], axis=-1)  # (B, 9)

        # Exact 4-player shanten, 4-player TUO vs true live wall, & pairwise wait-overlap invariants
        hands_4 = jnp.concatenate([my_hand_cnt[:, None, :], opp_hand_cnt], axis=1)  # (B, 4, 34)
        sh4 = _exact_shanten_4(hands_4)  # (B, 4)
        tuo_4 = jax.vmap(jax.vmap(compute_topological_ukeire_planes, in_axes=(0, None)))(
            hands_4, live_wall_cnt
        )  # (B, 4, 3, 34)
        tuo_sum_4 = jnp.clip(jnp.sum(tuo_4, axis=-1).reshape((bsz, 12)) / 8.0, 0.0, 3.0)
        ego_holds_opp_wait = jnp.clip(
            jnp.sum((my_hand_cnt[:, None, :] > 0.5) * opp_nf_waits, axis=-1) / 4.0, 0.0, 2.0
        )  # (B, 3)
        all_hands_tot = jnp.sum(hands_4, axis=1)  # (B, 34)
        other_holds_opp_wait = jnp.clip(
            jnp.sum(((all_hands_tot[:, None, :] - opp_hand_cnt) > 0.5) * opp_nf_waits, axis=-1) / 8.0,
            0.0,
            2.0,
        )  # (B, 3)
        meld_cnt_4 = jnp.clip(jnp.sum(pub_f[:, 12:16, :], axis=-1) / 12.0, 0.0, 1.5)

        grp_ego = _rotate_4x4_abs_to_ego(
            jax.lax.stop_gradient(grp_logits_abs.astype(jnp.float32)),
            current_player,
        )
        grp_probs_ego = log_sinkhorn_4x4(grp_ego, num_iters=self.config.sinkhorn_iters)
        grp_probs_flat = grp_probs_ego.reshape((bsz, NUM_PLAYERS * NUM_PLAYERS))

        w_flat = w_table_ego.astype(jnp.float32).reshape((bsz, NUM_PLAYERS * NUM_PLAYERS))
        u_curr_ego = jnp.sum(grp_probs_ego * w_table_ego.astype(jnp.float32), axis=-1)  # (B, 4)
        cond_vec = jnp.concatenate(
            [
                scalars.astype(jnp.float32),
                w_flat,
                grp_probs_flat,
                u_curr_ego,
                opp_tenpai_scalars,
                ego_hand_scalars,
                opp_dora_scalars,
                sh4 / 6.0,
                (sh4 <= 0.5).astype(jnp.float32),
                (sh4 <= 1.5).astype(jnp.float32),
                (sh4 <= 2.5).astype(jnp.float32),
                (sh4[:, 0:1] - sh4[:, 1:]) / 6.0,
                tuo_sum_4,
                ego_holds_opp_wait,
                other_holds_opp_wait,
                meld_cnt_4,
            ],
            axis=-1,
        )  # (B, 164)

        # Strictly stop-gradient on reused river spatial features so Belief & Oracle remain isolated!
        stopped_river = (
            jax.lax.stop_gradient(river_spatial_feat)
            if river_spatial_feat is not None
            else None
        )

        _, g_sp, _ = EquivariantMahjongBackbone(
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

        tab_dim = int(self.config.tabular_hidden_dim)
        head_dim = int(self.config.head_hidden_dim)
        tab_h = nn.LayerNorm()(nn.gelu(nn.Dense(tab_dim)(cond_vec)))
        tab_h = nn.LayerNorm()(tab_h + nn.gelu(nn.Dense(tab_dim)(tab_h)))
        sp_resid = nn.Dense(
            tab_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="spatial_resid_proj",
        )(g_sp)
        h = nn.LayerNorm()(nn.gelu(nn.Dense(head_dim)(tab_h + sp_resid)))

        # Zero-initialized residual over frozen GRP prior logits -> smooth Phase 1 warmup!
        delta_logits_ego = nn.Dense(
            NUM_PLAYERS * NUM_PLAYERS,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="birkhoff_delta_head",
        )(h).reshape((bsz, NUM_PLAYERS, NUM_PLAYERS))

        combined_ego = grp_ego / float(self.tau_prior) + delta_logits_ego
        probs_ego = log_sinkhorn_4x4(combined_ego, num_iters=self.config.sinkhorn_iters)
        probs_abs = _rotate_ego_to_abs(probs_ego, current_player)

        # Direct zero-initialized intra-round utility-shift predictor in ego frame
        delta_u_ego = nn.Dense(
            NUM_PLAYERS,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="delta_u_head",
        )(h)  # (B, 4)
        delta_u_abs = _rotate_ego_to_abs(delta_u_ego, current_player)

        # Dual-Horizon training-only auxiliary heads (ego frame)
        aux_h = nn.gelu(nn.Dense(max(head_dim // 2, 32))(h))
        round_winner_logits = nn.Dense(NUM_PLAYERS + 1, name="round_winner_head")(aux_h)  # (B, 5)
        round_score_delta = nn.Dense(NUM_PLAYERS, name="round_score_delta_head")(aux_h)   # (B, 4)

        out = {
            "probs_4x4": probs_abs,                    # (B, 4, 4) absolute seat order
            "probs_ego_4x4": probs_ego,                # (B, 4, 4) ego seat order
            "logits_ego_4x4": combined_ego,
            "delta_u_ego": delta_u_ego,                # (B, 4) predicted E[U_target - U_grp_curr]
            "delta_u_abs": delta_u_abs,                # (B, 4) absolute seat order
            "round_winner_logits": round_winner_logits,  # (B, 5) ego [0..3, draw]
            "round_score_delta": round_score_delta,      # (B, 4) ego score delta / 100
        }
        if squeezed:
            out = {k: v[0] for k, v in out.items()}
        return out

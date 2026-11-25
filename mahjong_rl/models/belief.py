"""P0 Quasi-Static Belief Network (`B_fast` -> `B_EMA`) for Sovereign-Mahjong V6.

Predicts calibrated per-opponent deal-in probabilities, conditional score-bin distributions,
high-value meld-feed (call) threat, tenpai probabilities, and tsumo threat strictly from
zero-leakage public observations (`obs_v6`) and unbiased 5D opponent utility features.

Key invariants:
1. 36x Algebraically Equivariant (`S_3^suit x S_3^sangen`) via `EquivariantMahjongBackbone`
   and shared suit / shared honor tile-level heads.
2. Unweighted standard BCE with prior log-odds bias init (`-4.184137` = logit(0.015))
   guarantees zero probability inflation (`E[p_hat] == p_true`).
3. Exposes `river_spatial_feat` for zero-copy `stop_gradient` reuse by the History-State Oracle.
"""

from __future__ import annotations

from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp

from mahjong_rl.config import BeliefConfig
from mahjong_rl.models.equivariant import EquivariantMahjongBackbone

NUM_TILES = 34
NUM_OPPS = 3
NUM_SCORE_BINS = 3


class BeliefNet(nn.Module):
    """36x Equivariant Opponent Belief Model (`B_fast` and `B_EMA`)."""

    config: BeliefConfig = BeliefConfig()

    @nn.compact
    def __call__(
        self,
        planes: jnp.ndarray,              # (..., 53, 34)
        scalars: jnp.ndarray,             # (..., 56)
        river_tokens: jnp.ndarray,        # (..., 4, 24, 23)
        river_tile_ids: jnp.ndarray,      # (..., 4, 24)
        river_valid_mask: jnp.ndarray,    # (..., 4, 24)
        w_opp_5d: jnp.ndarray,            # (..., 3, 5)
        temperature_cal: float | jnp.ndarray = 1.0,
    ) -> Dict[str, jnp.ndarray]:
        squeezed = planes.ndim == 2
        if squeezed:
            planes = planes[None, ...]
            scalars = scalars[None, ...]
            river_tokens = river_tokens[None, ...]
            river_tile_ids = river_tile_ids[None, ...]
            river_valid_mask = river_valid_mask[None, ...]
            w_opp_5d = w_opp_5d[None, ...]

        bsz = planes.shape[0]
        w_opp_flat = w_opp_5d.astype(jnp.float32).reshape((bsz, NUM_OPPS * 5))
        cond_vec = jnp.concatenate([scalars.astype(jnp.float32), w_opp_flat], axis=-1)  # (B, 71)

        x_34, g, river_feat = EquivariantMahjongBackbone(
            num_blocks=self.config.num_blocks,
            channels=self.config.channels,
            river_dim=self.config.river_dim,
            attn_heads=self.config.attn_heads,
            se_reduction=4,
            global_hidden_dim=self.config.global_hidden_dim,
        )(
            planes=planes,
            cond_vec=cond_vec,
            river_tokens=river_tokens,
            river_tile_ids=river_tile_ids,
            river_valid_mask=river_valid_mask,
        )

        g_tile = jnp.broadcast_to(g[:, None, :], (bsz, NUM_TILES, g.shape[-1]))
        xg = jnp.concatenate([x_34, g_tile], axis=-1)  # (B, 34, C + G)
        xg_suits = xg[:, :27, :]
        xg_honors = xg[:, 27:34, :]

        # Shared suit & shared honor hidden projections (exact S_3^suit x S_3^sangen equivariance!)
        h_dim = self.config.global_hidden_dim
        suit_h = nn.gelu(nn.Dense(h_dim, name="suit_belief_proj")(xg_suits))
        honor_h = nn.gelu(nn.Dense(h_dim, name="honor_belief_proj")(xg_honors))

        # 1. Deal-in logits (B, 3, 34) with prior log-odds bias init (-4.184137)
        prior_bias = nn.initializers.constant(self.config.ron_prior_bias_init)
        deal_suits = nn.Dense(NUM_OPPS, bias_init=prior_bias, name="suit_deal_head")(suit_h)     # (B, 27, 3)
        deal_honors = nn.Dense(NUM_OPPS, bias_init=prior_bias, name="honor_deal_head")(honor_h)  # (B, 7, 3)
        deal_logits = jnp.transpose(jnp.concatenate([deal_suits, deal_honors], axis=1), (0, 2, 1))  # (B, 3, 34)

        # 2. Conditional score-bin logits (B, 3, 34, 3)
        sb_suits = nn.Dense(NUM_OPPS * NUM_SCORE_BINS, name="suit_score_head")(suit_h)
        sb_honors = nn.Dense(NUM_OPPS * NUM_SCORE_BINS, name="honor_score_head")(honor_h)
        sb_all = jnp.concatenate([sb_suits, sb_honors], axis=1).reshape((bsz, NUM_TILES, NUM_OPPS, NUM_SCORE_BINS))
        score_bin_logits = jnp.transpose(sb_all, (0, 2, 1, 3))  # (B, 3, 34, 3)

        # 3. High-value meld-feed (call) threat logits (B, 3, 34)
        call_bias = nn.initializers.constant(-3.0)
        ct_suits = nn.Dense(NUM_OPPS, bias_init=call_bias, name="suit_call_head")(suit_h)
        ct_honors = nn.Dense(NUM_OPPS, bias_init=call_bias, name="honor_call_head")(honor_h)
        call_threat_logits = jnp.transpose(jnp.concatenate([ct_suits, ct_honors], axis=1), (0, 2, 1))  # (B, 3, 34)

        # 4. Global opponent tenpai & tsumo threat heads (B, 3)
        g_h = nn.gelu(nn.Dense(h_dim)(g))
        tenpai_logits = nn.Dense(NUM_OPPS, bias_init=nn.initializers.constant(-1.5), name="tenpai_head")(g_h)
        tsumo_logits = nn.Dense(NUM_OPPS, bias_init=nn.initializers.constant(-3.5), name="tsumo_head")(g_h)

        t_cal = jnp.maximum(jnp.asarray(temperature_cal, dtype=jnp.float32), 0.1)
        p_deal = jax.nn.sigmoid(deal_logits / t_cal)
        p_score_bin = jax.nn.softmax(score_bin_logits, axis=-1)
        p_call_threat = jax.nn.sigmoid(call_threat_logits)
        p_tenpai = jax.nn.sigmoid(tenpai_logits)
        p_tsumo = jax.nn.sigmoid(tsumo_logits)

        out = {
            "deal_logits": deal_logits,
            "score_bin_logits": score_bin_logits,
            "call_threat_logits": call_threat_logits,
            "tenpai_logits": tenpai_logits,
            "tsumo_logits": tsumo_logits,
            "p_deal": p_deal,
            "p_score_bin": p_score_bin,
            "p_call_threat": p_call_threat,
            "p_tenpai": p_tenpai,
            "p_tsumo": p_tsumo,
            "river_spatial_feat": river_feat,  # (B, 34, 4 * river_dim)
        }
        if squeezed:
            out = {k: v[0] for k, v in out.items()}
        return out

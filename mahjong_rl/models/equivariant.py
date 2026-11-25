"""Sovereign-Mahjong V6 36x Algebraically Equivariant Architecture (S_3^suit x S_3^sangen).

Mathematical & structural guarantees:
1. Exact 36x equivariance under ALL 6 suit permutations (m/p/s) x ALL 6 dragon permutations
   (White/Green/Red):
   - `CausalRiverScatterEncoder`: shared dilated causal conv + shared per-suit causal conv
     + sentinel-guarded `segment_sum` scatter onto 34 tiles.
   - `SuitBlockedResBlock`: single shared 1D suit conv (`k=3`) across m/p/s, symmetric
     Cross-Suit Position-Aligned Sanshoku coupling, single shared `1x1` honor projection,
     global FiLM conditioning, and permutation-invariant Squeeze-and-Excitation.
   - `EquivariantTileAttention`: 34-token MHSA with shared 9-rank embedding tiled across
     m/p/s, 4-wind embedding on [27:31], and a SINGLE shared dragon embedding tiled across
     [31:34].
   - `TileAlignedPolicyHead`: tile-aligned heads for discards (`0..33`), red-5s (`34..36`),
     self-kans (`37..70`), safe-gathered `TSUMOGIRI (71)`, and 5-tile local suit-neighborhood
     slicer for calls (`72..86`).
2. Mutually Disjoint Bounded Physics Residual Gates:
   - `beta_risk = beta_risk_max * sigmoid(raw_beta_risk)` acts ONLY on discard actions
     (`0..36, 71`) via signed `Delta_U_ron_87` (preserving South-4 Sashikomi!).
   - `beta_offense = beta_offense_max * sigmoid(raw_beta_offense)` acts ONLY on `RIICHI (72)`
     (`1.0x`) and `PON..CHI_R_RED (75..83)` (`0.6x`), strictly ZERO on `TSUMO (73)`, `RON (74)`,
     `PASS (84)`, `KYUUSHU (85)`, `DUMMY (86)`.
3. Legal-action zero-mean centering + bidirectional `[-8.0, +8.0]` clipping (`exp(16) ~ 8.88e6`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from mahjax.common.action import Action
from mahjax.common.constants import RED_FIVE_TILE_TYPES
from mahjong_rl.config import ModelConfig

NUM_TILES = 34
NUM_ACTIONS = Action.NUM_ACTION  # 87


def _make_discard_action_mask() -> jnp.ndarray:
    mask = jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)
    mask = mask.at[0:37].set(1.0)
    mask = mask.at[Action.TSUMOGIRI].set(1.0)
    return mask


def _make_offense_action_weights() -> jnp.ndarray:
    weights = jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)
    weights = weights.at[Action.RIICHI].set(1.0)
    # PON (75) .. CHI_R_RED (83); strictly 0 on TSUMO (73), RON (74), PASS (84), KYUUSHU (85), DUMMY (86)
    weights = weights.at[Action.PON : Action.CHI_R_RED + 1].set(0.6)
    return weights


def _scatter_river_to_34(
    feats: jnp.ndarray,
    tile_ids: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Safely scatter `(24, D)` river features onto `(34, D)` via sentinel-guarded segment_sum.

    Invalid slots (`valid_mask == False` or `tile_id < 0`) are routed to sentinel index `34`
    and discarded cleanly when slicing `[:34]`.
    """
    safe_ids = jnp.where(valid_mask & (tile_ids >= 0) & (tile_ids < NUM_TILES), tile_ids, NUM_TILES)
    masked_feats = jnp.where(valid_mask[:, None], feats, 0.0)
    scattered = jax.ops.segment_sum(masked_feats, safe_ids.astype(jnp.int32), num_segments=NUM_TILES + 1)
    return scattered[:NUM_TILES]  # (34, D)


class CausalRiverScatterEncoder(nn.Module):
    """Dual-stream causal river encoder (Global Dilated + Shared Per-Suit) scattered to 34 tiles.

    Strictly 36x equivariant: `river_tokens` carries zero suit-ID and zero dragon-ID features;
    the per-suit branch shares a single causal kernel identically across suits `0, 1, 2`.
    """

    river_dim: int = 32

    @nn.compact
    def __call__(
        self,
        river_tokens: jnp.ndarray,      # (B, 4, 24, 23)
        river_tile_ids: jnp.ndarray,    # (B, 4, 24) int32
        river_valid_mask: jnp.ndarray,  # (B, 4, 24) bool
    ) -> jnp.ndarray:
        bsz, n_seats, seq_len, feat_dim = river_tokens.shape
        x = river_tokens.astype(jnp.float32).reshape((bsz * n_seats, seq_len, feat_dim))
        tids = river_tile_ids.astype(jnp.int32).reshape((bsz * n_seats, seq_len))
        valid = river_valid_mask.astype(jnp.bool_).reshape((bsz * n_seats, seq_len))
        valid_f = valid.astype(jnp.float32)[..., None]

        # 1. Input projection
        h0 = nn.gelu(nn.Dense(self.river_dim)(x)) * valid_f

        # 2. Global dilated causal temporal convolutions (dilations 1, 2, 4)
        h = h0
        for dilation in (1, 2, 4):
            res = nn.LayerNorm()(h)
            res = nn.Conv(
                features=self.river_dim,
                kernel_size=(3,),
                kernel_dilation=(dilation,),
                padding="CAUSAL",
            )(res)
            h = (h + nn.gelu(res)) * valid_f

        # 3. Shared Per-Suit causal convolution (exact S_3^suit equivariance via weight sharing!)
        suit_conv = nn.Conv(
            features=self.river_dim,
            kernel_size=(3,),
            padding="CAUSAL",
            name="shared_per_suit_causal_conv",
        )
        suit_accum = jnp.zeros_like(h0)
        for suit_idx in range(3):
            s_mask = (valid & (tids >= suit_idx * 9) & (tids < (suit_idx + 1) * 9)).astype(jnp.float32)[..., None]
            s_out = nn.gelu(suit_conv(h0 * s_mask)) * s_mask
            suit_accum = suit_accum + s_out

        # Fuse global + per-suit causal representations
        fused = nn.LayerNorm()(h + suit_accum) * valid_f  # (B*4, 24, D)

        # 4. Safe scatter onto 34 tile positions per seat
        scattered = jax.vmap(_scatter_river_to_34)(fused, tids, valid)  # (B*4, 34, D)
        scattered = scattered.reshape((bsz, n_seats, NUM_TILES, self.river_dim))
        # Transpose to (B, 34, 4 * D)
        return jnp.transpose(scattered, (0, 2, 1, 3)).reshape((bsz, NUM_TILES, n_seats * self.river_dim))


class SuitBlockedResBlock(nn.Module):
    """Suit-Blocked Residual Block with Cross-Suit Sanshoku Coupling, FiLM, and SE.

    - Suits `m (0:9)`, `p (9:18)`, `s (18:27)` share ONE 1D conv weight tensor.
    - Cross-suit coupling injects `W_cross((other_suit_1 + other_suit_2) / 2)` symmetrically.
    - Honors `(27:34)` share ONE `1x1` projection so White/Green/Red dragons commute identically.
    """

    channels: int = 128
    se_reduction: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond_emb: jnp.ndarray) -> jnp.ndarray:
        # x: (B, 34, C), cond_emb: (B, G)
        bsz = x.shape[0]
        h = nn.LayerNorm()(x)

        # FiLM conditioning (broadcast uniformly across all 34 tiles -> permutation-invariant!)
        film = nn.Dense(2 * self.channels)(cond_emb)
        gamma, beta = jnp.split(film, 2, axis=-1)
        h = h * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        h = nn.gelu(h)

        # Split into 3 suits (B, 3, 9, C) and 7 honors (B, 7, C)
        suits = h[:, :27, :].reshape((bsz, 3, 9, self.channels))
        honors = h[:, 27:34, :]

        # First conv: batched across the 3 suits for exact weight sharing & GPU efficiency
        conv1 = nn.Conv(features=self.channels, kernel_size=(3,), padding="SAME", name="suit_conv1")
        suits_c1 = conv1(suits.reshape((bsz * 3, 9, self.channels))).reshape((bsz, 3, 9, self.channels))

        # Cross-Suit Position-Aligned Sanshoku Coupling (symmetric across the 2 other suits!)
        cross_proj = nn.Dense(self.channels, name="sanshoku_cross_proj")
        m_in, p_in, s_in = suits[:, 0], suits[:, 1], suits[:, 2]
        m_other = 0.5 * (p_in + s_in)
        p_other = 0.5 * (m_in + s_in)
        s_other = 0.5 * (m_in + p_in)
        other_stacked = jnp.stack([m_other, p_other, s_other], axis=1)  # (B, 3, 9, C)
        suits_h = suits_c1 + cross_proj(other_stacked)

        honors_h = nn.Dense(self.channels, name="honor_proj1")(honors)

        h2 = jnp.concatenate([suits_h.reshape((bsz, 27, self.channels)), honors_h], axis=1)
        h2 = nn.gelu(nn.LayerNorm()(h2))

        # Second shared conv / honor projection
        suits2 = h2[:, :27, :].reshape((bsz * 3, 9, self.channels))
        suits_out = nn.Conv(features=self.channels, kernel_size=(3,), padding="SAME", name="suit_conv2")(suits2)
        suits_out = suits_out.reshape((bsz, 27, self.channels))
        honors_out = nn.Dense(self.channels, name="honor_proj2")(h2[:, 27:34, :])
        out = jnp.concatenate([suits_out, honors_out], axis=1)

        # Squeeze-and-Excitation (global mean over 34 tiles is strictly permutation-invariant)
        se_hidden = max(self.channels // self.se_reduction, 16)
        se = jnp.mean(out, axis=1)
        se = nn.gelu(nn.Dense(se_hidden)(se))
        se = jax.nn.sigmoid(nn.Dense(self.channels)(se))
        out = out * se[:, None, :]

        return x + out


class EquivariantTileAttention(nn.Module):
    """34-token Multi-Head Self-Attention with exact S_3^suit x S_3^sangen positional encodings."""

    channels: int = 128
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Shared rank embeddings (9, C) tiled identically across m/p/s
        pos_rank = self.param("pos_rank", nn.initializers.normal(stddev=0.02), (9, self.channels))
        # Distinct wind embeddings (4, C) for E/S/W/N ([27:31])
        pos_winds = self.param("pos_winds", nn.initializers.normal(stddev=0.02), (4, self.channels))
        # SINGLE shared dragon embedding (1, C) tiled across White/Green/Red ([31:34])!
        pos_sangen = self.param("pos_sangen", nn.initializers.normal(stddev=0.02), (1, self.channels))

        pos_emb = jnp.concatenate(
            [
                jnp.tile(pos_rank, (3, 1)),    # 27 suit tiles
                pos_winds,                     # 4 winds
                jnp.tile(pos_sangen, (3, 1)),  # 3 dragons (identical!)
            ],
            axis=0,
        )  # (34, C)

        h = nn.LayerNorm()(x + pos_emb[None, :, :])
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.channels,
            out_features=self.channels,
            deterministic=True,
        )(h, h)
        x = x + attn_out

        # Pointwise FFN (shared across all 34 tiles)
        ff = nn.LayerNorm()(x)
        ff = nn.Dense(self.channels * 2)(ff)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.channels)(ff)
        return x + ff


class EquivariantMahjongBackbone(nn.Module):
    """Shared 36x Equivariant Backbone used by Policy, Belief, and History-State Oracle."""

    num_blocks: int = 4
    channels: int = 128
    river_dim: int = 32
    attn_heads: int = 4
    se_reduction: int = 4
    global_hidden_dim: int = 128

    @nn.compact
    def __call__(
        self,
        planes: jnp.ndarray,                              # (B, P, 34)
        cond_vec: jnp.ndarray,                            # (B, D_cond)
        river_tokens: Optional[jnp.ndarray] = None,       # (B, 4, 24, 23)
        river_tile_ids: Optional[jnp.ndarray] = None,     # (B, 4, 24)
        river_valid_mask: Optional[jnp.ndarray] = None,   # (B, 4, 24)
        precomputed_river_feat: Optional[jnp.ndarray] = None,  # (B, 34, 4*D_r) for Oracle reuse!
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if precomputed_river_feat is not None:
            river_feat = precomputed_river_feat.astype(jnp.float32)
        else:
            assert river_tokens is not None and river_tile_ids is not None and river_valid_mask is not None
            river_feat = CausalRiverScatterEncoder(river_dim=self.river_dim)(
                river_tokens, river_tile_ids, river_valid_mask
            )

        # Planes: (B, P, 34) -> (B, 34, P)
        planes_t = jnp.transpose(planes.astype(jnp.float32), (0, 2, 1))
        spatial_in = jnp.concatenate([planes_t, river_feat], axis=-1)  # (B, 34, P + 4*D_r)

        cond_emb = nn.gelu(nn.Dense(self.global_hidden_dim)(cond_vec.astype(jnp.float32)))
        cond_emb = nn.LayerNorm()(cond_emb)

        x = nn.Dense(self.channels)(spatial_in)
        x = nn.gelu(nn.LayerNorm()(x))

        for _ in range(self.num_blocks):
            x = SuitBlockedResBlock(channels=self.channels, se_reduction=self.se_reduction)(x, cond_emb)

        x = EquivariantTileAttention(channels=self.channels, num_heads=self.attn_heads)(x)
        x = nn.LayerNorm()(x)

        # Permutation-invariant global readout (mean + max + cond_emb)
        pool_mean = jnp.mean(x, axis=1)
        pool_max = jnp.max(x, axis=1)
        g = jnp.concatenate([pool_mean, pool_max, cond_emb], axis=-1)
        g = nn.gelu(nn.Dense(self.global_hidden_dim)(g))
        g = nn.LayerNorm()(g)
        return x, g, river_feat


def extract_chi_neighborhood_features(
    x_34: jnp.ndarray,         # (B, 34, C)
    target_type: jnp.ndarray,  # (B,) int32 (-1 if none, 0..33 if valid)
) -> jnp.ndarray:
    """5-Tile Local Suit-Neighborhood Slicer `[target-2, target-1, target, target+1, target+2]`.

    - Zero-pads cleanly across suit boundaries (never leaks 9m into 1p!).
    - For honor targets (`27..33`), places `x_34[target]` in the center slot (`offset=0`)
      and zeros the 4 flanking slots.
    - For `target_type < 0`, returns all zeros.
    - Strictly 36x equivariant!
    """
    bsz, _, channels = x_34.shape
    t = target_type.astype(jnp.int32)  # (B,)
    has_tgt = (t >= 0) & (t < NUM_TILES)
    is_suit = has_tgt & (t < 27)
    suit = jnp.where(is_suit, t // 9, 0)
    rank = jnp.where(is_suit, t % 9, 0)

    batch_idx = jnp.arange(bsz, dtype=jnp.int32)
    slices = []
    for offset in (-2, -1, 0, 1, 2):
        if offset == 0:
            idx = jnp.clip(t, 0, NUM_TILES - 1)
            valid = has_tgt
        else:
            r = rank + offset
            valid = is_suit & (r >= 0) & (r < 9)
            idx = suit * 9 + jnp.clip(r, 0, 8)
        gathered = x_34[batch_idx, idx]  # (B, C)
        slices.append(jnp.where(valid[:, None], gathered, 0.0))

    return jnp.concatenate(slices, axis=-1)  # (B, 5 * C)


class TileAlignedPolicyHead(nn.Module):
    """36x Equivariant 87-Action Policy Head aligned directly to `x_34` tile representations."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(
        self,
        x_34: jnp.ndarray,            # (B, 34, C)
        g: jnp.ndarray,               # (B, G)
        last_draw_type: jnp.ndarray,  # (B,) int32
        is_last_draw_red: jnp.ndarray,  # (B,) bool/float
        target_type: jnp.ndarray,     # (B,) int32
        target_is_red: jnp.ndarray,   # (B,) bool/float
    ) -> jnp.ndarray:
        bsz = x_34.shape[0]
        g_tile = jnp.broadcast_to(g[:, None, :], (bsz, NUM_TILES, g.shape[-1]))
        xg = jnp.concatenate([x_34, g_tile], axis=-1)  # (B, 34, C + G)

        # 1. Normal discards 0..33 (shared suit head + shared honor head)
        suit_disc = nn.gelu(nn.Dense(self.hidden_dim // 2)(xg[:, :27, :]))
        suit_disc_logits = nn.Dense(1)(suit_disc).squeeze(-1)  # (B, 27)

        honor_disc = nn.gelu(nn.Dense(self.hidden_dim // 2)(xg[:, 27:34, :]))
        honor_disc_logits = nn.Dense(1)(honor_disc).squeeze(-1)  # (B, 7)

        # 2. Red-5 discards 34..36 (shared red-5 head on tiles [4, 13, 22])
        red_idx = jnp.asarray(RED_FIVE_TILE_TYPES, dtype=jnp.int32)
        red_feats = xg[:, red_idx, :]  # (B, 3, C + G)
        red_h = nn.gelu(nn.Dense(self.hidden_dim // 2)(red_feats))
        red_logits = nn.Dense(1)(red_h).squeeze(-1)  # (B, 3)

        # 3. Self-kans 37..70 (shared suit kan head + shared honor kan head)
        suit_kan = nn.gelu(nn.Dense(self.hidden_dim // 2)(xg[:, :27, :]))
        suit_kan_logits = nn.Dense(1)(suit_kan).squeeze(-1)  # (B, 27)

        honor_kan = nn.gelu(nn.Dense(self.hidden_dim // 2)(xg[:, 27:34, :]))
        honor_kan_logits = nn.Dense(1)(honor_kan).squeeze(-1)  # (B, 7)

        # 4. Action 71: TSUMOGIRI (safe gather of drawn tile + is_last_draw_red + global)
        batch_idx = jnp.arange(bsz, dtype=jnp.int32)
        has_draw = (last_draw_type >= 0) & (last_draw_type < NUM_TILES)
        safe_ld = jnp.clip(last_draw_type.astype(jnp.int32), 0, NUM_TILES - 1)
        draw_feat = jnp.where(has_draw[:, None], x_34[batch_idx, safe_ld], 0.0)
        tsumo_in = jnp.concatenate(
            [draw_feat, g, is_last_draw_red.astype(jnp.float32)[:, None]],
            axis=-1,
        )
        tsumo_logit = nn.Dense(1)(nn.gelu(nn.Dense(self.hidden_dim)(tsumo_in)))  # (B, 1)

        # 5. Actions 72..86 (15 actions): 5-tile local suit-neighborhood slicer + target_is_red + global
        chi_nbhd = extract_chi_neighborhood_features(x_34, target_type)  # (B, 5 * C)
        call_in = jnp.concatenate(
            [chi_nbhd, g, target_is_red.astype(jnp.float32)[:, None]],
            axis=-1,
        )
        call_h = nn.gelu(nn.Dense(self.hidden_dim)(call_in))
        call_logits = nn.Dense(NUM_ACTIONS - (Action.TSUMOGIRI + 1))(call_h)  # (B, 15)

        return jnp.concatenate(
            [
                suit_disc_logits,   # 0..26 (27)
                honor_disc_logits,  # 27..33 (7)
                red_logits,         # 34..36 (3)
                suit_kan_logits,    # 37..63 (27)
                honor_kan_logits,   # 64..70 (7)
                tsumo_logit,        # 71 (1)
                call_logits,        # 72..86 (15)
            ],
            axis=-1,
        )  # (B, 87)


class SovereignPolicyNet(nn.Module):
    """Sovereign-Mahjong V6 36x Equivariant Policy Network with Disjoint Bounded Physics Gates."""

    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(
        self,
        planes: jnp.ndarray,              # (..., 53, 34)
        delta_u_ron_3opp: jnp.ndarray,    # (..., 3, 34)
        call_threat_plane: jnp.ndarray,   # (..., 34)
        scalars: jnp.ndarray,             # (..., 56)
        river_tokens: jnp.ndarray,        # (..., 4, 24, 23)
        river_tile_ids: jnp.ndarray,      # (..., 4, 24)
        river_valid_mask: jnp.ndarray,    # (..., 4, 24)
        last_draw_type: jnp.ndarray,      # (...,)
        is_last_draw_red: jnp.ndarray,    # (...,)
        target_type: jnp.ndarray,         # (...,)
        target_is_red: jnp.ndarray,       # (...,)
        w_ego: jnp.ndarray,               # (..., 4)
        delta_u_agari_3: jnp.ndarray,     # (..., 3)
        delta_u_ryuukyoku: jnp.ndarray,   # (...,)
        delta_u_ron_87: jnp.ndarray,      # (..., 87)
        legal_action_mask: jnp.ndarray,   # (..., 87)
        return_aux: bool = False,
    ) -> jnp.ndarray | Tuple[jnp.ndarray, dict]:
        squeezed = planes.ndim == 2
        if squeezed:
            planes = planes[None, ...]
            delta_u_ron_3opp = delta_u_ron_3opp[None, ...]
            call_threat_plane = call_threat_plane[None, ...]
            scalars = scalars[None, ...]
            river_tokens = river_tokens[None, ...]
            river_tile_ids = river_tile_ids[None, ...]
            river_valid_mask = river_valid_mask[None, ...]
            last_draw_type = last_draw_type[None, ...]
            is_last_draw_red = is_last_draw_red[None, ...]
            target_type = target_type[None, ...]
            target_is_red = target_is_red[None, ...]
            w_ego = w_ego[None, ...]
            delta_u_agari_3 = delta_u_agari_3[None, ...]
            delta_u_ryuukyoku = delta_u_ryuukyoku[None, ...]
            delta_u_ron_87 = delta_u_ron_87[None, ...]
            legal_action_mask = legal_action_mask[None, ...]

        # Combine 53 public+TUO planes + 3 Delta_U_ron_3opp + 1 call_threat_plane = 57 spatial planes
        spatial_planes = jnp.concatenate(
            [
                planes.astype(jnp.float32),
                delta_u_ron_3opp.astype(jnp.float32),
                call_threat_plane.astype(jnp.float32)[:, None, :],
            ],
            axis=1,
        )  # (B, 57, 34)

        # Combine 56 scalars + 4 W'_ego + 3 Delta_U_agari + 1 Delta_U_ryuukyoku = 64 conditioning dims
        cond_vec = jnp.concatenate(
            [
                scalars.astype(jnp.float32),
                w_ego.astype(jnp.float32),
                delta_u_agari_3.astype(jnp.float32),
                delta_u_ryuukyoku.astype(jnp.float32)[:, None],
            ],
            axis=-1,
        )  # (B, 64)

        x_34, g, _ = EquivariantMahjongBackbone(
            num_blocks=self.config.num_blocks,
            channels=self.config.channels,
            river_dim=self.config.river_dim,
            attn_heads=self.config.attn_heads,
            se_reduction=self.config.se_reduction,
            global_hidden_dim=self.config.global_hidden_dim,
        )(
            planes=spatial_planes,
            cond_vec=cond_vec,
            river_tokens=river_tokens,
            river_tile_ids=river_tile_ids,
            river_valid_mask=river_valid_mask,
        )

        raw_logits = TileAlignedPolicyHead(hidden_dim=self.config.global_hidden_dim)(
            x_34=x_34,
            g=g,
            last_draw_type=last_draw_type,
            is_last_draw_red=is_last_draw_red,
            target_type=target_type,
            target_is_red=target_is_red,
        )  # (B, 87)

        # Mutually Disjoint Bounded Strictly-Positive Physics Residual Gates
        raw_beta_risk = self.param("raw_beta_risk", nn.initializers.constant(-1.5), ())
        raw_beta_offense = self.param("raw_beta_offense", nn.initializers.constant(-1.5), ())
        beta_risk = self.config.beta_risk_max * jax.nn.sigmoid(raw_beta_risk)
        beta_offense = self.config.beta_offense_max * jax.nn.sigmoid(raw_beta_offense)

        discard_mask = _make_discard_action_mask()[None, :]        # (1, 87)
        offense_weights = _make_offense_action_weights()[None, :]  # (1, 87)

        # Defense residual: ONLY on discard actions (0..36, 71), signed (preserves Sashikomi!)
        risk_residual = beta_risk * (delta_u_ron_87.astype(jnp.float32) * discard_mask)

        # Offense residual: ONLY on RIICHI (72) & PON..CHI_R_RED (75..83); strictly 0 on PASS (84)!
        offense_gain = jnp.clip(jnp.mean(delta_u_agari_3.astype(jnp.float32), axis=-1) * 4.0, 0.0, 4.0)
        offense_residual = beta_offense * offense_gain[:, None] * offense_weights

        combined_logits = raw_logits + risk_residual + offense_residual

        # Zero-mean centering over legal actions + bidirectional clipping [-8.0, +8.0]
        mask = legal_action_mask.astype(jnp.bool_)
        mask_f = mask.astype(jnp.float32)
        legal_count = jnp.maximum(jnp.sum(mask_f, axis=-1, keepdims=True), 1.0)
        legal_mean = jnp.sum(jnp.where(mask, combined_logits, 0.0), axis=-1, keepdims=True) / legal_count
        centered = combined_logits - legal_mean
        bound = float(self.config.logit_clip_bound)
        clipped = jnp.clip(centered, -bound, bound)
        # Keep illegal logits cleanly zeroed (downstream log_softmax/sampling masks them with NEG_INF)
        final_logits = jnp.where(mask, clipped, 0.0)

        if squeezed:
            final_logits = final_logits[0]
        if return_aux:
            aux = {
                "beta_risk": beta_risk,
                "beta_offense": beta_offense,
                "raw_logits": raw_logits[0] if squeezed else raw_logits,
            }
            return final_logits, aux
        return final_logits

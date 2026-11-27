"""vNM positive-affine normalized utilities, unbiased 5D opponent dropout, and multi-rule table sampling.

All placement utility vectors `w` are normalized to exact zero-mean, unit-spread:
    w' = (w - mean(w)) / (w[0] - w[3])
which preserves the exact Nash equilibrium set (von Neumann-Morgenstern Theorem) while
eliminating exploration temperature drift across rules (Tenhou 4d..10d & M-League).
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

# Raw official Tenhou Hanchan & M-League Uma placement point vectors
RAW_TOKUJOU_4D: Tuple[float, float, float, float] = (75.0, 30.0, 0.0, -90.0)
RAW_TOKUJOU_5D: Tuple[float, float, float, float] = (75.0, 30.0, 0.0, -105.0)
RAW_TOKUJOU_6D: Tuple[float, float, float, float] = (75.0, 30.0, 0.0, -120.0)
RAW_PHOENIX_7D: Tuple[float, float, float, float] = (90.0, 45.0, 0.0, -135.0)
RAW_PHOENIX_8D: Tuple[float, float, float, float] = (90.0, 45.0, 0.0, -150.0)
RAW_PHOENIX_9D: Tuple[float, float, float, float] = (90.0, 45.0, 0.0, -165.0)
RAW_PHOENIX_10D: Tuple[float, float, float, float] = (90.0, 45.0, 0.0, -180.0)
RAW_M_LEAGUE_UMA: Tuple[float, float, float, float] = (50.0, 10.0, -10.0, -30.0)


def vnm_normalize_utility(w: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Apply positive-affine normalization: zero-mean and unit spread (w'[0] - w'[3] = 1.0).

    Supports arbitrary leading batch dimensions `(..., 4)`.
    """
    arr = jnp.asarray(w, dtype=jnp.float32)
    mean = jnp.mean(arr, axis=-1, keepdims=True)
    spread = jnp.maximum(arr[..., :1] - arr[..., 3:4], eps)
    return (arr - mean) / spread


# Pre-normalized canonical utility vectors (unit spread = 1.0, mean = 0.0)
W_TOKUJOU_4D = vnm_normalize_utility(jnp.array(RAW_TOKUJOU_4D, dtype=jnp.float32))
W_TOKUJOU_5D = vnm_normalize_utility(jnp.array(RAW_TOKUJOU_5D, dtype=jnp.float32))
W_TOKUJOU_6D = vnm_normalize_utility(jnp.array(RAW_TOKUJOU_6D, dtype=jnp.float32))
W_PHOENIX_7D = vnm_normalize_utility(jnp.array(RAW_PHOENIX_7D, dtype=jnp.float32))
W_PHOENIX_8D = vnm_normalize_utility(jnp.array(RAW_PHOENIX_8D, dtype=jnp.float32))
W_PHOENIX_9D = vnm_normalize_utility(jnp.array(RAW_PHOENIX_9D, dtype=jnp.float32))
W_PHOENIX_10D = vnm_normalize_utility(jnp.array(RAW_PHOENIX_10D, dtype=jnp.float32))
W_M_LEAGUE = vnm_normalize_utility(jnp.array(RAW_M_LEAGUE_UMA, dtype=jnp.float32))

# Canonical lookup table for Tenhou dan strings / IDs
DAN_UTILITY_MAP: Dict[str, jnp.ndarray] = {
    "4d": W_TOKUJOU_4D,
    "5d": W_TOKUJOU_5D,
    "6d": W_TOKUJOU_6D,
    "7d": W_PHOENIX_7D,
    "8d": W_PHOENIX_8D,
    "9d": W_PHOENIX_9D,
    "10d": W_PHOENIX_10D,
    "mleague": W_M_LEAGUE,
}

# Convex-hull rule bank covering full Tenhou 4d..10d + M-League corridor
RULE_CORRIDOR_BANK = jnp.stack(
    [
        W_PHOENIX_7D,   # 0: balanced core
        W_PHOENIX_8D,   # 1
        W_PHOENIX_9D,   # 2
        W_PHOENIX_10D,  # 3: extreme avoid-4th vertex
        W_TOKUJOU_4D,   # 4: mild avoid-4th
        W_M_LEAGUE,     # 5: top-seeking vertex
    ],
    axis=0,
)  # (6, 4)


def rotate_utility_matrix_ego(w_table_abs: jnp.ndarray, current_player: jnp.ndarray) -> jnp.ndarray:
    """Rotate absolute-seat utility matrix `(4, 4)` into ego-centric frame `[me, right, across, left]`."""
    order = (jnp.arange(4, dtype=jnp.int32) + current_player.astype(jnp.int32)) % 4
    return jnp.asarray(w_table_abs, dtype=jnp.float32)[order]  # (4, 4)


def build_opponent_utility_features(
    w_opp_3x4: jnp.ndarray,
    is_known_3: jnp.ndarray,
    rng_key: jax.Array | None = None,
    dropout_prob: float = 0.20,
    w_table_default: jnp.ndarray = W_PHOENIX_7D,
) -> jnp.ndarray:
    """Build 5D per-opponent utility features `[w'_1, w'_2, w'_3, w'_4, is_known]`.

    CRITICAL V6 INVARIANT (Round 26):
    When an opponent's rule is unknown or dropped out, NEVER fill with `[0, 0, 0, 0]`
    (which encodes an indifferent randomizer!). Always fall back to the room-average prior
    `w_table_default` paired with `is_known = 0.0`, preserving exact Bayesian neutrality.

    Supports both single-state `(3, 4)` / `(3,)` and batched `(..., 3, 4)` / `(..., 3)` inputs.
    """
    w_opp = jnp.asarray(w_opp_3x4, dtype=jnp.float32)
    known = jnp.asarray(is_known_3, dtype=jnp.float32)
    if rng_key is not None and dropout_prob > 0.0:
        keep = jax.random.bernoulli(rng_key, p=1.0 - float(dropout_prob), shape=known.shape).astype(jnp.float32)
        eff_known = known * keep
    else:
        eff_known = known

    default_b = jnp.broadcast_to(jnp.asarray(w_table_default, dtype=jnp.float32), w_opp.shape)
    eff_w = jnp.where(eff_known[..., None] > 0.5, w_opp, default_b)
    return jnp.concatenate([eff_w, eff_known[..., None]], axis=-1)  # (..., 3, 5)


def make_homogeneous_table(w: jnp.ndarray) -> jnp.ndarray:
    """Broadcast a single `(4,)` rule vector to all 4 seats `(4, 4)`."""
    return jnp.broadcast_to(jnp.asarray(w, dtype=jnp.float32)[None, :], (4, 4))


def sample_meta_table_pool(
    num_envs: int,
    rng_key: jax.Array,
    *,
    phoenix_7d_only: bool = False,
    pfsp_frac: float = 0.35,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Allocate the V6 Meta-Game Table Pool across `num_envs` parallel tables.

    When `phoenix_7d_only=True`:
    - All tables use `W_PHOENIX_7D`, split between homogeneous self-play (all 4 seats learning)
      and 1v3 PFSP tables (`pfsp_frac` fraction, learning seat rotated evenly across 0..3).

    When `phoenix_7d_only=False` (Tri-Regime Meta-Game Pool):
    - Block A (50% Homogeneous Self-Play, all 4 seats learning): 25% 7d / 15% 10d / 10% M-League
    - Block B (35% PFSP Historical Snapshot League, rotated single learning seat)
    - Block C (15% Extreme Cross-Dan Heterogeneous Tables, all 4 seats learning)

    Returns:
        w_tables: `(num_envs, 4, 4)` float32 normalized utility matrices (absolute seats 0..3)
        is_pfsp_env: `(num_envs,)` bool, True for PFSP tables
        learning_seat_mask: `(num_envs, 4)` bool, True for seats controlled by active `params_pi`
    """
    if phoenix_7d_only:
        n_pfsp = min(max(0, int(round(num_envs * float(pfsp_frac)))), max(num_envs - 1, 0))
        n_hom = num_envs - n_pfsp
        w_tables = jnp.broadcast_to(make_homogeneous_table(W_PHOENIX_7D)[None, ...], (num_envs, 4, 4))
        env_ids = jnp.arange(num_envs, dtype=jnp.int32)
        is_pfsp_env = env_ids >= n_hom
        pfsp_seat_ids = jnp.mod(jnp.maximum(env_ids - n_hom, 0), 4)
        pfsp_seats = jax.nn.one_hot(pfsp_seat_ids, 4, dtype=jnp.bool_)
        all_seats = jnp.ones((4,), dtype=jnp.bool_)
        learning_seat_mask = jnp.where(is_pfsp_env[:, None], pfsp_seats, all_seats[None, :])
        return w_tables, is_pfsp_env, learning_seat_mask

    n_hom = max(1, int(round(num_envs * 0.50)))
    n_pfsp = int(round(num_envs * 0.35))
    n_het = max(0, num_envs - n_hom - n_pfsp)
    if n_hom + n_pfsp + n_het > num_envs:
        n_pfsp = max(0, num_envs - n_hom - n_het)

    # Block A homogeneous breakdown (25% / 15% / 10% of total -> 50% / 30% / 20% of Block A)
    n_7d = max(1, int(round(n_hom * 0.50)))
    n_10d = int(round(n_hom * 0.30))
    n_ml = max(0, n_hom - n_7d - n_10d)
    if n_7d + n_10d + n_ml > n_hom:
        n_10d = max(0, n_hom - n_7d - n_ml)

    blocks = []
    if n_7d > 0:
        blocks.append(jnp.broadcast_to(make_homogeneous_table(W_PHOENIX_7D)[None, ...], (n_7d, 4, 4)))
    if n_10d > 0:
        blocks.append(jnp.broadcast_to(make_homogeneous_table(W_PHOENIX_10D)[None, ...], (n_10d, 4, 4)))
    if n_ml > 0:
        blocks.append(jnp.broadcast_to(make_homogeneous_table(W_M_LEAGUE)[None, ...], (n_ml, 4, 4)))

    # Block B PFSP tables (6:3:2 ratio across 7d / 10d / M-League)
    if n_pfsp > 0:
        p_7d = max(1, int(round(n_pfsp * (6.0 / 11.0))))
        p_10d = int(round(n_pfsp * (3.0 / 11.0)))
        p_ml = max(0, n_pfsp - p_7d - p_10d)
        if p_7d + p_10d + p_ml > n_pfsp:
            p_10d = max(0, n_pfsp - p_7d - p_ml)
        if p_7d > 0:
            blocks.append(jnp.broadcast_to(make_homogeneous_table(W_PHOENIX_7D)[None, ...], (p_7d, 4, 4)))
        if p_10d > 0:
            blocks.append(jnp.broadcast_to(make_homogeneous_table(W_PHOENIX_10D)[None, ...], (p_10d, 4, 4)))
        if p_ml > 0:
            blocks.append(jnp.broadcast_to(make_homogeneous_table(W_M_LEAGUE)[None, ...], (p_ml, 4, 4)))

    # Block C heterogeneous cross-dan stress tables
    if n_het > 0:
        idx = jax.random.randint(rng_key, shape=(n_het, 4), minval=0, maxval=RULE_CORRIDOR_BANK.shape[0])
        blocks.append(RULE_CORRIDOR_BANK[idx])

    w_tables = jnp.concatenate(blocks, axis=0)  # (num_envs, 4, 4)

    env_ids = jnp.arange(num_envs, dtype=jnp.int32)
    is_pfsp_env = (env_ids >= n_hom) & (env_ids < n_hom + n_pfsp)
    # Rotate PFSP learning seat evenly across [0, 1, 2, 3] so dealer/seat biases are eliminated
    pfsp_seat_ids = jnp.mod(jnp.maximum(env_ids - n_hom, 0), 4)
    pfsp_seats = jax.nn.one_hot(pfsp_seat_ids, 4, dtype=jnp.bool_)
    all_seats = jnp.ones((4,), dtype=jnp.bool_)
    learning_seat_mask = jnp.where(is_pfsp_env[:, None], pfsp_seats, all_seats[None, :])
    return w_tables, is_pfsp_env, learning_seat_mask

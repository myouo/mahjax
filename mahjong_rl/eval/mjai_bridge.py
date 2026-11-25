"""Deployment bridge: zero-search pure-forward Sovereign-Mahjong V6 actor with runtime rule switching.

Zero hidden information, zero search tree, ~1.8ms pure-forward inference:
    obs_v6 -> frozen B_EMA (calibrated T_cal) -> frozen GRP_base Analytical Physics Trinity
           -> single 36x Equivariant SovereignPolicyNet -> argmax / sample action!
Supports instant zero-shot rule switching (`W_PHOENIX_7D`, `W_PHOENIX_10D`, `W_M_LEAGUE`, etc.)
without retraining!
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from mahjax.mahjong_4p.state import State
from mahjong_rl.algo.legal_ron import compute_analytical_trinity, extract_table_snapshot
from mahjong_rl.algo.sovereign_v6 import NEG_INF, masked_categorical_log_probs
from mahjong_rl.algo.utility import (
    W_PHOENIX_7D,
    build_opponent_utility_features,
    make_homogeneous_table,
    rotate_utility_matrix_ego,
)
from mahjong_rl.features.obs_2d import clean_auto_reset_state
from mahjong_rl.features.obs_v6 import encode_obs_v6
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRPNet


def make_v6_policy_actor(
    policy_model: SovereignPolicyNet,
    policy_params: Any,
    belief_model: BeliefNet,
    belief_ema_params: Any,
    grp_model: GRPNet,
    grp_base_params: Any,
    *,
    w_rule: jnp.ndarray = W_PHOENIX_7D,
    w_table_abs: Optional[jnp.ndarray] = None,
    opp_rules_known: bool = False,
    belief_t_cal: float = 1.0,
    greedy: bool = True,
) -> Callable[[State, jax.Array], jnp.ndarray]:
    """Build a JIT-friendly `act(state, key) -> int32` callable for evaluation and `mahjax.ui`.

    Args:
        w_rule: Normalized `(4,)` utility rule vector (e.g. `W_PHOENIX_7D`, `W_PHOENIX_10D`,
            `W_M_LEAGUE`) applied zero-shot at inference time!
        w_table_abs: Optional `(4, 4)` heterogeneous seat utility matrix overriding `w_rule`.
        opp_rules_known: If False (real-world anonymous tables), passes `is_known=0.0` with
            unbiased fallback to `w_rule` (NEVER all-zeros!).
        belief_t_cal: Post-training calibrated temperature `T_cal` for `B_EMA`.
        greedy: If True, returns argmax legal action; if False, samples from policy logits.
    """
    w_tab = make_homogeneous_table(w_rule) if w_table_abs is None else jnp.asarray(w_table_abs, dtype=jnp.float32)
    is_known_vec = jnp.full((3,), 1.0 if opp_rules_known else 0.0, dtype=jnp.float32)
    grp_fn = lambda gf: grp_model.apply(grp_base_params, gf)

    def actor(state: State, key: jax.Array) -> jnp.ndarray:
        clean_st = clean_auto_reset_state(state)
        cp = clean_st.current_player.astype(jnp.int32)
        legal_mask = clean_st.legal_action_mask.astype(jnp.bool_)

        obs = encode_obs_v6(clean_st)
        tbl = extract_table_snapshot(clean_st)
        w_ego_4x4 = rotate_utility_matrix_ego(w_tab, cp)
        w_me = w_ego_4x4[0]
        w_opp_5d = build_opponent_utility_features(
            w_ego_4x4[1:],
            is_known_vec,
            rng_key=None,
            dropout_prob=0.0,
            w_table_default=w_rule,
        )

        b_out = belief_model.apply(
            belief_ema_params,
            planes=obs["planes"],
            scalars=obs["scalars"],
            river_tokens=obs["river_tokens"],
            river_tile_ids=obs["river_tile_ids"],
            river_valid_mask=obs["river_valid_mask"],
            w_opp_5d=w_opp_5d,
            temperature_cal=belief_t_cal,
        )
        du_ron_3opp, du_ron_87, du_agari_3, du_ryu, call_threat = compute_analytical_trinity(
            s_table=tbl,
            p_deal_3opp=b_out["p_deal"],
            p_score_bins=b_out["p_score_bin"],
            p_call_threat_3=b_out["p_call_threat"],
            p_tenpai_3=b_out["p_tenpai"],
            w_me=w_me,
            last_draw_type=obs["last_draw_type"],
            is_last_draw_red=obs["is_last_draw_red"],
            grp_fn=grp_fn,
        )
        logits = policy_model.apply(
            policy_params,
            planes=obs["planes"],
            delta_u_ron_3opp=du_ron_3opp,
            call_threat_plane=call_threat,
            scalars=obs["scalars"],
            river_tokens=obs["river_tokens"],
            river_tile_ids=obs["river_tile_ids"],
            river_valid_mask=obs["river_valid_mask"],
            last_draw_type=obs["last_draw_type"],
            is_last_draw_red=obs["is_last_draw_red"],
            target_type=obs["target_type"],
            target_is_red=obs["target_is_red"],
            w_ego=w_me,
            delta_u_agari_3=du_agari_3,
            delta_u_ryuukyoku=du_ryu,
            delta_u_ron_87=du_ron_87,
            legal_action_mask=legal_mask,
        )
        if greedy:
            return jnp.argmax(jnp.where(legal_mask, logits, NEG_INF)).astype(jnp.int32)
        log_p = masked_categorical_log_probs(logits, legal_mask)
        return jax.random.categorical(key, log_p).astype(jnp.int32)

    return actor


def register_ui_agent(
    app: Any,
    *,
    name: str,
    policy_model: SovereignPolicyNet,
    policy_params: Any,
    belief_model: BeliefNet,
    belief_ema_params: Any,
    grp_model: GRPNet,
    grp_base_params: Any,
    w_rule: jnp.ndarray = W_PHOENIX_7D,
    belief_t_cal: float = 1.0,
    env_ids: Sequence[str] = ("red_4p", "no_red_4p"),
    description: str = "Sovereign-Mahjong V6 (Zero-Search 36x Equivariant) agent",
    first: bool = True,
) -> Any:
    """Register a trained Sovereign-Mahjong V6 agent onto a `mahjax.ui.app` FastAPI app."""
    act_fn = jax.jit(
        make_v6_policy_actor(
            policy_model=policy_model,
            policy_params=policy_params,
            belief_model=belief_model,
            belief_ema_params=belief_ema_params,
            grp_model=grp_model,
            grp_base_params=grp_base_params,
            w_rule=w_rule,
            belief_t_cal=belief_t_cal,
            greedy=True,
        )
    )
    return app.state.registry.add_agent(
        name=name,
        act=act_fn,
        env_ids=env_ids,
        description=description,
        first=first,
    )

"""V6 Invariant Tests 1 & 2: 36x Algebraic Equivariance (S_3^suit x S_3^sangen) & Zero Leakage."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjong_rl.algo.utility import W_PHOENIX_7D, build_opponent_utility_features
from mahjong_rl.config import BeliefConfig, ModelConfig
from mahjong_rl.data.mjai_dataset import (
    SYMMETRY_PERMS_36,
    build_tile_and_action_perm_36,
    permute_v6_batch,
)
from mahjong_rl.features.obs_2d import clean_auto_reset_state
from mahjong_rl.features.obs_v6 import (
    GLOBAL_SCALAR_DIM_V6,
    NUM_OBS_V6_PLANES,
    compute_topological_ukeire_planes,
    encode_obs_v6,
)
from mahjong_rl.features.oracle import NUM_ORACLE_PLANES, encode_oracle_planes
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet


def _advance_to_mid_hand(seed: int = 7, steps: int = 35):
    env = mahjax.make("red_4p", round_mode="half", observe_type="dict", round_end="auto")

    @jax.jit
    def _roll(key):
        k_init, k_play = jax.random.split(key)
        st0 = env.init(k_init)

        def body(carry, _):
            st, r = carry
            r, ka, ks = jax.random.split(r, 3)
            act = mahjax.rule_based_player(st, ka)
            nxt = jax.lax.cond(st.terminated | st.truncated, lambda: st, lambda: env.step(st, act, ks))
            return (nxt, r), None

        (st_f, _), _ = jax.lax.scan(body, (st0, k_play), None, length=steps)
        return clean_auto_reset_state(st_f)

    return _roll(jax.random.PRNGKey(seed))


def test_v6_36x_algebraic_equivariance_belief_and_policy():
    """Test 1: Exact 36x equivariance under ALL 36 elements of S_3^suit x S_3^sangen."""
    st = _advance_to_mid_hand(seed=11, steps=42)
    obs = encode_obs_v6(st)

    # Construct non-trivial synthetic call target, last draw, and physics risk planes
    # so EVERY branch (suits, honors, red-5, self-kan, TSUMOGIRI, 5-tile chi neighborhood) is exercised!
    planes = np.asarray(obs["planes"], dtype=np.float32)[None, ...]            # (1, 53, 34)
    scalars = np.asarray(obs["scalars"], dtype=np.float32)[None, ...]          # (1, 56)
    river_tokens = np.asarray(obs["river_tokens"], dtype=np.float32)[None, ...]
    river_tile_ids = np.asarray(obs["river_tile_ids"], dtype=np.int32)[None, ...]
    river_valid_mask = np.asarray(obs["river_valid_mask"], dtype=np.bool_)[None, ...]

    rng_np = np.random.default_rng(123)
    du_ron_3opp = rng_np.normal(0.0, 0.2, size=(1, 3, 34)).astype(np.float32)
    call_threat = rng_np.uniform(0.0, 0.5, size=(1, 34)).astype(np.float32)
    du_agari_3 = np.array([[0.12, 0.28, 0.41]], dtype=np.float32)
    du_ryu = np.array([0.05], dtype=np.float32)

    # Build a valid equivariant delta_u_ron_87 consistent with du_ron_3opp & red fives & TSUMOGIRI
    last_draw_type = np.array([13], dtype=np.int32)  # 5p (suit 1, rank 4)
    is_last_draw_red = np.array([True], dtype=np.bool_)
    target_type = np.array([32], dtype=np.int32)     # Green dragon (tests S_3^sangen on calls too!)
    target_is_red = np.array([False], dtype=np.bool_)

    du_norm_34 = du_ron_3opp.sum(axis=1)  # (1, 34)
    du_aka_3 = du_norm_34[:, [4, 13, 22]] * 1.15
    du_ron_87 = np.zeros((1, 87), dtype=np.float32)
    du_ron_87[:, :34] = du_norm_34
    du_ron_87[:, 34:37] = du_aka_3
    du_ron_87[:, 71] = du_aka_3[:, 1]  # 5p red tsumogiri
    legal_mask = np.ones((1, 87), dtype=np.bool_)

    w_ego = np.asarray(W_PHOENIX_7D, dtype=np.float32)[None, :]
    w_opp_5d = np.asarray(
        build_opponent_utility_features(
            jnp.broadcast_to(W_PHOENIX_7D[None, :], (1, 3, 4)),
            jnp.ones((1, 3), dtype=jnp.float32),
        ),
        dtype=np.float32,
    )

    b_cfg = BeliefConfig(num_blocks=2, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    p_cfg = ModelConfig(num_blocks=2, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    b_net = BeliefNet(config=b_cfg)
    p_net = SovereignPolicyNet(config=p_cfg)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    b_params = b_net.init(
        k1,
        planes=jnp.asarray(planes),
        scalars=jnp.asarray(scalars),
        river_tokens=jnp.asarray(river_tokens),
        river_tile_ids=jnp.asarray(river_tile_ids),
        river_valid_mask=jnp.asarray(river_valid_mask),
        w_opp_5d=jnp.asarray(w_opp_5d),
    )
    p_params = p_net.init(
        k2,
        planes=jnp.asarray(planes),
        delta_u_ron_3opp=jnp.asarray(du_ron_3opp),
        call_threat_plane=jnp.asarray(call_threat),
        scalars=jnp.asarray(scalars),
        river_tokens=jnp.asarray(river_tokens),
        river_tile_ids=jnp.asarray(river_tile_ids),
        river_valid_mask=jnp.asarray(river_valid_mask),
        last_draw_type=jnp.asarray(last_draw_type),
        is_last_draw_red=jnp.asarray(is_last_draw_red),
        target_type=jnp.asarray(target_type),
        target_is_red=jnp.asarray(target_is_red),
        w_ego=jnp.asarray(w_ego),
        delta_u_agari_3=jnp.asarray(du_agari_3),
        delta_u_ryuukyoku=jnp.asarray(du_ryu),
        delta_u_ron_87=jnp.asarray(du_ron_87),
        legal_action_mask=jnp.asarray(legal_mask),
    )

    # Run exact equivariance check in strict FP32 (disabling Ampere TF32 10-bit mantissa rounding)
    with jax.default_matmul_precision("highest"):
        b_apply = jax.jit(b_net.apply)
        p_apply = jax.jit(p_net.apply)

        base_b = jax.device_get(
            b_apply(
                b_params,
                planes=jnp.asarray(planes),
                scalars=jnp.asarray(scalars),
                river_tokens=jnp.asarray(river_tokens),
                river_tile_ids=jnp.asarray(river_tile_ids),
                river_valid_mask=jnp.asarray(river_valid_mask),
                w_opp_5d=jnp.asarray(w_opp_5d),
            )
        )
        base_p = np.asarray(
            jax.device_get(
                p_apply(
                    p_params,
                    planes=jnp.asarray(planes),
                    delta_u_ron_3opp=jnp.asarray(du_ron_3opp),
                    call_threat_plane=jnp.asarray(call_threat),
                    scalars=jnp.asarray(scalars),
                    river_tokens=jnp.asarray(river_tokens),
                    river_tile_ids=jnp.asarray(river_tile_ids),
                    river_valid_mask=jnp.asarray(river_valid_mask),
                    last_draw_type=jnp.asarray(last_draw_type),
                    is_last_draw_red=jnp.asarray(is_last_draw_red),
                    target_type=jnp.asarray(target_type),
                    target_is_red=jnp.asarray(target_is_red),
                    w_ego=jnp.asarray(w_ego),
                    delta_u_agari_3=jnp.asarray(du_agari_3),
                    delta_u_ryuukyoku=jnp.asarray(du_ryu),
                    delta_u_ron_87=jnp.asarray(du_ron_87),
                    legal_action_mask=jnp.asarray(legal_mask),
                )
            )
        )

        # Test ALL 36 group elements!
        assert len(SYMMETRY_PERMS_36) == 36
        base_batch = {
            "planes": planes,
            "river_tile_ids": river_tile_ids,
            "last_draw_type": last_draw_type,
            "target_type": target_type,
            "action_mask": legal_mask,
        }
        max_belief_err = 0.0
        max_policy_err = 0.0

        for perm_idx, (sp, dp) in enumerate(SYMMETRY_PERMS_36):
            tile_map, act_map = build_tile_and_action_perm_36(sp, dp)
            inv_tile = np.argsort(tile_map)
            inv_act = np.argsort(act_map)

            pb = permute_v6_batch(base_batch, perm_idx)
            p_du_3opp = du_ron_3opp[..., inv_tile]
            p_ct = call_threat[..., inv_tile]
            p_du_87 = du_ron_87[..., inv_act]

            out_b = jax.device_get(
                b_apply(
                    b_params,
                    planes=jnp.asarray(pb["planes"]),
                    scalars=jnp.asarray(scalars),
                    river_tokens=jnp.asarray(river_tokens),
                    river_tile_ids=jnp.asarray(pb["river_tile_ids"]),
                    river_valid_mask=jnp.asarray(river_valid_mask),
                    w_opp_5d=jnp.asarray(w_opp_5d),
                )
            )
            out_p = np.asarray(
                jax.device_get(
                    p_apply(
                        p_params,
                        planes=jnp.asarray(pb["planes"]),
                        delta_u_ron_3opp=jnp.asarray(p_du_3opp),
                        call_threat_plane=jnp.asarray(p_ct),
                        scalars=jnp.asarray(scalars),
                        river_tokens=jnp.asarray(river_tokens),
                        river_tile_ids=jnp.asarray(pb["river_tile_ids"]),
                        river_valid_mask=jnp.asarray(river_valid_mask),
                        last_draw_type=jnp.asarray(pb["last_draw_type"]),
                        is_last_draw_red=jnp.asarray(is_last_draw_red),
                        target_type=jnp.asarray(pb["target_type"]),
                        target_is_red=jnp.asarray(target_is_red),
                        w_ego=jnp.asarray(w_ego),
                        delta_u_agari_3=jnp.asarray(du_agari_3),
                        delta_u_ryuukyoku=jnp.asarray(du_ryu),
                        delta_u_ron_87=jnp.asarray(p_du_87),
                        legal_action_mask=jnp.asarray(pb["action_mask"]),
                    )
                )
            )

            # Belief outputs must permute equivariantly along the 34-tile axis!
            err_deal = float(np.max(np.abs(out_b["deal_logits"] - base_b["deal_logits"][..., inv_tile])))
            err_sb = float(np.max(np.abs(out_b["score_bin_logits"] - base_b["score_bin_logits"][:, :, inv_tile, :])))
            err_ct = float(np.max(np.abs(out_b["call_threat_logits"] - base_b["call_threat_logits"][..., inv_tile])))
            err_ten = float(np.max(np.abs(out_b["tenpai_logits"] - base_b["tenpai_logits"])))
            max_belief_err = max(max_belief_err, err_deal, err_sb, err_ct, err_ten)

            # Policy logits must permute equivariantly across ALL 87 actions!
            err_pol = float(np.max(np.abs(out_p - base_p[..., inv_act])))
            max_policy_err = max(max_policy_err, err_pol)

    assert max_belief_err < 1e-4, f"Belief 36x equivariance error too high: {max_belief_err}"
    assert max_policy_err < 1e-4, f"Policy 36x equivariance error too high: {max_policy_err}"


def test_v6_zero_hidden_information_leakage_including_ura_dora():
    """Test 2: Randomizing opponent hands, unrevealed wall, AND ura-dora causes ZERO change to obs_v6."""
    st = _advance_to_mid_hand(seed=19, steps=30)
    obs1 = jax.device_get(encode_obs_v6(st))
    ora1 = np.asarray(jax.device_get(encode_oracle_planes(st)))

    assert obs1["planes"].shape == (NUM_OBS_V6_PLANES, 34)
    assert obs1["scalars"].shape == (GLOBAL_SCALAR_DIM_V6,)
    assert ora1.shape == (NUM_ORACLE_PLANES, 34)

    # Perturb ALL hidden information: opponent hands, deck, AND ura_dora_indicators!
    cp = int(st.current_player)
    opp_seats = [s for s in range(4) if s != cp]
    new_hand = st.players.hand
    new_hand_red = st.players.hand_with_red
    # Permute opponents' concealed hands among the 3 opponents
    new_hand = new_hand.at[jnp.asarray(opp_seats)].set(
        new_hand[jnp.asarray([opp_seats[1], opp_seats[2], opp_seats[0]])]
    )
    new_hand_red = new_hand_red.at[jnp.asarray(opp_seats)].set(
        new_hand_red[jnp.asarray([opp_seats[1], opp_seats[2], opp_seats[0]])]
    )
    # Shuffle remaining deck & flip ura_dora_indicators completely
    new_deck = jnp.roll(st.round_state.deck, shift=17)
    new_ura = (st.round_state.ura_dora_indicators + 5) % 34

    perturbed_st = st.replace(
        players=st.players.replace(hand=new_hand, hand_with_red=new_hand_red),
        round_state=st.round_state.replace(deck=new_deck, ura_dora_indicators=new_ura),
    )

    obs2 = jax.device_get(encode_obs_v6(perturbed_st))
    ora2 = np.asarray(jax.device_get(encode_oracle_planes(perturbed_st)))

    # Public obs_v6 MUST be byte-for-byte identical!
    for k in ("planes", "scalars", "river_tokens", "river_tile_ids", "river_valid_mask"):
        np.testing.assert_array_equal(obs1[k], obs2[k], err_msg=f"Hidden info leaked into obs_v6['{k}']!")

    # Privileged Oracle planes MUST detect the perturbation!
    assert not np.array_equal(ora1, ora2), "Oracle planes failed to reflect hidden state changes"


def test_topological_ukeire_operator_multi_shanten():
    """TUO gives accurate non-zero ryanmen, kanchan, and pair wait potentials at high shanten."""
    hand = jnp.zeros((34,), dtype=jnp.int32)
    # 4m (3) + 5m (4) -> ryanmen waiting on 3m (2) & 6m (5)
    # 2p (10) + 4p (12) -> kanchan waiting on 3p (11)
    # East wind (27) -> pair potential
    hand = hand.at[3].set(1).at[4].set(1).at[10].set(1).at[12].set(1).at[27].set(1)
    rem = jnp.full((34,), 4.0, dtype=jnp.float32) - hand.astype(jnp.float32)

    tuo = np.asarray(compute_topological_ukeire_planes(hand, rem))  # (3, 34)
    assert tuo.shape == (3, 34)
    # Both 4m (3) and 5m (4) receive rem[3m] + rem[6m] = 4 + 4 = 8 -> 8 / 8 = 1.0!
    assert np.isclose(tuo[0, 3], 1.0)
    assert np.isclose(tuo[0, 4], 1.0)
    # Both 2p (10) and 4p (12) receive rem[3p] = 4 -> 4 / 8 = 0.5!
    assert np.isclose(tuo[1, 10], 0.5)
    assert np.isclose(tuo[1, 12], 0.5)
    # East wind (27) pair growth: rem[27] = 3 -> 3 / 4 = 0.75!
    assert np.isclose(tuo[2, 27], 0.75)

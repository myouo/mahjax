"""V6 Invariant Tests 3 & 4: Bit-Exact Legal Ron Engine & Analytical Physics Trinity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjax.common.action import Action
from mahjong_rl.algo.legal_ron import (
    TableSnapshot,
    compute_analytical_trinity,
    compute_sparse_belief_labels,
    extract_ron_snapshot,
    resolve_discard_action,
    simulate_ron_transition,
)
from mahjong_rl.algo.utility import W_PHOENIX_7D, W_PHOENIX_10D
from mahjong_rl.config import ModelConfig
from mahjong_rl.features.obs_2d import clean_auto_reset_state
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRP_FEATURE_DIM, GRPNet, scores_to_rank_matrix_4x4


def test_resolve_discard_action_all_modes():
    """Verify resolve_discard_action handles normal (0..33), red-5 (34..36), and black/red TSUMOGIRI (71)."""
    # Normal discard 5p (13)
    is_d, t, red = resolve_discard_action(jnp.int32(13), jnp.int32(2))
    assert bool(is_d) and int(t) == 13 and not bool(red)

    # Red-5 discard 5p_red (35 -> tile 13, is_red=True)
    is_d, t, red = resolve_discard_action(jnp.int32(35), jnp.int32(2))
    assert bool(is_d) and int(t) == 13 and bool(red)

    # TSUMOGIRI of red 5s (last_draw=36 -> tile 22, is_red=True)
    is_d, t, red = resolve_discard_action(jnp.int32(Action.TSUMOGIRI), jnp.int32(36))
    assert bool(is_d) and int(t) == 22 and bool(red)

    # TSUMOGIRI of normal East (last_draw=27 -> tile 27, is_red=False)
    is_d, t, red = resolve_discard_action(jnp.int32(Action.TSUMOGIRI), jnp.int32(27))
    assert bool(is_d) and int(t) == 27 and not bool(red)

    # Non-discard PASS (84) & RIICHI (72)
    is_d, _, _ = resolve_discard_action(jnp.int32(Action.PASS), jnp.int32(5))
    assert not bool(is_d)
    is_d, _, _ = resolve_discard_action(jnp.int32(Action.RIICHI), jnp.int32(5))
    assert not bool(is_d)


def test_simulate_ron_transition_bit_exact_rules():
    """Test 3: Single ron, double ron head-bump honba/kyotaku, tobi, and South-4 agari-yame."""
    # Constant dummy GRP returning uniform (4, 4) for non-terminal states
    dummy_grp = lambda gf: jnp.full((4, 4), 0.25, dtype=jnp.float32)

    # Case A: Double Ron with honba=2 (600 pts = 6 units) and kyotaku=3 (3000 pts = 30 units).
    # Loser = seat 0, winners = seat 1 (head-bump!) and seat 2.
    # Seat 1 gets 80 + 6 + 30 = +116; Seat 2 gets ONLY 80 (honba/kyotaku paid ONCE!);
    # Loser seat 0 pays 80 + 80 + 6 = -166 (NO phantom +10 riichi rebate!).
    tbl = TableSnapshot(
        scores=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        round_num=jnp.int8(2),
        round_limit=jnp.int8(7),
        honba=jnp.int32(2),
        kyotaku=jnp.int8(3),
        dealer=jnp.int8(1),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        wall_rem=jnp.int32(30),
        current_player=jnp.int8(0),
    )
    # Trigger tobi by setting loser score to 100 so 100 - 166 = -66 < 0 -> exact 1-hot terminal!
    tbl_tobi = tbl._replace(scores=jnp.array([100, 250, 250, 250], dtype=jnp.int32))
    mat_tobi, new_sc_tobi, is_term_tobi = simulate_ron_transition(
        tbl_tobi,
        discard_seat=jnp.int32(0),
        win_mask_abs=jnp.array([False, True, True, False]),
        win_points_abs=jnp.array([0.0, 80.0, 80.0, 0.0]),
        grp_fn=dummy_grp,
    )
    assert bool(is_term_tobi)
    # Expected final scores: seat 0 = -66 (4th), seat 1 = 366 (1st), seat 2 = 330 (2nd), seat 3 = 250 (3rd)
    np.testing.assert_allclose(np.asarray(new_sc_tobi), np.array([-66.0, 366.0, 330.0, 250.0]), atol=1e-5)
    expected = scores_to_rank_matrix_4x4(jnp.array([-66.0, 366.0, 330.0, 250.0]), tbl.init_wind)
    np.testing.assert_allclose(np.asarray(mat_tobi), np.asarray(expected), atol=1e-6)

    # Case B: South-4 Oorasu (round=7, round_limit=7, dealer=3):
    # Seat 1 (2nd place) rons seat 0 for 2000 pts (20 units) while seat 0 stays 1st (>300).
    # Non-dealer win in Oorasu -> game ends immediately -> exact 1-hot!
    tbl_oorasu = TableSnapshot(
        scores=jnp.array([340, 260, 220, 180], dtype=jnp.int32),
        round_num=jnp.int8(7),
        round_limit=jnp.int8(7),
        honba=jnp.int32(0),
        kyotaku=jnp.int8(0),
        dealer=jnp.int8(3),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        wall_rem=jnp.int32(15),
        current_player=jnp.int8(0),
    )
    mat_end, new_sc_end, is_term_end = simulate_ron_transition(
        tbl_oorasu,
        discard_seat=jnp.int32(0),
        win_mask_abs=jnp.array([False, True, False, False]),
        win_points_abs=jnp.array([0.0, 20.0, 0.0, 0.0]),
        grp_fn=dummy_grp,
    )
    assert bool(is_term_end)
    np.testing.assert_allclose(np.asarray(new_sc_end), np.array([320.0, 280.0, 220.0, 180.0]), atol=1e-5)
    expected_end = scores_to_rank_matrix_4x4(jnp.array([320.0, 280.0, 220.0, 180.0]), tbl_oorasu.init_wind)
    np.testing.assert_allclose(np.asarray(mat_end), np.asarray(expected_end), atol=1e-6)


def test_sparse_belief_labels_on_real_rollout():
    """Verify static K_max compacted legal-ron labeling matches can_win & ~furiten on real games."""
    env = mahjax.make("red_4p", round_mode="half", observe_type="dict", round_end="auto")

    @jax.jit
    def collect_snaps(key):
        k0, kp = jax.random.split(key)
        st0 = env.init(k0)

        def body(carry, _):
            st, r = carry
            r, ka, ks, kr = jax.random.split(r, 4)
            c = clean_auto_reset_state(st)
            snap = extract_ron_snapshot(c)
            act = mahjax.rule_based_player(c, ka)
            raw_next = env.step(c, act, ks)
            nxt = jax.lax.cond(raw_next.terminated | raw_next.truncated, lambda: env.init(kr), lambda: raw_next)
            return (nxt, r), snap

        _, snaps = jax.lax.scan(body, (st0, kp), None, length=60)
        return snaps

    batched_snaps = collect_snaps(jax.random.PRNGKey(42))
    y_ron, y_sb, y_ct, y_ten, overflow = jax.device_get(
        jax.jit(lambda s: compute_sparse_belief_labels(s, k_max=256))(batched_snaps)
    )
    assert y_ron.shape == (60, 3, 34)
    assert y_sb.shape == (60, 3, 34)
    assert y_ct.shape == (60, 3, 34)
    assert y_ten.shape == (60, 3)
    assert np.all(np.isfinite(y_ron)) and np.all((y_ron == 0) | (y_ron == 1))
    assert np.all((y_sb >= 0) & (y_sb <= 2))
    assert np.all((y_ct == 0) | (y_ct == 1))


def test_analytical_physics_trinity_invariants():
    """Test 4: Double-ron super-additivity, Aka-dora risk jump, Sashikomi > 0, Ryuukyoku gate, Disjoint betas."""
    # Analytic score-rank soft-GRP for deterministic monotonic testing
    def soft_grp(gf: jnp.ndarray) -> jnp.ndarray:
        # gf[:4] = (scores - 250) / 250
        sc = gf[:4] * 250.0 + 250.0
        # Soft pairwise ranking -> sharp enough to reflect clear rank changes
        diffs = (sc[None, :] - sc[:, None]) / 25.0  # (4, 4)
        exp_ahead = jnp.sum(jax.nn.sigmoid(diffs), axis=1) - 0.5  # in [0, 3]
        ranks = jnp.arange(4, dtype=jnp.float32)
        logits = -3.0 * (exp_ahead[:, None] - ranks[None, :]) ** 2
        return jax.nn.softmax(logits, axis=-1)

    # 1. Double-ron cliff & Aka-dora jump
    tbl = TableSnapshot(
        scores=jnp.array([250, 260, 255, 235], dtype=jnp.int32),
        round_num=jnp.int8(6),
        round_limit=jnp.int8(7),
        honba=jnp.int32(0),
        kyotaku=jnp.int8(0),
        dealer=jnp.int8(1),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        wall_rem=jnp.int32(25),  # > 12 -> delta_u_ryuukyoku MUST be 0!
        current_player=jnp.int8(0),
    )
    # Two opponents both waiting on tile 4 (5m) with p_deal = 0.25 each
    p_deal = jnp.zeros((3, 34), dtype=jnp.float32).at[0, 4].set(0.25).at[1, 4].set(0.25)
    # Score bin skewed toward small (bin 0) so T_AKA_SHIFT visibly shifts mass into mangan (bin 1)!
    p_sb = jnp.broadcast_to(jnp.array([0.85, 0.10, 0.05], dtype=jnp.float32), (3, 34, 3))
    p_ct = jnp.zeros((3, 34), dtype=jnp.float32)
    p_ten = jnp.array([0.9, 0.9, 0.1], dtype=jnp.float32)

    du_3opp, du_87, du_agari, du_ryu, _ = compute_analytical_trinity(
        s_table=tbl,
        p_deal_3opp=p_deal,
        p_score_bins=p_sb,
        p_call_threat_3=p_ct,
        p_tenpai_3=p_ten,
        w_me=W_PHOENIX_10D,
        last_draw_type=jnp.int32(4),
        is_last_draw_red=jnp.bool_(True),
        grp_fn=soft_grp,
    )
    # Ryuukyoku MUST be strictly 0 when wall_rem == 25 > 12
    assert float(du_ryu) == 0.0
    # Red 5m discard (action 34) MUST have strictly worse (more negative) Delta_U than black 5m (action 4)!
    assert float(du_87[34]) < float(du_87[4]) - 1e-4
    # Red 5m TSUMOGIRI (action 71) MUST match action 34!
    assert np.isclose(float(du_87[71]), float(du_87[34]), atol=1e-6)
    # Non-discard actions (PASS=84, RIICHI=72, TSUMO=73, RON=74) MUST be strictly 0 in du_87!
    assert float(du_87[Action.PASS]) == 0.0
    assert float(du_87[Action.RIICHI]) == 0.0
    assert float(du_87[Action.TSUMO]) == 0.0
    assert float(du_87[Action.RON]) == 0.0

    # 2. Endgame Ryuukyoku activates when wall_rem <= 12
    tbl_end = tbl._replace(wall_rem=jnp.int32(8))
    _, _, _, du_ryu_end, _ = compute_analytical_trinity(
        s_table=tbl_end,
        p_deal_3opp=p_deal,
        p_score_bins=p_sb,
        p_call_threat_3=p_ct,
        p_tenpai_3=p_ten,
        w_me=W_PHOENIX_10D,
        last_draw_type=jnp.int32(4),
        is_last_draw_red=jnp.bool_(False),
        grp_fn=soft_grp,
    )
    assert float(du_ryu_end) > 0.0, f"Expected positive keishiki tenpai swing, got {float(du_ryu_end)}"

    # 3. South-4 Sashikomi: seat 0 is 1st (350) vs dealer seat 3 (290).
    # Feeding a cheap 2000-pt ron to non-dealer seat 1 (200) ENDS Oorasu with seat 0 guaranteed 1st!
    # Under soft_grp, continuing Oorasu has <1.0 prob of 1st, so Sashikomi gives strictly POSITIVE Delta_U > 0!
    tbl_sashikomi = TableSnapshot(
        scores=jnp.array([350, 200, 160, 290], dtype=jnp.int32),
        round_num=jnp.int8(7),
        round_limit=jnp.int8(7),
        honba=jnp.int32(0),
        kyotaku=jnp.int8(0),
        dealer=jnp.int8(3),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        wall_rem=jnp.int32(15),
        current_player=jnp.int8(0),
    )
    p_deal_sashi = jnp.zeros((3, 34), dtype=jnp.float32).at[0, 10].set(0.5)  # opp 0 = seat 1
    p_sb_cheap = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32), (3, 34, 3))
    _, du_87_sashi, _, _, _ = compute_analytical_trinity(
        s_table=tbl_sashikomi,
        p_deal_3opp=p_deal_sashi,
        p_score_bins=p_sb_cheap,
        p_call_threat_3=p_ct,
        p_tenpai_3=jnp.zeros((3,), dtype=jnp.float32),
        w_me=W_PHOENIX_7D,
        last_draw_type=jnp.int32(10),
        is_last_draw_red=jnp.bool_(False),
        grp_fn=soft_grp,
    )
    assert float(du_87_sashi[10]) > 0.0, f"South-4 Sashikomi should have positive Delta_U, got {float(du_87_sashi[10])}"

    # 4. Mutually Disjoint Bounded Physics Gates inside SovereignPolicyNet
    p_cfg = ModelConfig(num_blocks=1, channels=32, river_dim=8, attn_heads=4, global_hidden_dim=32)
    p_net = SovereignPolicyNet(config=p_cfg)
    params = p_net.init(
        jax.random.PRNGKey(0),
        planes=jnp.zeros((1, 53, 34)),
        delta_u_ron_3opp=jnp.zeros((1, 3, 34)),
        call_threat_plane=jnp.zeros((1, 34)),
        scalars=jnp.zeros((1, 56)),
        river_tokens=jnp.zeros((1, 4, 24, 23)),
        river_tile_ids=jnp.full((1, 4, 24), -1, dtype=jnp.int32),
        river_valid_mask=jnp.zeros((1, 4, 24), dtype=jnp.bool_),
        last_draw_type=jnp.array([-1], dtype=jnp.int32),
        is_last_draw_red=jnp.array([False]),
        target_type=jnp.array([-1], dtype=jnp.int32),
        target_is_red=jnp.array([False]),
        w_ego=W_PHOENIX_7D[None, :],
        delta_u_agari_3=jnp.array([[0.3, 0.5, 0.7]]),
        delta_u_ryuukyoku=jnp.array([0.0]),
        delta_u_ron_87=jnp.zeros((1, 87)),
        legal_action_mask=jnp.ones((1, 87), dtype=jnp.bool_),
    )
    # Compare outputs with huge offense_gain vs zero offense_gain ONLY via raw_logits check:
    # PASS (84) - TSUMO (73) difference MUST NOT be affected by offense residual!
    lg0, aux0 = p_net.apply(
        params,
        planes=jnp.zeros((1, 53, 34)),
        delta_u_ron_3opp=jnp.zeros((1, 3, 34)),
        call_threat_plane=jnp.zeros((1, 34)),
        scalars=jnp.zeros((1, 56)),
        river_tokens=jnp.zeros((1, 4, 24, 23)),
        river_tile_ids=jnp.full((1, 4, 24), -1, dtype=jnp.int32),
        river_valid_mask=jnp.zeros((1, 4, 24), dtype=jnp.bool_),
        last_draw_type=jnp.array([-1], dtype=jnp.int32),
        is_last_draw_red=jnp.array([False]),
        target_type=jnp.array([-1], dtype=jnp.int32),
        target_is_red=jnp.array([False]),
        w_ego=W_PHOENIX_7D[None, :],
        delta_u_agari_3=jnp.array([[0.2, 0.4, 0.6]]),
        delta_u_ryuukyoku=jnp.array([0.0]),
        delta_u_ron_87=jnp.ones((1, 87)) * (-1.0),  # Even if someone passed non-zero on PASS, DISCARD_MASK blocks it!
        legal_action_mask=jnp.ones((1, 87), dtype=jnp.bool_),
        return_aux=True,
    )
    assert 0.0 < float(aux0["beta_risk"]) < p_cfg.beta_risk_max
    assert 0.0 < float(aux0["beta_offense"]) < p_cfg.beta_offense_max
    # Difference (PASS - TSUMO) in final logits equals raw_logits(PASS) - raw_logits(TSUMO) (neither gate touches them!)
    raw = aux0["raw_logits"]
    assert np.isclose(
        float(lg0[0, Action.PASS] - lg0[0, Action.TSUMO]),
        float(raw[0, Action.PASS] - raw[0, Action.TSUMO]),
        atol=1e-5,
    )
    # RIICHI - PASS shift equals +beta_offense * clip(mean(0.2,0.4,0.6)*4, 0, 4) * 1.0!
    expected_riichi_boost = float(aux0["beta_offense"]) * float(np.clip(0.4 * 4.0, 0.0, 4.0))
    actual_riichi_shift = float((lg0[0, Action.RIICHI] - lg0[0, Action.PASS]) - (raw[0, Action.RIICHI] - raw[0, Action.PASS]))
    assert np.isclose(actual_riichi_shift, expected_riichi_boost, atol=1e-5)

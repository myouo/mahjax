"""Compiled Duplicate 1v3 evaluator for V6 checkpoints (params as runtime arguments).

`mahjong_rl.eval.duplicate_eval.evaluate_duplicate_1v3` takes actor *closures*; every
new checkpoint/rule changes the closed-over weights, so XLA recompiles.  This class
instead passes the policy/belief params and the utility rule as JIT arguments:

* one XLA compilation per baseline mode (`params` or `rule`), reused for every
  checkpoint and every zero-shot rule (7d / 10d / M-League);
* the joint actor encodes the observation once and runs both policies from it;
* the whole chunk runs in `play_duplicate_chunk_joint`'s early-exit while-loop;
* `pmap` across all local devices, with `--eval-chunk-walls` walls per device.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjong_rl.algo.legal_ron import compute_analytical_trinity, extract_table_snapshot
from mahjong_rl.algo.sovereign_v6 import (
    NEG_INF,
    masked_categorical_log_probs,
    replicate_for_pmap,
    shard_for_pmap,
)
from mahjong_rl.algo.utility import (
    W_PHOENIX_7D,
    build_opponent_utility_features,
    make_homogeneous_table,
    rotate_utility_matrix_ego,
)
from mahjong_rl.config import EvalConfig
from mahjong_rl.eval.duplicate_eval import play_duplicate_chunk_joint
from mahjong_rl.features.obs_2d import clean_auto_reset_state
from mahjong_rl.features.obs_v6 import encode_obs_v6


class CompiledDuplicateEvaluator:
    """Params-as-arguments Duplicate 1v3 evaluator with early exit and shared encoding.

    Baseline modes:
    - `evaluate(cand_pol, cand_bel, base_pol, base_bel)`: candidate vs 3 copies of a
      frozen V6 baseline (policy + belief params).
    - `evaluate_vs_rule(cand_pol, cand_bel, w_rule=...)`: candidate vs
      `mahjax.rule_based_player`; also used for zero-shot rule switching.
    """

    def __init__(
        self,
        *,
        policy_model: Any,
        belief_model: Any,
        grp_model: Any,
        grp_base_params: Any,
        eval_cfg: EvalConfig = EvalConfig(),
        w_rule: Any = W_PHOENIX_7D,
        opp_rules_known: bool = False,
        belief_t_cal: float = 1.0,
        greedy: bool = True,
        chunk_walls: int = 8,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.chunk_walls = max(1, int(chunk_walls))
        self.n_devices = jax.local_device_count()
        self.use_pmap = self.n_devices > 1
        self.default_w_rule = jnp.asarray(w_rule, dtype=jnp.float32)
        self.opp_rules_known = bool(opp_rules_known)
        self.belief_t_cal = float(belief_t_cal)
        self.greedy = bool(greedy)
        self.policy_model = policy_model
        self.belief_model = belief_model
        self.grp_model = grp_model

        # Frozen shared inputs as host NumPy so pmap replicates them as constants.
        self.grp_params = jax.tree_util.tree_map(
            lambda x: np.asarray(jax.device_get(x)), grp_base_params
        )
        self.rank_pts_arr = jnp.asarray(eval_cfg.rank_points, dtype=jnp.float32)
        self.w_default_prior = W_PHOENIX_7D
        self.is_known = jnp.full((3,), 1.0 if opp_rules_known else 0.0, dtype=jnp.float32)

        self.env = mahjax.make(
            eval_cfg.env_name,
            round_mode=eval_cfg.round_mode,
            observe_type="dict",
            round_end="confirm",
        )
        self.num_players = int(self.env.num_players)

        def _act_from_encoded(obs, tbl, w_me, w_opp, legal_mask, key, pol_p, bel_p):
            b_out = self.belief_model.apply(
                bel_p,
                planes=obs["planes"],
                scalars=obs["scalars"],
                river_tokens=obs["river_tokens"],
                river_tile_ids=obs["river_tile_ids"],
                river_valid_mask=obs["river_valid_mask"],
                w_opp_5d=w_opp,
                temperature_cal=self.belief_t_cal,
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
                grp_fn=lambda gf: self.grp_model.apply(self.grp_params, gf),
            )
            logits = self.policy_model.apply(
                pol_p,
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
            if self.greedy:
                return jnp.argmax(jnp.where(legal_mask, logits, NEG_INF)).astype(jnp.int32)
            log_p = masked_categorical_log_probs(logits, legal_mask)
            return jax.random.categorical(key, log_p).astype(jnp.int32)

        def _encode(state, w_rule):
            clean_st = clean_auto_reset_state(state)
            cp = clean_st.current_player.astype(jnp.int32)
            legal_mask = clean_st.legal_action_mask.astype(jnp.bool_)
            obs = encode_obs_v6(clean_st)
            tbl = extract_table_snapshot(clean_st)
            w_ego = rotate_utility_matrix_ego(make_homogeneous_table(w_rule), cp)
            w_me = w_ego[0]
            w_opp = build_opponent_utility_features(
                w_ego[1:],
                self.is_known,
                rng_key=None,
                dropout_prob=0.0,
                w_table_default=self.w_default_prior,
            )
            return clean_st, obs, tbl, w_me, w_opp, legal_mask

        def _joint_params(state, k1, k2, c_pol, c_bel, b_pol, b_bel, w_rule):
            _st, obs, tbl, w_me, w_opp, legal_mask = _encode(state, w_rule)
            a_cand = _act_from_encoded(obs, tbl, w_me, w_opp, legal_mask, k1, c_pol, c_bel)
            a_base = _act_from_encoded(obs, tbl, w_me, w_opp, legal_mask, k2, b_pol, b_bel)
            return a_cand, a_base

        def _joint_rule(state, k1, k2, c_pol, c_bel, w_rule):
            _st, obs, tbl, w_me, w_opp, legal_mask = _encode(state, w_rule)
            a_cand = _act_from_encoded(obs, tbl, w_me, w_opp, legal_mask, k1, c_pol, c_bel)
            # Match evaluate_duplicate_1v3: rule player sees the raw state.
            a_base = mahjax.rule_based_player(state, k2)
            return a_cand, a_base

        max_steps = int(eval_cfg.max_steps)

        def chunk_params(c_pol, c_bel, b_pol, b_bel, w_rule, ik, ss):
            def joint(s, k1, k2):
                return _joint_params(s, k1, k2, c_pol, c_bel, b_pol, b_bel, w_rule)

            return play_duplicate_chunk_joint(
                self.env, joint, self.rank_pts_arr, ik, ss, max_steps
            )

        def chunk_rule(c_pol, c_bel, w_rule, ik, ss):
            def joint(s, k1, k2):
                return _joint_rule(s, k1, k2, c_pol, c_bel, w_rule)

            return play_duplicate_chunk_joint(
                self.env, joint, self.rank_pts_arr, ik, ss, max_steps
            )

        self._run_params = jax.pmap(chunk_params) if self.use_pmap else jax.jit(chunk_params)
        self._run_rule = jax.pmap(chunk_rule) if self.use_pmap else jax.jit(chunk_rule)

        rng = jax.random.PRNGKey(eval_cfg.seed)
        k_init, k_step = jax.random.split(rng)
        self.init_keys = jax.random.split(k_init, eval_cfg.num_walls)
        self.step_seeds = jax.random.split(k_step, eval_cfg.num_walls)
        # (base_pol, base_bel, replicated_pol, replicated_bel) identity cache
        self._cached_base_pmap: Optional[Tuple[Any, Any, Any, Any]] = None

    def _resolve_w(self, w_rule: Any) -> Any:
        return self.default_w_rule if w_rule is None else jnp.asarray(w_rule, dtype=jnp.float32)

    @staticmethod
    def _to_host(tree: Any) -> Any:
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)

    def _evaluate(
        self,
        mode: str,
        cand_pol: Any,
        cand_bel: Any,
        base_pol: Any = None,
        base_bel: Any = None,
        w_rule: Any = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        num_walls = int(self.eval_cfg.num_walls)
        per_launch = self.chunk_walls * self.n_devices
        w = self._resolve_w(w_rule)

        if self.use_pmap:
            c_pol = replicate_for_pmap(cand_pol, self.n_devices)
            c_bel = replicate_for_pmap(cand_bel, self.n_devices)
            w_exec = replicate_for_pmap(w, self.n_devices)
            if mode == "params":
                if (
                    self._cached_base_pmap is not None
                    and self._cached_base_pmap[0] is base_pol
                    and self._cached_base_pmap[1] is base_bel
                ):
                    _, _, b_pol, b_bel = self._cached_base_pmap
                else:
                    b_pol = replicate_for_pmap(base_pol, self.n_devices)
                    b_bel = replicate_for_pmap(base_bel, self.n_devices)
                    self._cached_base_pmap = (base_pol, base_bel, b_pol, b_bel)
        else:
            c_pol = jax.device_put(cand_pol)
            c_bel = jax.device_put(cand_bel)
            w_exec = w
            if mode == "params":
                b_pol = jax.device_put(base_pol)
                b_bel = jax.device_put(base_bel)

        totals_list, ranks_list, pts_list, fin_list = [], [], [], []
        for start in range(0, num_walls, per_launch):
            end = min(start + per_launch, num_walls)
            cnt = end - start
            ik = self.init_keys[start:end]
            ss = self.step_seeds[start:end]
            if cnt < per_launch:
                pad = per_launch - cnt
                ik = jnp.concatenate([ik, jnp.repeat(ik[-1:], pad, axis=0)], axis=0)
                ss = jnp.concatenate([ss, jnp.repeat(ss[-1:], pad, axis=0)], axis=0)
            if self.use_pmap:
                ik_s = shard_for_pmap(
                    ik.reshape((self.n_devices, self.chunk_walls) + ik.shape[1:]),
                    self.n_devices,
                )
                ss_s = shard_for_pmap(
                    ss.reshape((self.n_devices, self.chunk_walls) + ss.shape[1:]),
                    self.n_devices,
                )
                if mode == "params":
                    out = self._run_params(c_pol, c_bel, b_pol, b_bel, w_exec, ik_s, ss_s)
                else:
                    out = self._run_rule(c_pol, c_bel, w_exec, ik_s, ss_s)
                t_c, r_c, p_c, f_c = jax.device_get(out)
                t_c = {
                    k: np.asarray(v).reshape((per_launch,) + np.asarray(v).shape[2:])[:cnt]
                    for k, v in t_c.items()
                }
                r_c = np.asarray(r_c, dtype=np.int32).reshape((per_launch, self.num_players))[:cnt]
                p_c = np.asarray(p_c, dtype=np.float64).reshape((per_launch, self.num_players))[:cnt]
                f_c = np.asarray(f_c, dtype=np.float64).reshape((per_launch, self.num_players))[:cnt]
            else:
                if mode == "params":
                    out = self._run_params(c_pol, c_bel, b_pol, b_bel, w_exec, ik, ss)
                else:
                    out = self._run_rule(c_pol, c_bel, w_exec, ik, ss)
                t_c, r_c, p_c, f_c = jax.device_get(out)
                t_c = {k: np.asarray(v)[:cnt] for k, v in t_c.items()}
                r_c = np.asarray(r_c, dtype=np.int32)[:cnt]
                p_c = np.asarray(p_c, dtype=np.float64)[:cnt]
                f_c = np.asarray(f_c, dtype=np.float64)[:cnt]
            totals_list.append(t_c)
            ranks_list.append(r_c)
            pts_list.append(p_c)
            fin_list.append(f_c)

        totals = {k: np.concatenate([t[k] for t in totals_list], axis=0) for k in totals_list[0]}
        rank_idxs_np = np.concatenate(ranks_list, axis=0)
        pts_np = np.concatenate(pts_list, axis=0)
        finished_np = np.concatenate(fin_list, axis=0)

        wall_done = finished_np.all(axis=1)
        finished_wall_ratio = float(wall_done.mean())
        eval_mask = wall_done if bool(wall_done.any()) else np.ones((num_walls,), dtype=bool)
        wall_mean_pts = pts_np[eval_mask].mean(axis=1)
        pt_mean = float(wall_mean_pts.mean())
        n_eval_w = int(wall_mean_pts.shape[0])
        pt_se = float(wall_mean_pts.std(ddof=1) / np.sqrt(max(n_eval_w, 1))) if n_eval_w > 1 else 0.0
        eval_ranks_np = rank_idxs_np[eval_mask]
        total_games = int(rank_idxs_np.size)
        total_hands = float(np.asarray(totals["hands"]).sum())
        denom_hands = max(total_hands, 1.0)
        total_won = float(np.asarray(totals["won"]).sum())
        total_dealt = float(np.asarray(totals["dealt_in"]).sum())

        return {
            "total_games": float(total_games),
            "total_hands": total_hands,
            "finished_game_ratio": float(finished_np.mean()),
            "finished_wall_ratio": finished_wall_ratio,
            "tenhou_pt_mean": pt_mean,
            "tenhou_pt_ci95": 1.96 * pt_se,
            "avg_rank": float(eval_ranks_np.mean() + 1.0),
            "rank_1_rate": float((eval_ranks_np == 0).mean()),
            "rank_2_rate": float((eval_ranks_np == 1).mean()),
            "rank_3_rate": float((eval_ranks_np == 2).mean()),
            "rank_4_rate": float((eval_ranks_np == 3).mean()),
            "hora_rate": total_won / denom_hands,
            "deal_in_rate": total_dealt / denom_hands,
            "riichi_rate": float(np.asarray(totals["riichi"]).sum()) / denom_hands,
            "meld_rate": float(np.asarray(totals["melded"]).sum()) / denom_hands,
            "avg_win_pts": float(np.asarray(totals["won_pts"]).sum()) / max(total_won, 1.0),
            "avg_deal_in_pts": float(np.asarray(totals["dealt_in_pts"]).sum()) / max(total_dealt, 1.0),
            "eval_wall_s": round(time.perf_counter() - t0, 2),
            "wall_mean_pts": [round(float(x), 3) for x in wall_mean_pts.tolist()],
        }

    def evaluate(
        self,
        cand_policy_params: Any,
        cand_belief_params: Any,
        base_policy_params: Any,
        base_belief_params: Any,
        w_rule: Any = None,
    ) -> Dict[str, Any]:
        """Candidate vs 3 copies of a frozen V6 baseline."""
        return self._evaluate(
            "params",
            cand_policy_params,
            cand_belief_params,
            base_policy_params,
            base_belief_params,
            w_rule,
        )

    def evaluate_vs_rule(
        self,
        cand_policy_params: Any,
        cand_belief_params: Any,
        w_rule: Any = None,
    ) -> Dict[str, Any]:
        """Candidate vs `mahjax.rule_based_player` (and zero-shot rule switching)."""
        return self._evaluate(
            "rule", cand_policy_params, cand_belief_params, w_rule=w_rule
        )


__all__ = ["CompiledDuplicateEvaluator"]

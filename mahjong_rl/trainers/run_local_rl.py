"""Clean Sovereign-Mahjong V6 local RL retraining & paired Duplicate 1v3 evaluation runner.

Loads unpoisoned Phase 0 checkpoints (`bc_phase0c.pkl`, `belief_phase0a.pkl`,
`grp_phase0b.pkl`), trains Phase 1 (Oracle warmup) + Phase 2 (R-NaD/PPO) via
`train_rl_v6` using the fixed riichi/belief-safe defaults, periodically evaluates
against frozen BC via Duplicate 1v3, tracks `rl_best.pkl` + `rl_eval_history.json`,
and saves `rl_phase2.pkl`.

Usage:
    /opt/miniconda3/envs/usual/bin/python -m mahjong_rl.trainers.run_local_rl \
        --ckpt-dir checkpoints/sovereign_v6 --num-updates 500
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjong_rl.algo.sovereign_v6 import replicate_for_pmap, shard_for_pmap
from mahjong_rl.algo.utility import W_PHOENIX_7D
from mahjong_rl.config import (
    BeliefConfig,
    EvalConfig,
    GRPConfig,
    ModelConfig,
    OracleConfig,
    RLConfig,
    RNaDConfig,
)
from mahjong_rl.eval.duplicate_eval import play_duplicate_chunk
from mahjong_rl.eval.mjai_bridge import make_v6_policy_actor
from mahjong_rl.jax_runtime import enable_persistent_compilation_cache
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.grp import GRPNet
from mahjong_rl.trainers.train_rl import train_rl_v6


def _atomic_dump_pkl(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(payload), f)
    os.replace(tmp, path)
    return path


class _CompiledDuplicateVsBCEvaluator:
    """Single-compiled Duplicate 1v3 evaluator for periodic checkpoints without recompile."""

    def __init__(
        self,
        *,
        policy_model: Any,
        belief_model: BeliefNet,
        grp_model: GRPNet,
        grp_params: Any,
        belief_t_cal: float,
        eval_cfg: EvalConfig,
        chunk_walls: int = 16,
    ) -> None:
        self.policy_model = policy_model
        self.belief_model = belief_model
        self.grp_model = grp_model
        self.n_devices = jax.local_device_count()
        self._use_pmap = self.n_devices > 1
        # On multi-GPU keep the frozen GRP weights on host so pmap replicates them as constants.
        self.grp_params = jax.device_get(grp_params) if self._use_pmap else jax.device_put(grp_params)
        self.belief_t_cal = float(belief_t_cal)
        self.eval_cfg = eval_cfg
        self.chunk_walls = max(1, int(chunk_walls))

        self.env = mahjax.make(
            eval_cfg.env_name,
            round_mode=eval_cfg.round_mode,
            observe_type="dict",
            round_end="confirm",
        )
        self.num_players = self.env.num_players
        rank_pts_arr = jnp.asarray(eval_cfg.rank_points, dtype=np.float32)

        def eval_chunk(cand_pol, cand_bel, base_pol, base_bel, init_keys, step_seeds):
            cand_actor = make_v6_policy_actor(
                policy_model,
                cand_pol,
                belief_model,
                cand_bel,
                grp_model,
                self.grp_params,
                w_rule=W_PHOENIX_7D,
                opp_rules_known=False,
                belief_t_cal=self.belief_t_cal,
                greedy=True,
            )
            base_actor = make_v6_policy_actor(
                policy_model,
                base_pol,
                belief_model,
                base_bel,
                grp_model,
                self.grp_params,
                w_rule=W_PHOENIX_7D,
                opp_rules_known=False,
                belief_t_cal=self.belief_t_cal,
                greedy=True,
            )
            # One flat batched while-loop over (walls x 4 seat rotations) with early exit.
            return play_duplicate_chunk(
                self.env,
                cand_actor,
                base_actor,
                rank_pts_arr,
                init_keys,
                step_seeds,
                int(eval_cfg.max_steps),
            )

        self._run = jax.pmap(eval_chunk) if self._use_pmap else jax.jit(eval_chunk)
        rng = jax.random.PRNGKey(eval_cfg.seed)
        k_init, k_step = jax.random.split(rng)
        self.init_keys = jax.random.split(k_init, eval_cfg.num_walls)
        self.step_seeds = jax.random.split(k_step, eval_cfg.num_walls)

    def evaluate(self, cand_pol: Any, cand_bel: Any, base_pol: Any, base_bel: Any) -> Dict[str, float]:
        t0 = time.perf_counter()
        if self._use_pmap:
            c_p = replicate_for_pmap(cand_pol, self.n_devices)
            c_b = replicate_for_pmap(cand_bel, self.n_devices)
            b_p = replicate_for_pmap(base_pol, self.n_devices)
            b_b = replicate_for_pmap(base_bel, self.n_devices)
        else:
            c_p = jax.device_put(cand_pol)
            c_b = jax.device_put(cand_bel)
            b_p = jax.device_put(base_pol)
            b_b = jax.device_put(base_bel)

        totals_list, ranks_list, pts_list, fin_list = [], [], [], []
        num_walls = int(self.eval_cfg.num_walls)
        per_launch = self.chunk_walls * self.n_devices
        for start in range(0, num_walls, per_launch):
            end = min(start + per_launch, num_walls)
            cnt = end - start
            ik_slice = self.init_keys[start:end]
            ss_slice = self.step_seeds[start:end]
            if cnt < per_launch:
                pad = per_launch - cnt
                ik_slice = jnp.concatenate([ik_slice, jnp.repeat(ik_slice[-1:], pad, axis=0)], axis=0)
                ss_slice = jnp.concatenate([ss_slice, jnp.repeat(ss_slice[-1:], pad, axis=0)], axis=0)
            if self._use_pmap:
                ik = shard_for_pmap(
                    ik_slice.reshape((self.n_devices, self.chunk_walls) + ik_slice.shape[1:]),
                    self.n_devices,
                )
                ss = shard_for_pmap(
                    ss_slice.reshape((self.n_devices, self.chunk_walls) + ss_slice.shape[1:]),
                    self.n_devices,
                )
                t_c, r_c, p_c, f_c = jax.device_get(self._run(c_p, c_b, b_p, b_b, ik, ss))
                t_c = {
                    k: np.asarray(v).reshape((per_launch,) + np.asarray(v).shape[2:])[:cnt]
                    for k, v in t_c.items()
                }
                r_c = np.asarray(r_c, dtype=np.int32).reshape((per_launch, self.num_players))[:cnt]
                p_c = np.asarray(p_c, dtype=np.float64).reshape((per_launch, self.num_players))[:cnt]
                f_c = np.asarray(f_c, dtype=np.float64).reshape((per_launch, self.num_players))[:cnt]
            else:
                t_c, r_c, p_c, f_c = jax.device_get(
                    self._run(c_p, c_b, b_p, b_b, ik_slice, ss_slice)
                )
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
        total_hands = float(totals["hands"].sum())
        denom_hands = max(total_hands, 1.0)
        total_won = float(totals["won"].sum())
        total_dealt = float(totals["dealt_in"].sum())
        return {
            "total_games": float(rank_idxs_np.size),
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
            "riichi_rate": float(totals["riichi"].sum()) / denom_hands,
            "meld_rate": float(totals["melded"].sum()) / denom_hands,
            "eval_wall_s": round(time.perf_counter() - t0, 2),
        }


def main(argv: Optional[List[str]] = None) -> None:
    enable_persistent_compilation_cache()
    parser = argparse.ArgumentParser(description="Clean Sovereign-Mahjong V6 RL trainer from Phase 0 checkpoints.")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/sovereign_v6"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (defaults to --ckpt-dir)")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Parallel games per rollout (GPU batch). Raise until VRAM is the limit.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=256,
        help="Rollout horizon per env; transitions/update = num_envs * num_steps.",
    )
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--warmup-updates", type=int, default=80)
    parser.add_argument("--outer-reset-steps", type=int, default=0)
    parser.add_argument("--pfsp-snapshot-interval", type=int, default=200)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--policy-lr", type=float, default=3e-5)
    parser.add_argument("--alpha-rnad", type=float, default=0.12)
    parser.add_argument("--rho-mix-bc", type=float, default=0.10)
    parser.add_argument("--rho-mix-uniform", type=float, default=0.005)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--lambda-intra", type=float, default=1.0)
    parser.add_argument("--lambda-inter", type=float, default=0.0)
    parser.add_argument("--mask-unterminated-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phoenix-7d-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pfsp-frac", type=float, default=0.625)
    parser.add_argument("--opp-temperature", type=float, default=0.35)
    parser.add_argument("--oracle-replay-updates", type=int, default=6)
    parser.add_argument("--freeze-belief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-walls", type=int, default=16)
    parser.add_argument(
        "--eval-chunk-walls",
        type=int,
        default=16,
        help="Walls per device per launch (4*this lanes/GPU). Lower it if VRAM OOMs.",
    )
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for req_name in ("bc_phase0c.pkl", "belief_phase0a.pkl", "grp_phase0b.pkl"):
        if not (ckpt_dir / req_name).exists():
            raise FileNotFoundError(f"Missing required unpoisoned Phase 0 checkpoint: {ckpt_dir / req_name}")

    with open(ckpt_dir / "bc_phase0c.pkl", "rb") as f:
        d_bc = pickle.load(f)
    with open(ckpt_dir / "belief_phase0a.pkl", "rb") as f:
        d_bel = pickle.load(f)
    with open(ckpt_dir / "grp_phase0b.pkl", "rb") as f:
        d_grp = pickle.load(f)

    model_cfg = ModelConfig(**d_bc["model_config"])
    bel_cfg = BeliefConfig(**d_bel["config"])
    grp_cfg = GRPConfig(**d_grp["config"])
    bc_params = jax.device_put(d_bc["params"])
    bel_params = jax.device_put(d_bel["params"])
    grp_params = jax.device_put(d_grp["params"])
    belief_t_cal = float(d_bel.get("t_cal", d_bc.get("t_cal", 1.0)))

    print(
        f"[Phase 0 Verified] BC top1={d_bc.get('val', {}).get('val_bc_top1_acc', 0.0):.4f} "
        f"ent={d_bc.get('val', {}).get('val_bc_entropy', 0.0):.3f} | "
        f"Belief AUC={d_bel.get('val', {}).get('val_deal_roc_auc', 0.0):.4f} "
        f"cal={d_bel.get('val', {}).get('val_deal_calibration_ratio', 0.0):.3f} | "
        f"GRP CE={d_grp.get('val', {}).get('val_grp_ce', 0.0):.4f}",
        flush=True,
    )

    n_devices = jax.local_device_count() if args.multi_gpu else 1
    n_per_dev = (args.num_envs // max(n_devices, 1)) * args.num_steps
    rl_cfg = RLConfig(
        env_name="red_4p",
        round_mode="half",
        model=model_cfg,
        belief=bel_cfg,
        grp=grp_cfg,
        oracle=OracleConfig(replay_updates=int(args.oracle_replay_updates)),
        rnad=RNaDConfig(
            lambda_intra=float(args.lambda_intra),
            lambda_inter=float(args.lambda_inter),
            mask_unterminated_tail=bool(args.mask_unterminated_tail),
            alpha_rnad=float(args.alpha_rnad),
            outer_reset_steps=int(args.outer_reset_steps),
            pfsp_snapshot_interval=int(args.pfsp_snapshot_interval),
            reset_warmup_steps=30,
            rho_mix_bc=float(args.rho_mix_bc),
            rho_mix_uniform=float(args.rho_mix_uniform),
            entropy_coef=float(args.entropy_coef),
            policy_lr=float(args.policy_lr),
            phoenix_7d_only=bool(args.phoenix_7d_only),
            pfsp_frac=float(args.pfsp_frac),
            opp_temperature=float(args.opp_temperature),
            freeze_belief=bool(args.freeze_belief),
            ron_k_max=max(8192, 2 * n_per_dev),
            warmup_updates=int(args.warmup_updates),
            audit_interval=25,
        ),
        num_envs=int(args.num_envs),
        num_steps=int(args.num_steps),
        num_updates=int(args.num_updates),
        update_epochs=int(args.update_epochs),
        minibatch_size=int(args.minibatch_size),
        seed=int(args.seed),
    )

    resume_dst = out_dir / "rl_resume_state.pkl"
    resume_in = resume_dst if (args.resume and resume_dst.exists()) else None
    eval_hist_path = out_dir / "rl_eval_history.json"
    eval_records: List[Dict[str, Any]] = []
    if args.resume and eval_hist_path.exists():
        try:
            eval_records = json.loads(eval_hist_path.read_text(encoding="utf-8"))
        except Exception:
            eval_records = []
    best_state = {"pt": float("-inf"), "upd": None}
    if args.resume and (out_dir / "rl_best.pkl").exists():
        try:
            with open(out_dir / "rl_best.pkl", "rb") as f:
                db = pickle.load(f)
            best_state["upd"] = db.get("completed_updates")
            ev = db.get("eval_vs_bc") or {}
            if "tenhou_pt_mean" in ev:
                best_state["pt"] = float(ev["tenhou_pt_mean"])
        except Exception:
            pass

    evaluator: Optional[_CompiledDuplicateVsBCEvaluator] = None

    def _on_checkpoint(completed_upd: int, saved_resume_path: Path) -> None:
        nonlocal evaluator
        if args.eval_every <= 0:
            return
        if int(completed_upd) <= int(args.warmup_updates) and int(completed_upd) < int(args.num_updates):
            return
        if (completed_upd % int(args.eval_every) != 0) and (completed_upd < int(args.num_updates)):
            return
        if any(int(r.get("update", -1)) == int(completed_upd) for r in eval_records):
            return

        with open(saved_resume_path, "rb") as f:
            d = pickle.load(f)
        cand_pol = d.get("params_pi", d.get("policy_params"))
        cand_bel = d.get("params_b_ema", d.get("belief_ema_params"))
        hist = d.get("history") or []
        train_last = hist[-1] if hist else {}

        from mahjong_rl.models.equivariant import SovereignPolicyNet

        if evaluator is None:
            eval_cfg = EvalConfig(
                env_name="red_4p",
                round_mode="half",
                num_walls=int(args.eval_walls),
                max_steps=2400,
                seed=int(args.eval_seed),
            )
            evaluator = _CompiledDuplicateVsBCEvaluator(
                policy_model=SovereignPolicyNet(config=model_cfg),
                belief_model=BeliefNet(config=bel_cfg),
                grp_model=GRPNet(config=grp_cfg),
                grp_params=grp_params,
                belief_t_cal=belief_t_cal,
                eval_cfg=eval_cfg,
                chunk_walls=int(args.eval_chunk_walls),
            )

        ev = evaluator.evaluate(cand_pol, cand_bel, bc_params, bel_params)
        pt_mean = float(ev["tenhou_pt_mean"])
        ci95 = float(ev["tenhou_pt_ci95"])
        is_best = (best_state["upd"] is None) or (pt_mean > best_state["pt"])
        if is_best:
            best_state["pt"] = pt_mean
            best_state["upd"] = int(completed_upd)
            _atomic_dump_pkl(
                out_dir / "rl_best.pkl",
                {
                    "completed_updates": int(completed_upd),
                    "model_config": asdict(model_cfg),
                    "belief_config": asdict(bel_cfg),
                    "grp_config": asdict(grp_cfg),
                    "policy_params": cand_pol,
                    "belief_ema_params": cand_bel,
                    "grp_base_params": jax.device_get(grp_params),
                    "t_cal": belief_t_cal,
                    "eval_vs_bc": ev,
                },
            )

        rec = {
            "update": int(completed_upd),
            "pt_vs_bc": pt_mean,
            "ci95_vs_bc": ci95,
            "avg_rank": float(ev["avg_rank"]),
            "rank_1_rate": float(ev["rank_1_rate"]),
            "rank_2_rate": float(ev["rank_2_rate"]),
            "rank_3_rate": float(ev["rank_3_rate"]),
            "rank_4_rate": float(ev["rank_4_rate"]),
            "hora_rate": float(ev["hora_rate"]),
            "deal_in_rate": float(ev["deal_in_rate"]),
            "riichi_rate": float(ev["riichi_rate"]),
            "meld_rate": float(ev["meld_rate"]),
            "p_riichi_legal": float(train_last.get("p_riichi_legal", 0.0)),
            "bc_kl": float(train_last.get("bc_kl", 0.0)),
            "riichi_adv_gap": float(train_last.get("riichi_adv_gap", 0.0)),
            "riichi_pg": float(train_last.get("riichi_pg", 0.0)),
            "oracle_r2": float(train_last.get("oracle_r2", 0.0)),
            "oracle_r2_late": float(train_last.get("oracle_r2_late", 0.0)),
            "belief_pred_mean": float(train_last.get("belief_pred_mean", 0.0)),
            "is_best": bool(is_best),
            "eval_wall_s": float(ev["eval_wall_s"]),
        }
        eval_records.append(rec)
        eval_hist_path.write_text(json.dumps(eval_records, indent=2), encoding="utf-8")
        fin_w = float(ev.get("finished_wall_ratio", 1.0))
        fin_str = f" finW={fin_w*100:.0f}%" if fin_w < 1.0 else ""
        print(
            f"[EVAL upd {completed_upd:4d}] vs 3x BC ({int(ev['total_games'])}g): "
            f"pt={pt_mean:+.2f}±{ci95:.2f} avgR={ev['avg_rank']:.2f} "
            f"R4={ev['rank_4_rate']*100:.1f}% deal={ev['deal_in_rate']*100:.2f}% "
            f"hora={ev['hora_rate']*100:.2f}% rii={ev['riichi_rate']*100:.2f}% "
            f"p_rii_legal={rec['p_riichi_legal']:.3f}{fin_str} best={'*' if is_best else ' '} ({ev['eval_wall_s']:.1f}s)",
            flush=True,
        )

    res = train_rl_v6(
        rl_cfg,
        policy_params_init=bc_params,
        belief_params_init=bel_params,
        belief_t_cal=belief_t_cal,
        grp_params_init=grp_params,
        enable_self_healing=True,
        multi_gpu=bool(args.multi_gpu),
        verbose=True,
        log_every=int(args.log_every),
        resume_path=resume_in,
        checkpoint_path=resume_dst,
        checkpoint_every=int(args.checkpoint_every),
        on_checkpoint=_on_checkpoint if args.eval_every > 0 else None,
    )

    _atomic_dump_pkl(
        out_dir / "rl_phase2.pkl",
        {
            "completed_updates": int(args.num_updates),
            "model_config": asdict(model_cfg),
            "belief_config": asdict(bel_cfg),
            "grp_config": asdict(grp_cfg),
            "policy_params": res.policy_params,
            "belief_ema_params": res.belief_ema_params,
            "oracle_params": res.oracle_params,
            "t_cal": belief_t_cal,
            "history": res.history,
        },
    )
    print(f"[DONE] Saved clean RL checkpoint -> {out_dir / 'rl_phase2.pkl'}", flush=True)


if __name__ == "__main__":
    main()

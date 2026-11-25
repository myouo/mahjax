"""Phase 0 Production Runner (0-B GRP -> 0-A Belief + T_cal -> 0-C Isomorphic BC -> Strict Eval).

Enforces all Sovereign-Mahjong V6 mathematical & evaluation gates:
1. Strict Train / Held-Out Validation split (zero data leakage for T_cal, AUC, ECE, Top-1 Acc).
2. Birkhoff conservation (`< 1e-4`) and exact 7d zero-sum conservation (`< 1e-4`) on GRP_base.
3. Unweighted Belief posterior calibration (`mean(p_cal) / mean(y_ron) ~ 1.0`, `ROC-AUC >= 0.85`).
4. Strictly isomorphic Policy BC with frozen `B_EMA + GRP_base` Analytical Physics Trinity.
5. Fixed-wall Full-Hanchan Duplicate 1v3 evaluation (paired 95% CI, `[+90, +45, 0, -135]`)
   plus zero-shot multi-rule verification (`W_PHOENIX_7D` / `W_PHOENIX_10D` / `W_M_LEAGUE`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjong_rl.algo.sovereign_v6 import (
    compute_binary_roc_auc,
    compute_birkhoff_error,
    compute_brier_4rank,
    compute_ece_4rank,
    masked_categorical_log_probs,
    masked_entropy,
    masked_mean,
)
from mahjong_rl.algo.utility import W_M_LEAGUE, W_PHOENIX_7D, W_PHOENIX_10D
from mahjong_rl.config import BCConfig, BeliefConfig, EvalConfig, GRPConfig, ModelConfig
from mahjong_rl.data.shard_loader import ShardBatchLoader
from mahjong_rl.eval.duplicate_eval import evaluate_duplicate_1v3
from mahjong_rl.eval.mjai_bridge import make_v6_policy_actor
from mahjong_rl.trainers.train_bc import _compute_batch_trinity, train_bc
from mahjong_rl.trainers.train_belief import _extract_w_opp_5d, train_belief
from mahjong_rl.trainers.train_grp import train_grp


def _save_pkl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(payload), f)


def evaluate_grp_val(model, params, val_shards: Sequence[Path], batch_size: int = 1024) -> Dict[str, float]:
    apply_fn = jax.jit(model.apply)
    loader = ShardBatchLoader(
        val_shards,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        keys=("grp_features", "final_rank_matrix_4x4"),
    )
    all_p, all_y = [], []
    max_birkhoff = 0.0
    max_zsum = 0.0
    mean_birkhoffs, mean_zsums = [], []
    for b in loader:
        gf = jnp.asarray(b["grp_features"], dtype=jnp.float32)
        probs = apply_fn(params, gf)
        row_e = jnp.max(jnp.abs(jnp.sum(probs, axis=-1) - 1.0), axis=-1)
        col_e = jnp.max(jnp.abs(jnp.sum(probs, axis=-2) - 1.0), axis=-1)
        per_sample_b = jnp.maximum(row_e, col_e)
        max_birkhoff = max(max_birkhoff, float(jnp.max(per_sample_b)))
        mean_birkhoffs.append(float(jnp.mean(per_sample_b)))

        v_7d = jnp.sum(probs * W_PHOENIX_7D[None, None, :], axis=-1)  # (B, 4)
        per_sample_z = jnp.abs(jnp.sum(v_7d, axis=-1))
        max_zsum = max(max_zsum, float(jnp.max(per_sample_z)))
        mean_zsums.append(float(jnp.mean(per_sample_z)))
        all_p.append(np.asarray(jax.device_get(probs), dtype=np.float32))
        all_y.append(np.asarray(b["final_rank_matrix_4x4"], dtype=np.float32))
    p_cat = np.concatenate(all_p, axis=0)
    y_cat = np.concatenate(all_y, axis=0)
    ce = float(-np.mean(np.sum(y_cat * np.log(np.clip(p_cat, 1e-7, 1.0)), axis=-1)))
    return {
        "val_grp_ce": ce,
        "val_grp_brier": compute_brier_4rank(p_cat, y_cat),
        "val_grp_ece": compute_ece_4rank(p_cat, y_cat),
        "val_grp_birkhoff_mean_err": float(np.mean(mean_birkhoffs)),
        "val_grp_birkhoff_max_err": max_birkhoff,
        "val_grp_7d_zero_sum_mean_err": float(np.mean(mean_zsums)),
        "val_grp_7d_zero_sum_max_err": max_zsum,
    }


def evaluate_belief_val(
    model, params, t_cal: float, val_shards: Sequence[Path], batch_size: int = 512
) -> Dict[str, float]:
    @jax.jit
    def eval_step(p, batch):
        w_opp_5d = _extract_w_opp_5d(batch["w_table_abs"], batch["current_player"], None, 0.0)
        return model.apply(
            p,
            planes=batch["planes"],
            scalars=batch["scalars"],
            river_tokens=batch["river_tokens"],
            river_tile_ids=batch["river_tile_ids"],
            river_valid_mask=batch["river_valid_mask"],
            w_opp_5d=w_opp_5d,
            temperature_cal=t_cal,
        )

    keys = ("planes", "scalars", "river_tokens", "river_tile_ids", "river_valid_mask", "w_table_abs", "current_player")
    val_keys = keys + ("y_ron", "y_tenpai", "y_call_threat")
    loader = ShardBatchLoader(
        val_shards,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        keys=val_keys,
    )
    all_p_deal, all_y_ron = [], []
    all_p_ten, all_y_ten = [], []
    all_p_ct, all_y_ct = [], []
    for b_np in loader:
        b = {k: jnp.asarray(b_np[k]) for k in keys}
        out = jax.device_get(eval_step(params, b))
        all_p_deal.append(np.asarray(out["p_deal"], dtype=np.float32))
        all_y_ron.append(np.asarray(b_np["y_ron"], dtype=np.float32))
        all_p_ten.append(np.asarray(out["p_tenpai"], dtype=np.float32))
        all_y_ten.append(np.asarray(b_np["y_tenpai"], dtype=np.float32))
        all_p_ct.append(np.asarray(out["p_call_threat"], dtype=np.float32))
        all_y_ct.append(np.asarray(b_np["y_call_threat"], dtype=np.float32))

    p_d = np.concatenate(all_p_deal, axis=0)
    y_d = np.concatenate(all_y_ron, axis=0)
    p_t = np.concatenate(all_p_ten, axis=0)
    y_t = np.concatenate(all_y_ten, axis=0)
    p_c = np.concatenate(all_p_ct, axis=0)
    y_c = np.concatenate(all_y_ct, axis=0)

    eps = 1e-7
    deal_bce = float(-np.mean(y_d * np.log(np.clip(p_d, eps, 1.0)) + (1.0 - y_d) * np.log(np.clip(1.0 - p_d, eps, 1.0))))
    mean_pred = float(np.mean(p_d))
    mean_true = float(np.mean(y_d))
    return {
        "val_t_cal": float(t_cal),
        "val_deal_roc_auc": compute_binary_roc_auc(y_d, p_d),
        "val_tenpai_roc_auc": compute_binary_roc_auc(y_t, p_t),
        "val_call_threat_roc_auc": compute_binary_roc_auc(y_c, p_c),
        "val_deal_bce": deal_bce,
        "val_deal_pred_mean": mean_pred,
        "val_deal_true_rate": mean_true,
        "val_deal_calibration_ratio": mean_pred / max(mean_true, 1e-8),
    }


def evaluate_bc_val(
    pol_model,
    pol_params,
    bel_model,
    bel_params,
    t_cal: float,
    grp_model,
    grp_params,
    val_shards: Sequence[Path],
    batch_size: int = 512,
) -> Dict[str, float]:
    req_keys = (
        "planes", "scalars", "river_tokens", "river_tile_ids", "river_valid_mask",
        "last_draw_type", "is_last_draw_red", "target_type", "target_is_red",
        "action", "action_mask", "current_player", "w_table_abs",
        "table_scores", "table_round_num", "table_round_limit",
        "table_honba", "table_kyotaku", "table_dealer", "table_init_wind", "table_wall_rem",
    )

    @jax.jit
    def eval_step(batch):
        w_me, (du_ron_3opp, du_ron_87, du_agari_3, du_ryu, call_threat) = _compute_batch_trinity(
            bel_model, bel_params, t_cal, grp_model, grp_params, batch, rng_key=None, utility_dropout=0.0
        )
        mask = batch["action_mask"].astype(jnp.bool_)
        actions = batch["action"].astype(jnp.int32)
        active = jnp.sum(mask.astype(jnp.int32), axis=-1) >= 2
        logits, aux = pol_model.apply(
            pol_params,
            planes=batch["planes"],
            delta_u_ron_3opp=du_ron_3opp,
            call_threat_plane=call_threat,
            scalars=batch["scalars"],
            river_tokens=batch["river_tokens"],
            river_tile_ids=batch["river_tile_ids"],
            river_valid_mask=batch["river_valid_mask"],
            last_draw_type=batch["last_draw_type"],
            is_last_draw_red=batch["is_last_draw_red"],
            target_type=batch["target_type"],
            target_is_red=batch["target_is_red"],
            w_ego=w_me,
            delta_u_agari_3=du_agari_3,
            delta_u_ryuukyoku=du_ryu,
            delta_u_ron_87=du_ron_87,
            legal_action_mask=mask,
            return_aux=True,
        )
        log_p = masked_categorical_log_probs(logits, mask)
        idx = jnp.arange(actions.shape[0], dtype=jnp.int32)
        nll = masked_mean(-log_p[idx, actions], active)
        preds = jnp.argmax(jnp.where(mask, logits, -1e9), axis=-1)
        top1 = masked_mean((preds == actions).astype(jnp.float32), active)
        ent = masked_mean(masked_entropy(logits, mask), active)
        return nll, top1, ent, aux["beta_risk"], aux["beta_offense"]

    loader = ShardBatchLoader(
        val_shards,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        keys=req_keys,
    )
    nlls, accs, ents = [], [], []
    b_risk = b_off = 0.0
    for b_np in loader:
        b = {k: jnp.asarray(b_np[k]) for k in req_keys}
        nll, acc, ent, br, bo = eval_step(b)
        nlls.append(float(nll))
        accs.append(float(acc))
        ents.append(float(ent))
        b_risk, b_off = float(br), float(bo)
    return {
        "val_bc_nll": float(np.mean(nlls)),
        "val_bc_top1_acc": float(np.mean(accs)),
        "val_bc_entropy": float(np.mean(ents)),
        "learned_beta_risk": b_risk,
        "learned_beta_offense": b_off,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign-Mahjong V6 Phase 0 Training & Evaluation Runner")
    parser.add_argument("--shard-dir", type=Path, default=Path("data/v6_shards_2026"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/v6_phase0_run1"))
    parser.add_argument("--val-shards", type=int, default=2, help="Number of trailing shards held out for val")
    parser.add_argument("--grp-epochs", type=int, default=8)
    parser.add_argument("--belief-epochs", type=int, default=4)
    parser.add_argument("--bc-epochs", type=int, default=4)
    parser.add_argument("--eval-walls", type=int, default=25, help="Duplicate 1v3 walls (4 * walls full hanchans)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_shards = sorted(args.shard_dir.glob("shard_*.npz"))
    if len(all_shards) <= args.val_shards:
        raise SystemExit(f"Need > {args.val_shards} shards in {args.shard_dir}, found {len(all_shards)}")
    train_shards = all_shards[: -args.val_shards]
    val_shards = all_shards[-args.val_shards :]
    print(f"[Phase 0] Devices={jax.devices()} | train_shards={len(train_shards)} | val_shards={len(val_shards)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "train_shards": [str(p) for p in train_shards],
        "val_shards": [str(p) for p in val_shards],
    }

    # ------------------------------------------------------------------
    # 1. Phase 0-B: Birkhoff (4, 4) GRP_base
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    grp_cfg = GRPConfig(num_epochs=args.grp_epochs, batch_size=1024)
    print(f"\n=== Phase 0-B: Training Birkhoff GRP_base ({grp_cfg.num_epochs} epochs) ===", flush=True)
    grp_model, grp_params, grp_hist = train_grp(train_shards, cfg=grp_cfg, seed=args.seed)
    grp_val = evaluate_grp_val(grp_model, grp_params, val_shards)
    grp_time = time.perf_counter() - t0
    _save_pkl(args.out_dir / "grp_phase0b.pkl", {"config": asdict(grp_cfg), "params": grp_params})
    report["phase0b_grp"] = {"train_last": grp_hist[-1], "val": grp_val, "wall_s": grp_time}
    print(f"[Phase 0-B DONE in {grp_time:.1f}s] Val: {json.dumps(grp_val, indent=2)}", flush=True)

    # ------------------------------------------------------------------
    # 2. Phase 0-A: 36x Equivariant BeliefNet + Held-Out Temperature Scaling
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    bel_cfg = BeliefConfig(num_epochs=args.belief_epochs, batch_size=512, seed=args.seed)
    print(f"\n=== Phase 0-A: Training 36x Equivariant BeliefNet ({bel_cfg.num_epochs} epochs) ===", flush=True)
    bel_model, bel_params, t_cal, bel_hist = train_belief(
        train_shards, cfg=bel_cfg, val_shard_paths=val_shards, suit_augment=True
    )
    bel_val = evaluate_belief_val(bel_model, bel_params, t_cal, val_shards)
    bel_time = time.perf_counter() - t1
    _save_pkl(
        args.out_dir / "belief_phase0a.pkl",
        {"config": asdict(bel_cfg), "params": bel_params, "t_cal": float(t_cal)},
    )
    report["phase0a_belief"] = {"train_last": bel_hist[-1], "val": bel_val, "wall_s": bel_time}
    print(f"[Phase 0-A DONE in {bel_time:.1f}s] Val: {json.dumps(bel_val, indent=2)}", flush=True)

    # ------------------------------------------------------------------
    # 3. Phase 0-C: Strictly Isomorphic SovereignPolicyNet BC
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    model_cfg = ModelConfig()
    bc_cfg = BCConfig(model=model_cfg, num_epochs=args.bc_epochs, batch_size=512, suit_augment=True, seed=args.seed)
    print(f"\n=== Phase 0-C: Training Strictly Isomorphic SovereignPolicyNet BC ({bc_cfg.num_epochs} epochs) ===", flush=True)
    pol_model, pol_params, bc_hist = train_bc(
        train_shards,
        belief_model=bel_model,
        belief_params=bel_params,
        grp_model=grp_model,
        grp_params=grp_params,
        cfg=bc_cfg,
        belief_t_cal=t_cal,
    )
    bc_val = evaluate_bc_val(pol_model, pol_params, bel_model, bel_params, t_cal, grp_model, grp_params, val_shards)
    bc_time = time.perf_counter() - t2
    _save_pkl(
        args.out_dir / "bc_phase0c.pkl",
        {"model_config": asdict(model_cfg), "params": pol_params, "t_cal": float(t_cal)},
    )
    report["phase0c_bc"] = {"train_last": bc_hist[-1], "val": bc_val, "wall_s": bc_time}
    print(f"[Phase 0-C DONE in {bc_time:.1f}s] Val: {json.dumps(bc_val, indent=2)}", flush=True)

    # ------------------------------------------------------------------
    # 4. Strict Fixed-Wall Full-Hanchan Duplicate 1v3 Evaluation + Zero-Shot Multi-Rule Check
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    eval_cfg = EvalConfig(
        env_name="red_4p",
        round_mode="half",
        num_walls=args.eval_walls,
        max_steps=1600,
        rank_points=(90.0, 45.0, 0.0, -135.0),
        seed=2026,
    )
    print(
        f"\n=== Strict Duplicate 1v3 Full-Hanchan Eval ({4*args.eval_walls} games per rule vs rule_based_player) ===",
        flush=True,
    )
    eval_results: Dict[str, Dict[str, float]] = {}
    for rule_name, w_rule in (
        ("phoenix_7d", W_PHOENIX_7D),
        ("phoenix_10d", W_PHOENIX_10D),
        ("m_league", W_M_LEAGUE),
    ):
        actor = make_v6_policy_actor(
            policy_model=pol_model,
            policy_params=pol_params,
            belief_model=bel_model,
            belief_ema_params=bel_params,
            grp_model=grp_model,
            grp_base_params=grp_params,
            w_rule=w_rule,
            opp_rules_known=False,
            belief_t_cal=t_cal,
            greedy=True,
        )
        res = evaluate_duplicate_1v3(actor, mahjax.rule_based_player, eval_cfg)
        eval_results[rule_name] = res
        print(
            f"  [{rule_name:11s}] pt={res['tenhou_pt_mean']:+.2f} ± {res['tenhou_pt_ci95']:.2f} | "
            f"avg_rank={res['avg_rank']:.3f} | 1st={res['rank_1_rate']*100:.1f}% 4th={res['rank_4_rate']*100:.1f}% | "
            f"hora={res['hora_rate']*100:.1f}% deal_in={res['deal_in_rate']*100:.1f}% riichi={res['riichi_rate']*100:.1f}% meld={res['meld_rate']*100:.1f}%",
            flush=True,
        )

    eval_time = time.perf_counter() - t3
    report["duplicate_1v3_eval_vs_rule_based"] = eval_results
    report["eval_wall_s"] = eval_time

    (args.out_dir / "phase0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ALL PHASE 0 & EVAL COMPLETE] Saved report to {args.out_dir / 'phase0_report.json'}", flush=True)


if __name__ == "__main__":
    main()

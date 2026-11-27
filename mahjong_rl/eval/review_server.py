"""Local FastAPI + Sovereign-Mahjong V6 GPU backend for KillerDucky interactive replay review & What-If branching.

Usage:
    /opt/miniconda3/envs/usual/bin/python -m mahjong_rl.eval.review_server --port 8000
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

import mahjax
from mahjax.common.action import Action
from mahjax.ui import view as ui_view
from mahjax.ui.rules import Rules
from mahjong_rl.algo.utility import vnm_normalize_utility
from mahjong_rl.data.mjai_parquet_converter import _init_round_state_from_deck
from mahjong_rl.eval.replay_reviewer import (
    REVIEW_RULE_MAP,
    build_eval_step_fn,
    classify_event_kind,
    format_action_cn,
    format_round_label,
    is_discard_action,
    load_replay_text_to_mjai,
    load_v6_reviewer_models,
    review_replay,
    serialize_p_deal,
    snapshot_frame,
)

MAX_SESSIONS = 16
MAX_BRANCHES = 64
DEFAULT_CKPT_DIR = Path("checkpoints/sovereign_v6")
DEFAULT_SAMPLE_JSON = Path("web/reviewer/public/sample_b70ecabe.json")
DEFAULT_DIST_DIR = Path("web/reviewer/dist")


@dataclass
class BranchNode:
    """One ply in a What-If branch, preserving pre-action state for instant undo/re-branch."""

    pre_state: Any
    pre_river_origins: List[List[int]]
    pre_turn_counter: List[int]
    pre_round_wins: List[Dict[str, Any]]
    pre_step_in_round: int
    ai_eval: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    frame: Dict[str, Any]


@dataclass
class BranchSession:
    branch_id: str
    session_id: Optional[str]
    round_idx: int
    fork_frame_idx: int
    fork_step_in_round: int
    pov_seat: int
    rule: str
    round_num: int
    honba: int
    kyotaku: int
    dealer: int
    round_label: str
    score_start: List[int]
    history: List[BranchNode] = field(default_factory=list)
    round_over: bool = False


class _ReviewEngine:
    """Thread-safe container holding Sovereign V6 models, JIT fns, sessions, and What-If branches."""

    def __init__(self, ckpt_dir: Path, policy_ckpt: Optional[Path], sample_json: Path) -> None:
        self.ckpt_dir = ckpt_dir
        self.policy_ckpt = policy_ckpt
        self.sample_json = sample_json
        self.lock = threading.Lock()
        self.ui_rules = Rules("red_4p")
        self.env = mahjax.make("red_4p", round_mode="half", observe_type="dict", round_end="confirm")
        self.step_fn = jax.jit(self.env.step)
        self.dummy_key = jax.random.PRNGKey(0)
        self.models: Optional[Dict[str, Any]] = None
        self.eval_fns: Dict[str, Any] = {}
        self.sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.branches: OrderedDict[str, BranchSession] = OrderedDict()

    def ensure_loaded(self) -> None:
        if self.models is not None:
            return
        self.models = load_v6_reviewer_models(
            self.ckpt_dir,
            policy_ckpt=self.policy_ckpt,
        )
        # Warm up step_fn & default 7d eval_fn so first What-If request is fast.
        eval_fn = self.get_eval_fn("7d")
        warm_state = jax.device_get(self.env.init(self.dummy_key))
        legal = self.ui_rules.legal_actions(warm_state)
        if legal:
            jax.block_until_ready(self.step_fn(warm_state, jnp.int32(legal[0]), self.dummy_key))
        probs, du_ron, p_tenpai, p_deal = eval_fn(warm_state)
        jax.block_until_ready((probs, du_ron, p_tenpai, p_deal))

        if self.sample_json.exists() and "sample" not in self.sessions:
            try:
                data = json.loads(self.sample_json.read_text(encoding="utf-8"))
                self.sessions["sample"] = data
            except Exception:
                pass

    def get_eval_fn(self, rule: str) -> Any:
        assert self.models is not None
        key = rule.lower().strip()
        if key in self.eval_fns:
            return self.eval_fns[key]
        if key not in REVIEW_RULE_MAP:
            raise HTTPException(status_code=400, detail=f"未知规则 {rule!r}，可选: {sorted(REVIEW_RULE_MAP)}")
        w_rule = REVIEW_RULE_MAP[key]
        fn = build_eval_step_fn(
            policy_model=self.models["policy_model"],
            policy_params=self.models["policy_params"],
            belief_model=self.models["belief_model"],
            belief_params=self.models["belief_params"],
            grp_model=self.models["grp_model"],
            grp_params=self.models["grp_params"],
            w_rule=w_rule,
            belief_t_cal=self.models["belief_t_cal"],
        )
        self.eval_fns[key] = fn
        return fn

    def store_session(self, report: Dict[str, Any], session_id: Optional[str] = None) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        if sid in self.sessions:
            self.sessions.move_to_end(sid)
        self.sessions[sid] = report
        while len(self.sessions) > MAX_SESSIONS:
            # Prefer keeping "sample" if possible
            oldest = next((k for k in self.sessions.keys() if k != "sample"), None)
            if oldest is None:
                break
            self.sessions.pop(oldest, None)
        return sid

    def store_branch(self, branch: BranchSession) -> None:
        self.branches[branch.branch_id] = branch
        self.branches.move_to_end(branch.branch_id)
        while len(self.branches) > MAX_BRANCHES:
            self.branches.popitem(last=False)

    def run_eval(self, eval_fn: Any, st: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        probs_j, du_ron_j, p_tenpai_j, p_deal_j = eval_fn(st)
        return (
            np.asarray(jax.device_get(probs_j), dtype=np.float64),
            np.asarray(jax.device_get(du_ron_j), dtype=np.float64),
            np.asarray(jax.device_get(p_tenpai_j), dtype=np.float64),
            np.asarray(jax.device_get(p_deal_j), dtype=np.float64),
        )


def _clone_origins(origins: List[List[int]]) -> List[List[int]]:
    return [list(col) for col in origins]


def _clone_wins(wins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(w) for w in wins]


def _branch_origin_code(branch_ply: int) -> int:
    """Negative origin encoding for What-If river tiles (`-2` -> branch_ply 0, `-3` -> branch_ply 1, ...)."""
    return -(int(branch_ply) + 2)


def _emit_round_end_node(
    engine: _ReviewEngine,
    branch: BranchSession,
    *,
    end_state: Any,
    river_origins: List[List[int]],
    turn_counter: List[int],
    round_wins: List[Dict[str, Any]],
    step_in_round: int,
    result_type: str,
    reason: Optional[str],
    abort_seat: Optional[int],
) -> None:
    round_result = ui_view.build_round_result(
        engine.ui_rules,
        type=result_type,
        reason=reason,
        state=end_state,
        winners=round_wins,
        score_start=branch.score_start,
        game_over=False,
        abort_seat=abort_seat,
    )
    ura_dora: List[int] = []
    for w in round_result.get("winners") or []:
        if w.get("uraDora"):
            ura_dora = [int(x) for x in w["uraDora"]]
            break

    branch_ply = len(branch.history)
    frame = snapshot_frame(
        engine.ui_rules,
        end_state,
        idx=branch_ply,
        round_idx=branch.round_idx,
        ply=branch_ply,
        step_in_round=step_in_round,
        round_label=branch.round_label,
        turn=max(max(turn_counter), 1),
        seat=int(end_state.current_player),
        event_kind="round_end",
        river_origins=river_origins,
        chosen_action=-1,
        is_decision=False,
        ai_eval=None,
        ura_dora=ura_dora,
        round_result=round_result,
    )
    frame["is_branch"] = True
    frame["branch_ply"] = branch_ply
    branch.history.append(
        BranchNode(
            pre_state=end_state,
            pre_river_origins=_clone_origins(river_origins),
            pre_turn_counter=list(turn_counter),
            pre_round_wins=_clone_wins(round_wins),
            pre_step_in_round=step_in_round,
            ai_eval=None,
            frame=frame,
        )
    )
    branch.round_over = True


def _apply_branch_action_and_advance(
    engine: _ReviewEngine,
    branch: BranchSession,
    eval_fn: Any,
    *,
    st: Any,
    river_origins: List[List[int]],
    turn_counter: List[int],
    round_wins: List[Dict[str, Any]],
    step_in_round: int,
    action: int,
    ai_eval: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Apply Hero's `action` at the tail of `branch.history`, then advance opponents until Hero's next turn."""
    ui_rules = engine.ui_rules
    cp = int(st.current_player)
    mask_np = np.asarray(st.legal_action_mask, dtype=bool)
    if action < 0 or action >= len(mask_np) or not bool(mask_np[action]):
        legal = [int(a) for a in np.flatnonzero(mask_np).tolist()]
        raise HTTPException(status_code=400, detail=f"动作 {action} 当前不合法，合法动作: {legal}")

    # Record Hero's chosen step
    branch_ply = len(branch.history)
    is_disc = is_discard_action(action)
    next_turns = list(turn_counter)
    if is_disc:
        next_turns[cp] += 1
    ev_kind = classify_event_kind(action)
    num_legal = int(mask_np.sum())
    if ai_eval is None and num_legal >= 2:
        ai_eval = engine.run_eval(eval_fn, st)

    hero_frame = snapshot_frame(
        ui_rules,
        st,
        idx=branch_ply,
        round_idx=branch.round_idx,
        ply=branch_ply,
        step_in_round=step_in_round,
        round_label=branch.round_label,
        turn=max(next_turns[cp], 1),
        seat=cp,
        event_kind=ev_kind,
        river_origins=river_origins,
        chosen_action=action,
        is_decision=(num_legal >= 2),
        ai_eval=ai_eval,
    )
    hero_frame["is_branch"] = True
    hero_frame["branch_ply"] = branch_ply
    branch.history.append(
        BranchNode(
            pre_state=st,
            pre_river_origins=_clone_origins(river_origins),
            pre_turn_counter=list(turn_counter),
            pre_round_wins=_clone_wins(round_wins),
            pre_step_in_round=step_in_round,
            ai_eval=ai_eval,
            frame=hero_frame,
        )
    )

    next_origins = _clone_origins(river_origins)
    if ev_kind == "discard":
        next_origins[cp].append(_branch_origin_code(branch_ply))
    next_wins = _clone_wins(round_wins)

    if action == Action.KYUUSHU:
        reason, abort_seat = ui_view.abortive_cause(ui_rules, st)
        _emit_round_end_node(
            engine,
            branch,
            end_state=st,
            river_origins=next_origins,
            turn_counter=next_turns,
            round_wins=next_wins,
            step_in_round=step_in_round,
            result_type="abortive",
            reason=reason,
            abort_seat=abort_seat,
        )
        return

    post_st = engine.step_fn(st, jnp.int32(action), engine.dummy_key)
    if action in (Action.TSUMO, Action.RON):
        next_wins.append(ui_view.build_win(ui_rules, st, post_st, cp, action == Action.RON))

    cur_st = post_st
    cur_step = step_in_round + 1
    cur_turns = next_turns

    # Advance opponents (and forced single-option Hero actions) until Hero's next real decision or round end
    for _ in range(180):
        if ui_rules.is_round_over(cur_st) or bool(cur_st.terminated) or bool(cur_st.truncated):
            r_type = (
                "ron"
                if any(w.get("from") is not None for w in next_wins)
                else ("tsumo" if next_wins else "draw")
            )
            _emit_round_end_node(
                engine,
                branch,
                end_state=cur_st,
                river_origins=next_origins,
                turn_counter=cur_turns,
                round_wins=next_wins,
                step_in_round=cur_step,
                result_type=r_type,
                reason=None,
                abort_seat=None,
            )
            return

        cur_cp = int(cur_st.current_player)
        cur_mask = np.asarray(cur_st.legal_action_mask, dtype=bool)
        legal = np.flatnonzero(cur_mask)
        if len(legal) == 0:
            break

        if len(legal) == 1 and int(legal[0]) == Action.KYUUSHU:
            reason, abort_seat = ui_view.abortive_cause(ui_rules, cur_st)
            _emit_round_end_node(
                engine,
                branch,
                end_state=cur_st,
                river_origins=next_origins,
                turn_counter=cur_turns,
                round_wins=next_wins,
                step_in_round=cur_step,
                result_type="abortive",
                reason=reason,
                abort_seat=abort_seat,
            )
            return

        if cur_cp == branch.pov_seat:
            if len(legal) == 1 and int(legal[0]) in (Action.TSUMOGIRI, Action.PASS):
                auto_act = int(legal[0])
                auto_kind = classify_event_kind(auto_act)
                auto_turns = list(cur_turns)
                if is_discard_action(auto_act):
                    auto_turns[cur_cp] += 1
                if auto_kind == "discard":
                    b_ply = len(branch.history)
                    fr = snapshot_frame(
                        ui_rules,
                        cur_st,
                        idx=b_ply,
                        round_idx=branch.round_idx,
                        ply=b_ply,
                        step_in_round=cur_step,
                        round_label=branch.round_label,
                        turn=max(auto_turns[cur_cp], 1),
                        seat=cur_cp,
                        event_kind="discard",
                        river_origins=next_origins,
                        chosen_action=auto_act,
                        is_decision=False,
                        ai_eval=None,
                    )
                    fr["is_branch"] = True
                    fr["branch_ply"] = b_ply
                    branch.history.append(
                        BranchNode(
                            pre_state=cur_st,
                            pre_river_origins=_clone_origins(next_origins),
                            pre_turn_counter=list(cur_turns),
                            pre_round_wins=_clone_wins(next_wins),
                            pre_step_in_round=cur_step,
                            ai_eval=None,
                            frame=fr,
                        )
                    )
                    next_origins[cur_cp].append(_branch_origin_code(b_ply))
                cur_st = engine.step_fn(cur_st, jnp.int32(auto_act), engine.dummy_key)
                cur_turns = auto_turns
                cur_step += 1
                continue

            # Stop at Hero's live decision!
            live_eval = engine.run_eval(eval_fn, cur_st)
            b_ply = len(branch.history)
            pending_frame = snapshot_frame(
                ui_rules,
                cur_st,
                idx=b_ply,
                round_idx=branch.round_idx,
                ply=b_ply,
                step_in_round=cur_step,
                round_label=branch.round_label,
                turn=max(cur_turns[cur_cp] + 1, 1),
                seat=cur_cp,
                event_kind="pending",
                river_origins=next_origins,
                chosen_action=-1,
                is_decision=True,
                ai_eval=live_eval,
            )
            pending_frame["is_branch"] = True
            pending_frame["branch_ply"] = b_ply
            branch.history.append(
                BranchNode(
                    pre_state=cur_st,
                    pre_river_origins=_clone_origins(next_origins),
                    pre_turn_counter=list(cur_turns),
                    pre_round_wins=_clone_wins(next_wins),
                    pre_step_in_round=cur_step,
                    ai_eval=live_eval,
                    frame=pending_frame,
                )
            )
            branch.round_over = False
            return

        # Opponent turn: run Sovereign V6 AI greedily
        opp_eval: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        if len(legal) == 1:
            opp_act = int(legal[0])
        else:
            opp_eval = engine.run_eval(eval_fn, cur_st)
            probs = opp_eval[0]
            order = legal[np.argsort(-probs[legal])]
            opp_act = int(order[0])

        if opp_act == Action.KYUUSHU:
            reason, abort_seat = ui_view.abortive_cause(ui_rules, cur_st)
            _emit_round_end_node(
                engine,
                branch,
                end_state=cur_st,
                river_origins=next_origins,
                turn_counter=cur_turns,
                round_wins=next_wins,
                step_in_round=cur_step,
                result_type="abortive",
                reason=reason,
                abort_seat=abort_seat,
            )
            return

        opp_kind = classify_event_kind(opp_act)
        opp_turns = list(cur_turns)
        if is_discard_action(opp_act):
            opp_turns[cur_cp] += 1

        if opp_kind in ("discard", "riichi", "call", "win"):
            b_ply = len(branch.history)
            opp_fr = snapshot_frame(
                ui_rules,
                cur_st,
                idx=b_ply,
                round_idx=branch.round_idx,
                ply=b_ply,
                step_in_round=cur_step,
                round_label=branch.round_label,
                turn=max(opp_turns[cur_cp], 1),
                seat=cur_cp,
                event_kind=opp_kind,
                river_origins=next_origins,
                chosen_action=opp_act,
                is_decision=False,
                ai_eval=None,
            )
            opp_fr["is_branch"] = True
            opp_fr["branch_ply"] = b_ply
            branch.history.append(
                BranchNode(
                    pre_state=cur_st,
                    pre_river_origins=_clone_origins(next_origins),
                    pre_turn_counter=list(cur_turns),
                    pre_round_wins=_clone_wins(next_wins),
                    pre_step_in_round=cur_step,
                    ai_eval=None,
                    frame=opp_fr,
                )
            )
            if opp_kind == "discard":
                next_origins[cur_cp].append(_branch_origin_code(b_ply))

        opp_post = engine.step_fn(cur_st, jnp.int32(opp_act), engine.dummy_key)
        if opp_act in (Action.TSUMO, Action.RON):
            next_wins.append(ui_view.build_win(ui_rules, cur_st, opp_post, cur_cp, opp_act == Action.RON))

        cur_st = opp_post
        cur_turns = opp_turns
        cur_step += 1


# -----------------------------------------------------------------------------
# Request / Response Models
# -----------------------------------------------------------------------------


class ReviewReq(BaseModel):
    url: Optional[str] = None
    content: Optional[str] = None
    filename: Optional[str] = None
    rule: str = "7d"


class BranchStartReq(BaseModel):
    session_id: Optional[str] = None
    round_idx: int = Field(..., ge=0)
    step_in_round: int = Field(..., ge=0)
    fork_frame_idx: int = Field(..., ge=0)
    pov_seat: int = Field(..., ge=0, le=3)
    action: int = Field(..., ge=0, lt=87)
    rule: str = "7d"
    round_seed: Optional[Dict[str, Any]] = None


class BranchStepReq(BaseModel):
    branch_id: str
    from_branch_ply: int = Field(..., ge=0)
    action: int = Field(..., ge=0, lt=87)
    fork_new: bool = False


def create_app(
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    policy_ckpt: Optional[Path] = None,
    sample_json: Path = DEFAULT_SAMPLE_JSON,
    dist_dir: Path = DEFAULT_DIST_DIR,
) -> FastAPI:
    engine = _ReviewEngine(ckpt_dir=ckpt_dir, policy_ckpt=policy_ckpt, sample_json=sample_json)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        with engine.lock:
            engine.ensure_loaded()
        yield

    app = FastAPI(title="Sovereign-Mahjong V6 Interactive Review Server", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        with engine.lock:
            engine.ensure_loaded()
            devices = [str(d) for d in jax.devices()]
            return {
                "status": "ok",
                "device": devices[0] if devices else "cpu",
                "policy_source": (engine.models or {}).get("policy_source", ""),
                "rules": sorted(REVIEW_RULE_MAP.keys()),
                "has_sample": engine.sample_json.exists() or ("sample" in engine.sessions),
            }

    @app.get("/api/sample")
    def get_sample() -> Dict[str, Any]:
        with engine.lock:
            engine.ensure_loaded()
            if "sample" in engine.sessions:
                return {"session_id": "sample", "report": engine.sessions["sample"]}
            if not engine.sample_json.exists():
                raise HTTPException(status_code=404, detail=f"未找到示例文件: {engine.sample_json}")
            data = json.loads(engine.sample_json.read_text(encoding="utf-8"))
            engine.sessions["sample"] = data
            return {"session_id": "sample", "report": data}

    @app.post("/api/load-report")
    def load_report(report: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(report, dict) or "frames" not in report or "seats" not in report:
            raise HTTPException(status_code=400, detail="无效的评测 JSON：缺少 frames 或 seats 字段")
        with engine.lock:
            engine.ensure_loaded()
            sid = engine.store_session(report)
            rounds = report.get("rounds") or []
            has_decks = bool(rounds and all("deck" in r and "actions" in r for r in rounds))
            return {"session_id": sid, "has_decks": has_decks}

    @app.post("/api/review")
    def review(req: ReviewReq) -> Dict[str, Any]:
        if not req.url and not req.content:
            raise HTTPException(status_code=400, detail="请提供天凤牌谱链接或上传牌谱文件内容")
        with engine.lock:
            engine.ensure_loaded()
            assert engine.models is not None
            try:
                if req.content:
                    events, player_names = load_replay_text_to_mjai(req.content)
                    src_label = req.filename or "uploaded_replay"
                    report = review_replay(
                        src_label,
                        engine.models,
                        rule=req.rule,
                        events_override=(events, player_names),
                    )
                else:
                    assert req.url is not None
                    report = review_replay(
                        req.url.strip(),
                        engine.models,
                        rule=req.rule,
                    )
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err
            except Exception as err:
                raise HTTPException(status_code=500, detail=f"牌谱解析或评测失败: {err}") from err
            sid = engine.store_session(report)
            return {"session_id": sid, "report": report}

    @app.post("/api/branch/start")
    def branch_start(req: BranchStartReq) -> Dict[str, Any]:
        with engine.lock:
            engine.ensure_loaded()
            session = engine.sessions.get(req.session_id or "") if req.session_id else None
            round_meta: Optional[Dict[str, Any]] = None
            round_frames: List[Dict[str, Any]] = []
            if session is not None:
                rounds = session.get("rounds") or []
                if 0 <= req.round_idx < len(rounds):
                    round_meta = rounds[req.round_idx]
                round_frames = [
                    fr for fr in (session.get("frames") or []) if int(fr.get("round_idx", -1)) == req.round_idx
                ]
            if round_meta is None and req.round_seed is not None:
                round_meta = req.round_seed

            if round_meta is None or "deck" not in round_meta or "actions" not in round_meta:
                raise HTTPException(
                    status_code=422,
                    detail="当前牌谱 JSON 缺少逐局牌山 (deck/actions)，无法进行实时 What-If 分支推演；请通过本后端重新评测牌谱或加载 v2 JSON。",
                )

            deck = np.asarray(round_meta["deck"], dtype=np.int8)
            actions = [int(a) for a in round_meta["actions"]]
            if req.step_in_round > len(actions):
                raise HTTPException(status_code=400, detail=f"step_in_round={req.step_in_round} 超出该局步数 {len(actions)}")

            round_num = int(round_meta["round_num"])
            honba = int(round_meta["honba"])
            kyotaku = int(round_meta["kyotaku"])
            dealer = int(round_meta["dealer"])
            init_scores = [int(x) for x in round_meta["init_scores"]]
            scores_100 = np.asarray([s // 100 for s in init_scores], dtype=np.int32)
            round_label = str(round_meta.get("round_label") or format_round_label(round_num, honba))

            st = _init_round_state_from_deck(
                jnp.asarray(deck),
                jnp.asarray(round_num),
                jnp.asarray(honba),
                jnp.asarray(kyotaku),
                jnp.asarray(dealer),
                jnp.asarray(scores_100),
                round_limit=7,
            )
            turn_counter = [0, 0, 0, 0]
            round_wins: List[Dict[str, Any]] = []
            for k in range(req.step_in_round):
                act = actions[k]
                cp_k = int(st.current_player)
                if is_discard_action(act):
                    turn_counter[cp_k] += 1
                post_st = engine.step_fn(st, jnp.int32(act), engine.dummy_key)
                if act in (Action.TSUMO, Action.RON):
                    round_wins.append(ui_view.build_win(engine.ui_rules, st, post_st, cp_k, act == Action.RON))
                st = post_st

            if int(st.current_player) != req.pov_seat:
                raise HTTPException(
                    status_code=400,
                    detail=f"该帧当前行动玩家为 P{int(st.current_player)}，非视角玩家 P{req.pov_seat}，无法直接分叉出牌。",
                )

            # Reconstruct exact original river_origins for tiles thrown before step_in_round
            prefix_origins: List[List[int]] = [[], [], [], []]
            if session is not None and 0 <= req.fork_frame_idx < len(session.get("frames") or []):
                fork_fr = session["frames"][req.fork_frame_idx]
                for p in range(4):
                    prefix_origins[p] = [
                        int(r[4]) if len(r) > 4 else -1 for r in (fork_fr.get("rivers") or [[], [], [], []])[p]
                    ]
            elif round_frames:
                for fr in round_frames:
                    if int(fr.get("step_in_round", 0)) < req.step_in_round and fr.get("event_kind") == "discard":
                        prefix_origins[int(fr["seat"])].append(int(fr["idx"]))
            else:
                for p in range(4):
                    prefix_origins[p] = [-1] * len(engine.ui_rules.river(st, p))

            rule_key = (req.rule or (session or {}).get("rule") or "7d").lower().strip()
            if rule_key not in REVIEW_RULE_MAP:
                rule_key = "7d"
            eval_fn = engine.get_eval_fn(rule_key)

            branch = BranchSession(
                branch_id=uuid.uuid4().hex[:12],
                session_id=req.session_id,
                round_idx=req.round_idx,
                fork_frame_idx=req.fork_frame_idx,
                fork_step_in_round=req.step_in_round,
                pov_seat=req.pov_seat,
                rule=rule_key,
                round_num=round_num,
                honba=honba,
                kyotaku=kyotaku,
                dealer=dealer,
                round_label=round_label,
                score_start=init_scores,
            )

            _apply_branch_action_and_advance(
                engine,
                branch,
                eval_fn,
                st=st,
                river_origins=prefix_origins,
                turn_counter=turn_counter,
                round_wins=round_wins,
                step_in_round=req.step_in_round,
                action=req.action,
                ai_eval=None,
            )
            engine.store_branch(branch)
            return {
                "branch_id": branch.branch_id,
                "fork_frame_idx": branch.fork_frame_idx,
                "frames": [node.frame for node in branch.history],
                "round_over": branch.round_over,
            }

    @app.post("/api/branch/step")
    def branch_step(req: BranchStepReq) -> Dict[str, Any]:
        with engine.lock:
            engine.ensure_loaded()
            branch = engine.branches.get(req.branch_id)
            if branch is None:
                raise HTTPException(status_code=404, detail=f"分支会话不存在或已过期: {req.branch_id}")
            if req.from_branch_ply >= len(branch.history):
                raise HTTPException(status_code=400, detail=f"无效的分支帧索引: {req.from_branch_ply}")

            target_node = branch.history[req.from_branch_ply]
            if int(target_node.pre_state.current_player) != branch.pov_seat:
                raise HTTPException(status_code=400, detail="只能在当前视角玩家的决策帧分叉/续玩")

            # Truncate tail for undo/re-branch, or clone non-destructively when fork_new=True
            if req.fork_new:
                work_branch = BranchSession(
                    branch_id=uuid.uuid4().hex[:12],
                    session_id=branch.session_id,
                    round_idx=branch.round_idx,
                    fork_frame_idx=branch.fork_frame_idx,
                    fork_step_in_round=branch.fork_step_in_round,
                    pov_seat=branch.pov_seat,
                    rule=branch.rule,
                    round_num=branch.round_num,
                    honba=branch.honba,
                    kyotaku=branch.kyotaku,
                    dealer=branch.dealer,
                    round_label=branch.round_label,
                    score_start=list(branch.score_start),
                    history=list(branch.history[: req.from_branch_ply]),
                    round_over=False,
                )
            else:
                branch.history = branch.history[: req.from_branch_ply]
                branch.round_over = False
                work_branch = branch

            eval_fn = engine.get_eval_fn(work_branch.rule)

            _apply_branch_action_and_advance(
                engine,
                work_branch,
                eval_fn,
                st=target_node.pre_state,
                river_origins=target_node.pre_river_origins,
                turn_counter=target_node.pre_turn_counter,
                round_wins=target_node.pre_round_wins,
                step_in_round=target_node.pre_step_in_round,
                action=req.action,
                ai_eval=target_node.ai_eval,
            )
            engine.store_branch(work_branch)
            return {
                "branch_id": work_branch.branch_id,
                "fork_frame_idx": work_branch.fork_frame_idx,
                "frames": [node.frame for node in work_branch.history],
                "round_over": work_branch.round_over,
            }

    # Serve built Vite app when available
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        assets_dir = dist_dir / "assets"
        tiles_dir = dist_dir / "tiles"

        class _NoCacheStaticFiles(StaticFiles):
            def file_response(self, *args: Any, **kwargs: Any) -> Any:
                resp = super().file_response(*args, **kwargs)
                resp.headers["Cache-Control"] = "no-cache"
                return resp

        if assets_dir.exists():
            app.mount("/assets", _NoCacheStaticFiles(directory=str(assets_dir)), name="assets")
        if tiles_dir.exists():
            app.mount("/tiles", _NoCacheStaticFiles(directory=str(tiles_dir)), name="tiles")

        @app.get("/", response_class=HTMLResponse)
        def serve_index() -> str:
            return (dist_dir / "index.html").read_text(encoding="utf-8")

        @app.get("/{file_path:path}")
        def serve_public_file(file_path: str) -> FileResponse:
            if file_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found")
            candidate = (dist_dir / file_path).resolve()
            if candidate.is_file() and str(candidate).startswith(str(dist_dir.resolve())):
                return FileResponse(candidate, headers={"Cache-Control": "no-cache"})
            raise HTTPException(status_code=404, detail="Not found")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign-Mahjong V6 Interactive Review Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--policy-ckpt", type=Path, default=None)
    parser.add_argument("--sample-json", type=Path, default=DEFAULT_SAMPLE_JSON)
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST_DIR)
    args = parser.parse_args()

    app = create_app(
        ckpt_dir=args.ckpt_dir,
        policy_ckpt=args.policy_ckpt,
        sample_json=args.sample_json,
        dist_dir=args.dist_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

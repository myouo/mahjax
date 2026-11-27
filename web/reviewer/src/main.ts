import "./style.css";
import {
  checkHealth,
  fetchSample,
  reviewReplayFile,
  reviewReplayUrl,
  startBranch,
  stepBranch,
  uploadReportForSession,
} from "./api";
import { fitBoard, renderBoard } from "./board";
import { getActiveBranch, getBranchDisplayFrame } from "./branch_graph";
import { renderModal } from "./modals";
import { renderSidebar } from "./sidebar";
import { escHtml, playerName } from "./tiles";
import {
  normalizeReport,
  resolveRiverJump,
  stepDiff,
  stepError,
  stepPovDecision,
  stepRound,
  stepTableEvent,
} from "./timeline";
import type { AppMode, AppState, GitBranch, ReviewFrame, ReviewReport, ReviewTab } from "./types";

const root = document.getElementById("app")!;

const state: AppState = {
  report: null,
  sessionId: null,
  backendOnline: false,
  backendInfo: null,
  mode: "review",
  reviewTab: "analysis",
  povSeat: 0,
  frameIdx: 0,
  revealSeats: [false, false, false, false],
  showAI: true,
  showDanger: false,
  wwydAnswers: {},
  branch: {
    active: false,
    forkFrameIdx: 0,
    forkRoundIdx: 0,
    forkLabel: "",
    branches: [],
    activeBranchId: "main",
    cursor: 0,
    nextBranchSeq: 1,
    loading: false,
  },
  theme: "light",
  highlightAction: null,
  pulseRiverOrigin: null,
  activeModal: null,
  errorThreshold: 0.2,
  toast: null,
};

let toastTimer: number | null = null;
let pulseTimer: number | null = null;

function showToast(message: string, kind: "info" | "warn" | "error" | "ok" = "info", ms = 2400): void {
  state.toast = { message, kind };
  renderToast();
  if (toastTimer !== null) window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    state.toast = null;
    renderToast();
  }, ms);
}

function renderToast(): void {
  const el = document.getElementById("toastHost");
  if (!el) return;
  if (!state.toast) {
    el.className = "toast-host hidden";
    el.innerHTML = "";
    return;
  }
  el.className = `toast-host tone-${state.toast.kind}`;
  el.textContent = state.toast.message;
}

function getActiveTimeline(): { frames: ReviewFrame[]; cursor: number } {
  if (state.branch.active) {
    if (state.branch.activeBranchId === "main") {
      const rf = state.report?.frames[state.branch.forkFrameIdx];
      return { frames: rf ? [rf] : [], cursor: 0 };
    }
    const b = getActiveBranch(state.branch);
    return { frames: b?.frames ?? [], cursor: state.branch.cursor };
  }
  return { frames: state.report?.frames ?? [], cursor: state.frameIdx };
}

function setActiveCursor(nextIdx: number): void {
  if (state.branch.active) {
    if (state.branch.activeBranchId === "main") return;
    const b = getActiveBranch(state.branch);
    const n = b?.frames.length ?? 0;
    if (!n) return;
    state.branch.cursor = Math.max(0, Math.min(nextIdx, n - 1));
  } else {
    const n = state.report?.frames.length ?? 0;
    if (!n) return;
    state.frameIdx = Math.max(0, Math.min(nextIdx, n - 1));
  }
  state.highlightAction = null;
  update();
}

function clearBranchState(restoreFork = true): void {
  if (!state.branch.active) return;
  const forkIdx = state.branch.forkFrameIdx;
  state.branch = {
    active: false,
    forkFrameIdx: 0,
    forkRoundIdx: 0,
    forkLabel: "",
    branches: [],
    activeBranchId: "main",
    cursor: 0,
    nextBranchSeq: 1,
    loading: false,
  };
  if (restoreFork && state.report && forkIdx >= 0 && forkIdx < state.report.frames.length) {
    state.frameIdx = forkIdx;
  }
  state.highlightAction = null;
}

function snapToNearestPovDecision(): void {
  const frames = state.report?.frames ?? [];
  if (!frames.length) return;
  const cur = frames[state.frameIdx];
  if (cur && cur.seat === state.povSeat && (cur.is_decision ?? true) && cur.event_kind !== "round_end") {
    return;
  }
  let idx = stepPovDecision(frames, state.frameIdx, 1, state.povSeat);
  if (idx < 0) idx = stepPovDecision(frames, state.frameIdx, -1, state.povSeat);
  if (idx >= 0) state.frameIdx = idx;
}

function setMode(nextMode: AppMode): void {
  if (nextMode === state.mode) return;
  if (nextMode === "review") {
    clearBranchState(true);
    state.mode = "review";
  } else if (nextMode === "wwyd") {
    clearBranchState(true);
    state.mode = "wwyd";
    snapToNearestPovDecision();
  } else {
    state.mode = "sandbox";
    if (!state.branch.active) {
      snapToNearestPovDecision();
    }
  }
  state.highlightAction = null;
  update();
}

function undoBranchStep(): void {
  if (!state.branch.active) return;
  if (state.branch.activeBranchId === "main") {
    clearBranchState(true);
    state.mode = "review";
    update();
    return;
  }
  const b = getActiveBranch(state.branch);
  if (!b) {
    clearBranchState(true);
    update();
    return;
  }
  const frames = b.frames;
  for (let i = state.branch.cursor - 1; i >= b.forkPly; i--) {
    const fr = frames[i];
    if (fr.seat === state.povSeat && fr.event_kind !== "round_end" && (fr.legal_actions?.length ?? 0) > 0) {
      state.branch.cursor = i;
      state.highlightAction = null;
      update();
      return;
    }
  }
  if (b.parentBranchId) {
    const parent = state.branch.branches.find((x) => x.id === b.parentBranchId);
    if (parent) {
      state.branch.activeBranchId = parent.id;
      let target = Math.max(parent.forkPly, Math.min(b.forkPly - 1, parent.frames.length - 1));
      for (let i = target; i >= parent.forkPly; i--) {
        const fr = parent.frames[i];
        if (fr.seat === state.povSeat && fr.event_kind !== "round_end" && (fr.legal_actions?.length ?? 0) > 0) {
          target = i;
          break;
        }
      }
      state.branch.cursor = target;
      state.highlightAction = null;
      update();
      return;
    }
  }
  state.branch.activeBranchId = "main";
  state.branch.cursor = 0;
  state.highlightAction = null;
  update();
}

function deleteActiveBranch(): void {
  if (!state.branch.active) return;
  if (state.branch.activeBranchId === "main") {
    clearBranchState(true);
    update();
    return;
  }
  const target = getActiveBranch(state.branch);
  const toDelete = new Set<string>([state.branch.activeBranchId]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const x of state.branch.branches) {
      if (x.parentBranchId && toDelete.has(x.parentBranchId) && !toDelete.has(x.id)) {
        toDelete.add(x.id);
        changed = true;
      }
    }
  }
  state.branch.branches = state.branch.branches.filter((x) => !toDelete.has(x.id));
  if (state.branch.branches.length === 0) {
    clearBranchState(true);
    update();
    return;
  }
  const parent = target?.parentBranchId
    ? state.branch.branches.find((x) => x.id === target.parentBranchId)
    : undefined;
  const fallback = parent ?? state.branch.branches[state.branch.branches.length - 1];
  state.branch.activeBranchId = fallback.id;
  state.branch.cursor = Math.max(0, fallback.frames.length - 1);
  state.highlightAction = null;
  update();
}

function loadReport(raw: ReviewReport, sessionId: string | null): void {
  const rep = normalizeReport(raw);
  state.report = rep;
  state.sessionId = sessionId;
  const firstSeat = Object.keys(rep.seats)[0];
  state.povSeat = firstSeat !== undefined ? parseInt(firstSeat, 10) : 0;
  state.frameIdx = 0;
  state.mode = "review";
  state.reviewTab = "analysis";
  state.wwydAnswers = {};
  state.highlightAction = null;
  clearBranchState(false);
  update();
}

function mountShell(): void {
  root.innerHTML = `<div class="app-shell">
    <header class="topbar">
      <div class="tb-left">
        <span class="brand">立直复盘 · V6</span>
        <div class="mode-seg" role="tablist" aria-label="视图模式">
          <button class="seg-btn" data-mode="review">复盘</button>
          <button class="seg-btn" data-mode="wwyd">何切</button>
          <button class="seg-btn" data-mode="sandbox">推演</button>
        </div>
      </div>

      <div class="seat-tabs" id="slotSeatTabs"></div>

      <div class="tb-right">
        <button class="btn" id="handsBtn">透视</button>
        <button class="btn" id="aiToggleBtn">AI柱</button>
        <button class="btn" id="dangerToggleBtn">危险条</button>
        <button class="btn" id="reportBtn">报告</button>
        <button class="btn" id="themeBtn">浅色</button>
        <button class="btn" id="importBtn"><span class="gpu-dot" id="gpuDot"></span>牌谱</button>
        <button class="btn" id="optionsBtn">选项</button>
        <button class="btn" id="helpBtn">?</button>
      </div>
    </header>

    <main class="workspace">
      <section class="board-stage" id="boardStage">
        <div class="kd-board" id="kdBoard">
          <div class="kd-opp kd-opp-top" id="slotOppTop"></div>
          <div class="kd-opp kd-opp-left" id="slotOppLeft"></div>
          <div class="kd-opp kd-opp-right" id="slotOppRight"></div>

          <div class="kd-pond kd-pond-top" id="slotPondTop"></div>
          <div class="kd-pond kd-pond-left" id="slotPondLeft"></div>
          <div class="kd-pond kd-pond-right" id="slotPondRight"></div>
          <div class="kd-pond kd-pond-bottom" id="slotPondBottom"></div>

          <div class="kd-center" id="slotCenter"></div>

          <div class="kd-ego-zone">
            <div class="action-bar empty" id="slotActionBar"></div>
            <div class="ego-area" id="slotEgoArea"></div>
          </div>
        </div>
      </section>

      <aside class="inspector" id="sidebarHost"></aside>
    </main>

    <div class="modal-backdrop hidden" id="modalHost"></div>
    <div class="toast-host hidden" id="toastHost"></div>
  </div>`;
}

function getHideAnswerAndUserAns(f: ReviewFrame): { hideAnswer: boolean; userAns: number | undefined } {
  if (state.mode !== "wwyd" || state.branch.active) {
    return { hideAnswer: false, userAns: undefined };
  }
  const userAns = state.wwydAnswers[f.idx];
  const isPovDec = f.seat === state.povSeat && (f.is_decision ?? true) && f.event_kind !== "round_end";
  return { hideAnswer: isPovDec && userAns === undefined, userAns };
}

function update(): void {
  const $ = (id: string) => document.getElementById(id)!;
  document.documentElement.dataset.theme = state.theme;

  // Mode segmented control
  root.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === state.mode);
  });

  // Topbar toggles
  const allOppRev = [0, 1, 2, 3].filter((p) => p !== state.povSeat).every((p) => state.revealSeats[p]);
  $("handsBtn").classList.toggle("active", allOppRev);
  $("aiToggleBtn").classList.toggle("active", state.showAI);
  $("dangerToggleBtn").classList.toggle("active", state.showDanger);
  $("themeBtn").textContent = state.theme === "light" ? "深色" : "浅色";

  const gpuDot = $("gpuDot");
  gpuDot.className = `gpu-dot ${state.backendOnline ? "on" : "off"}`;

  const rep = state.report;
  if (!rep) {
    $("slotCenter").innerHTML = `<div class="gi-core"><div class="muted">等待加载牌谱</div></div>`;
    renderModal(state, undefined);
    return;
  }

  const names =
    rep.player_names ?? [0, 1, 2, 3].map((p) => rep.seats[String(p)]?.name || `玩家${p + 1}`);
  $("slotSeatTabs").innerHTML = [0, 1, 2, 3]
    .map((p) => {
      const s = rep.seats[String(p)];
      const act = p === state.povSeat ? " active" : "";
      const rt = s ? s.rating.toFixed(1) : "-";
      return `<button class="seat-btn${act}" data-seat="${p}"><span class="sb-name">${escHtml(playerName(names, p))}</span><span class="sb-rt">${rt}</span></button>`;
    })
    .join("");

  const f = state.branch.active
    ? getBranchDisplayFrame(state.branch, rep)
    : rep.frames[state.frameIdx];
  if (!f) return;

  const { hideAnswer, userAns } = getHideAnswerAndUserAns(f);
  renderBoard(f, state, names, hideAnswer, userAns);
  renderSidebar(f, state, hideAnswer, userAns);
  renderModal(state, f);
  renderToast();
}

function nextHeroCommitCursor(frames: ReviewFrame[], fromPly: number, povSeat: number): number {
  for (let i = fromPly + 1; i < frames.length; i++) {
    const fr = frames[i];
    if (
      fr.event_kind === "round_end" ||
      (fr.seat === povSeat && (fr.event_kind === "pending" || (fr.is_decision ?? true)))
    ) {
      return i;
    }
  }
  return Math.max(0, frames.length - 1);
}

function sharesPrefixActions(a: ReviewFrame[], b: ReviewFrame[], uptoPly: number): boolean {
  if (a.length <= uptoPly || b.length <= uptoPly) return false;
  for (let i = 0; i < uptoPly; i++) {
    if (a[i]?.human_action !== b[i]?.human_action) return false;
  }
  return true;
}

async function startOrStepBranch(f: ReviewFrame, actionId: number): Promise<void> {
  if (!state.report || state.branch.loading) return;

  if (state.branch.active) {
    if (!state.backendOnline) {
      showToast("离线模式无法推演后续摸牌，请运行 review_server.py", "warn", 3200);
      return;
    }

    // 1. Standing on main root -> open/create root branch
    if (state.branch.activeBranchId === "main") {
      const existingRoot = state.branch.branches.find(
        (x) => x.parentBranchId === null && x.frames[0]?.human_action === actionId
      );
      if (existingRoot) {
        state.branch.activeBranchId = existingRoot.id;
        state.branch.cursor = nextHeroCommitCursor(existingRoot.frames, 0, state.povSeat);
        state.highlightAction = null;
        update();
        return;
      }
      const rootFr = state.report.frames[state.branch.forkFrameIdx];
      const rMeta = (state.report.rounds ?? []).find((r) => r.round_idx === state.branch.forkRoundIdx);
      state.branch.loading = true;
      showToast("新建分支推演中...", "info", 1000);
      try {
        const resp = await startBranch({
          session_id: state.sessionId,
          round_idx: rootFr.round_idx,
          step_in_round: rootFr.step_in_round ?? 0,
          fork_frame_idx: rootFr.idx,
          pov_seat: state.povSeat,
          action: actionId,
          rule: state.report.rule,
          round_seed: rMeta && rMeta.deck && rMeta.actions ? rMeta : undefined,
        });
        const seq = state.branch.nextBranchSeq++;
        const newB: GitBranch = {
          id: `b${seq}`,
          serverBranchId: resp.branch_id,
          name: `分支 ${seq}`,
          colorIdx: seq,
          parentBranchId: null,
          forkPly: 0,
          frames: resp.frames,
          roundOver: resp.round_over,
        };
        state.branch.branches.push(newB);
        state.branch.activeBranchId = newB.id;
        state.branch.cursor = Math.max(0, resp.frames.length - 1);
        state.highlightAction = null;
        update();
      } catch (err) {
        showToast(`分叉失败: ${err instanceof Error ? err.message : String(err)}`, "error", 3200);
      } finally {
        state.branch.loading = false;
      }
      return;
    }

    // 2. Standing on a branch node -> extend tip OR fork non-destructively from earlier commit
    const b = getActiveBranch(state.branch);
    if (!b) return;
    const curPly = f.branch_ply ?? state.branch.cursor;

    // Tip extension (pending frame at branch end)
    if (f.event_kind === "pending" && curPly === b.frames.length - 1) {
      state.branch.loading = true;
      try {
        const resp = await stepBranch({
          branch_id: b.serverBranchId,
          from_branch_ply: curPly,
          action: actionId,
          fork_new: false,
        });
        b.frames = resp.frames;
        b.roundOver = resp.round_over;
        state.branch.cursor = Math.max(0, resp.frames.length - 1);
        state.highlightAction = null;
        update();
      } catch (err) {
        showToast(`推演失败: ${err instanceof Error ? err.message : String(err)}`, "error", 3200);
      } finally {
        state.branch.loading = false;
      }
      return;
    }

    // Earlier Hero commit: dedup before forking
    if (b.frames[curPly]?.human_action === actionId) {
      state.branch.cursor = nextHeroCommitCursor(b.frames, curPly, state.povSeat);
      state.highlightAction = null;
      update();
      return;
    }

    const dupBranch = state.branch.branches.find(
      (ex) =>
        ex.frames[curPly]?.human_action === actionId &&
        sharesPrefixActions(ex.frames, b.frames, curPly)
    );
    if (dupBranch) {
      state.branch.activeBranchId = dupBranch.id;
      state.branch.cursor = nextHeroCommitCursor(dupBranch.frames, curPly, state.povSeat);
      state.highlightAction = null;
      update();
      return;
    }

    // Create new sibling/child branch without overwriting existing branches
    state.branch.loading = true;
    showToast("分叉推演中...", "info", 1000);
    try {
      if (curPly === 0) {
        const rootFr = state.report.frames[state.branch.forkFrameIdx];
        const rMeta = (state.report.rounds ?? []).find((r) => r.round_idx === state.branch.forkRoundIdx);
        const resp = await startBranch({
          session_id: state.sessionId,
          round_idx: rootFr.round_idx,
          step_in_round: rootFr.step_in_round ?? 0,
          fork_frame_idx: rootFr.idx,
          pov_seat: state.povSeat,
          action: actionId,
          rule: state.report.rule,
          round_seed: rMeta && rMeta.deck && rMeta.actions ? rMeta : undefined,
        });
        const seq = state.branch.nextBranchSeq++;
        const newB: GitBranch = {
          id: `b${seq}`,
          serverBranchId: resp.branch_id,
          name: `分支 ${seq}`,
          colorIdx: seq,
          parentBranchId: null,
          forkPly: 0,
          frames: resp.frames,
          roundOver: resp.round_over,
        };
        state.branch.branches.push(newB);
        state.branch.activeBranchId = newB.id;
        state.branch.cursor = Math.max(0, resp.frames.length - 1);
      } else {
        const resp = await stepBranch({
          branch_id: b.serverBranchId,
          from_branch_ply: curPly,
          action: actionId,
          fork_new: true,
        });
        const seq = state.branch.nextBranchSeq++;
        const newB: GitBranch = {
          id: `b${seq}`,
          serverBranchId: resp.branch_id,
          name: `分支 ${seq}`,
          colorIdx: seq,
          parentBranchId: b.id,
          forkPly: curPly,
          frames: resp.frames,
          roundOver: resp.round_over,
        };
        state.branch.branches.push(newB);
        state.branch.activeBranchId = newB.id;
        state.branch.cursor = Math.max(0, resp.frames.length - 1);
      }
      state.highlightAction = null;
      update();
    } catch (err) {
      showToast(`分叉失败: ${err instanceof Error ? err.message : String(err)}`, "error", 3200);
    } finally {
      state.branch.loading = false;
    }
    return;
  }

  // 3. Starting brand new Git tree from replay
  if (!state.backendOnline) {
    showToast("离线模式无法推演后续摸牌，请运行 review_server.py", "warn", 3200);
    return;
  }
  const rMeta = (state.report.rounds ?? []).find((r) => r.round_idx === f.round_idx);
  state.branch.loading = true;
  showToast("推演中...", "info", 1000);
  try {
    const resp = await startBranch({
      session_id: state.sessionId,
      round_idx: f.round_idx,
      step_in_round: f.step_in_round ?? 0,
      fork_frame_idx: f.idx,
      pov_seat: state.povSeat,
      action: actionId,
      rule: state.report.rule,
      round_seed: rMeta && rMeta.deck && rMeta.actions ? rMeta : undefined,
    });
    const firstBranch: GitBranch = {
      id: "b1",
      serverBranchId: resp.branch_id,
      name: "分支 1",
      colorIdx: 1,
      parentBranchId: null,
      forkPly: 0,
      frames: resp.frames,
      roundOver: resp.round_over,
    };
    state.mode = "sandbox";
    state.branch = {
      active: true,
      forkFrameIdx: resp.fork_frame_idx,
      forkRoundIdx: f.round_idx,
      forkLabel: `${f.round_label} ${f.turn}巡`,
      branches: [firstBranch],
      activeBranchId: "b1",
      cursor: Math.max(0, resp.frames.length - 1),
      nextBranchSeq: 2,
      loading: false,
    };
    state.highlightAction = null;
    update();
  } catch (err) {
    state.branch.loading = false;
    showToast(`分叉失败: ${err instanceof Error ? err.message : String(err)}`, "error", 3600);
  }
}

async function handlePlayerAction(actionId: number): Promise<void> {
  if (!state.report || state.branch.loading) return;
  const { frames, cursor } = getActiveTimeline();
  const f = frames[cursor];
  if (!f) return;

  if (f.seat !== state.povSeat || f.event_kind === "round_end") {
    const actor = playerName(state.report?.player_names, f.seat);
    showToast(`当前为 ${actor} 行动帧，切回自家回合即可出牌`, "warn", 1800);
    return;
  }
  const legal = f.legal_actions ?? [];
  if (legal.length > 0 && !legal.includes(actionId)) {
    showToast("当前局面不可执行该动作", "warn", 1600);
    return;
  }

  if (state.mode === "wwyd" && !state.branch.active) {
    if (state.wwydAnswers[f.idx] === undefined) {
      state.wwydAnswers[f.idx] = actionId;
      update();
    } else {
      showToast("本题已作答，可点「下一题」或「以此手进入推演」", "info", 1800);
    }
    return;
  }

  await startOrStepBranch(f, actionId);
}

function handleRiverTileClick(originCode: number): void {
  const target = resolveRiverJump(originCode, state.branch);
  if (target.mode === "none") return;

  state.pulseRiverOrigin = originCode;
  if (pulseTimer !== null) window.clearTimeout(pulseTimer);
  pulseTimer = window.setTimeout(() => {
    state.pulseRiverOrigin = null;
    update();
  }, 600);

  if (target.mode === "branch") {
    state.branch.cursor = target.branchPly;
    update();
    return;
  }
  if (state.branch.active) {
    clearBranchState(false);
    state.mode = "review";
  } else if (state.mode === "sandbox") {
    state.mode = "review";
  }
  setActiveCursor(target.frameIdx);
}

function stepTable(dir: 1 | -1): void {
  const { frames, cursor } = getActiveTimeline();
  const next = stepTableEvent(frames, cursor, dir, state.povSeat);
  if (next !== cursor) setActiveCursor(next);
}

function stepDecision(dir: 1 | -1): void {
  const { frames, cursor } = getActiveTimeline();
  const next = stepPovDecision(frames, cursor, dir, state.povSeat);
  if (next >= 0) setActiveCursor(next);
  else showToast(`该方向已无 ${playerName(state.report?.player_names, state.povSeat)} 的决策`, "info", 1300);
}

function stepMistake(dir: 1 | -1): void {
  if (state.branch.active) return;
  const frames = state.report?.frames ?? [];
  let next = stepError(frames, state.frameIdx, dir, state.povSeat, state.errorThreshold);
  if (next < 0 && state.errorThreshold < 1.0) {
    next = stepDiff(frames, state.frameIdx, dir, state.povSeat);
  }
  if (next >= 0) setActiveCursor(next);
  else showToast("该方向已无失误或分歧手", "info", 1300);
}

function jumpRoundDir(dir: 1 | -1): void {
  if (!state.report) return;
  if (state.branch.active) clearBranchState(false);
  const next = stepRound(state.report, state.frameIdx, dir);
  setActiveCursor(next);
}

function toggleAllOpponentHands(): void {
  const opp = [0, 1, 2, 3].filter((p) => p !== state.povSeat);
  const allOn = opp.every((p) => state.revealSeats[p]);
  for (const p of opp) state.revealSeats[p] = !allOn;
  update();
}

async function ingestReviewJson(rawText: string): Promise<void> {
  const parsed = JSON.parse(rawText) as ReviewReport;
  if (!parsed.frames || !parsed.seats) {
    throw new Error("无效的评测 JSON：缺少 frames 或 seats 字段");
  }
  let sid: string | null = null;
  if (state.backendOnline) {
    try {
      const res = await uploadReportForSession(parsed);
      sid = res.session_id;
    } catch {
      sid = null;
    }
  }
  state.activeModal = null;
  loadReport(parsed, sid);
  showToast("已加载牌谱", "ok");
}

async function handleRawFile(file: File): Promise<void> {
  const text = await file.text();
  if (file.name.endsWith(".json")) {
    try {
      const obj = JSON.parse(text);
      if (obj && obj.frames && obj.seats) {
        await ingestReviewJson(text);
        return;
      }
    } catch {
      // fall through as raw replay
    }
  }
  if (!state.backendOnline) {
    showToast("导入原始牌谱 (.mjlog/.json/.jsonl) 需要本地 V6 后端在线", "error", 3200);
    return;
  }
  const ruleSelect = document.getElementById("importRuleSelect") as HTMLSelectElement | null;
  const rule = ruleSelect?.value || "7d";
  const prog = document.getElementById("importProgressMsg");
  if (prog) prog.textContent = "正在由 Sovereign-Mahjong V6 GPU 评测...";
  try {
    const res = await reviewReplayFile(text, file.name, rule);
    state.activeModal = null;
    loadReport(res.report, res.session_id);
    showToast(`评测完成：${res.report.rounds_total} 局`, "ok");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (prog) prog.textContent = `评测失败: ${msg}`;
    showToast(`评测失败: ${msg}`, "error", 3600);
  }
}

// Mount shell once & fit tabletop smoothly
mountShell();
fitBoard();
const stageEl = document.getElementById("boardStage");
if (stageEl && "ResizeObserver" in window) {
  new ResizeObserver(() => fitBoard()).observe(stageEl);
}
window.addEventListener("resize", fitBoard);

// Round Scrub Track click & drag
let scrubbing = false;
function scrubFromEvent(clientX: number): void {
  const track = document.getElementById("roundScrubTrack");
  if (!track) return;
  const start = parseInt(track.dataset.scrubStart ?? "0", 10);
  const end = parseInt(track.dataset.scrubEnd ?? "0", 10);
  const rect = track.getBoundingClientRect();
  if (rect.width <= 0) return;
  const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  const idx = Math.round(start + ratio * Math.max(0, end - start));
  if (idx !== state.frameIdx) setActiveCursor(idx);
}
root.addEventListener("pointerdown", (e) => {
  const target = e.target as HTMLElement | null;
  if (target?.closest("#roundScrubTrack")) {
    scrubbing = true;
    scrubFromEvent(e.clientX);
  }
});
window.addEventListener("pointermove", (e) => {
  if (scrubbing) scrubFromEvent(e.clientX);
});
window.addEventListener("pointerup", () => {
  scrubbing = false;
});

// Click delegation
root.addEventListener("click", (e) => {
  const target = e.target as HTMLElement | null;
  if (!target) return;

  if (target.id === "modalHost" || target.closest("#modalCloseBtn")) {
    state.activeModal = null;
    update();
    return;
  }

  const modeEl = target.closest<HTMLElement>("[data-mode]");
  if (modeEl) {
    setMode(modeEl.dataset.mode as AppMode);
    return;
  }

  const tabEl = target.closest<HTMLElement>("[data-review-tab]");
  if (tabEl) {
    state.reviewTab = tabEl.dataset.reviewTab as ReviewTab;
    update();
    return;
  }

  const riverEl = target.closest<HTMLElement>("[data-river-origin]");
  if (riverEl) {
    handleRiverTileClick(parseInt(riverEl.dataset.riverOrigin ?? "-1", 10));
    return;
  }

  const handEl = target.closest<HTMLElement>("[data-hand-action]");
  if (handEl) {
    const act = parseInt(handEl.dataset.handAction ?? "-1", 10);
    if (act >= 0) void handlePlayerAction(act);
    return;
  }

  const spEl = target.closest<HTMLElement>("[data-special-action]");
  if (spEl) {
    const act = parseInt(spEl.dataset.specialAction ?? "-1", 10);
    if (act >= 0) void handlePlayerAction(act);
    return;
  }

  const wwydForkEl = target.closest<HTMLElement>("[data-wwyd-fork]");
  if (wwydForkEl) {
    const act = parseInt(wwydForkEl.dataset.wwydFork ?? "-1", 10);
    const f = state.report?.frames[state.frameIdx];
    if (f && act >= 0) void startOrStepBranch(f, act);
    return;
  }

  if (target.closest("#aiBarsSvg")) {
    state.showAI = !state.showAI;
    update();
    return;
  }

  if (target.closest("#dangerBarsStrip")) {
    state.activeModal = "danger";
    update();
    return;
  }

  const toggleSeatEl = target.closest<HTMLElement>("[data-toggle-seat]");
  if (toggleSeatEl) {
    const s = parseInt(toggleSeatEl.dataset.toggleSeat ?? "-1", 10);
    if (s >= 0 && s < 4) {
      state.revealSeats[s] = !state.revealSeats[s];
      update();
    }
    return;
  }

  const seatEl = target.closest<HTMLElement>("[data-seat]");
  if (seatEl) {
    const newSeat = parseInt(seatEl.dataset.seat ?? "0", 10);
    if (seatEl.closest("#modalHost")) state.activeModal = null;
    if (newSeat !== state.povSeat) {
      if (state.branch.active) clearBranchState(true);
      state.povSeat = newSeat;
      if (state.mode === "wwyd") snapToNearestPovDecision();
    }
    update();
    return;
  }

  const roundEl = target.closest<HTMLElement>("[data-jump-round]");
  if (roundEl && state.report) {
    const rIdx = parseInt(roundEl.dataset.jumpRound ?? "0", 10);
    const rMeta = (state.report.rounds ?? []).find((r) => r.round_idx === rIdx);
    const targetFrame = rMeta ? rMeta.first_frame_idx : state.report.frames.findIndex((x) => x.round_idx === rIdx);
    state.activeModal = null;
    clearBranchState(false);
    if (state.mode === "sandbox") state.mode = "review";
    if (targetFrame >= 0) setActiveCursor(targetFrame);
    return;
  }

  const jumpEl = target.closest<HTMLElement>("[data-jump-frame]");
  if (jumpEl) {
    const idx = parseInt(jumpEl.dataset.jumpFrame ?? "-1", 10);
    if (idx >= 0) {
      state.activeModal = null;
      clearBranchState(false);
      state.mode = "review";
      setActiveCursor(idx);
    }
    return;
  }

  const gitRootEl = target.closest<HTMLElement>("[data-git-root]");
  if (gitRootEl && state.branch.active) {
    state.branch.activeBranchId = "main";
    state.branch.cursor = 0;
    state.highlightAction = null;
    update();
    return;
  }

  const gitActualEl = target.closest<HTMLElement>("[data-git-main-actual]");
  if (gitActualEl && state.branch.active && state.report) {
    const targetFrame = Math.min(state.branch.forkFrameIdx + 1, state.report.frames.length - 1);
    clearBranchState(false);
    state.mode = "review";
    setActiveCursor(targetFrame);
    return;
  }

  const gitBranchEl = target.closest<HTMLElement>("[data-git-branch]");
  if (gitBranchEl && state.branch.active) {
    const bId = gitBranchEl.dataset.gitBranch ?? "";
    const ply = parseInt(gitBranchEl.dataset.gitPly ?? "0", 10);
    if (bId && state.branch.branches.some((b) => b.id === bId)) {
      state.branch.activeBranchId = bId;
      state.branch.cursor = Math.max(0, ply);
      state.highlightAction = null;
      update();
    }
    return;
  }

  const branchStepEl = target.closest<HTMLElement>("[data-branch-step]");
  if (branchStepEl && state.branch.active) {
    const idx = parseInt(branchStepEl.dataset.branchStep ?? "-1", 10);
    if (idx >= 0) setActiveCursor(idx);
    return;
  }

  const candRow = target.closest<HTMLElement>("[data-cand-action]");
  if (candRow) {
    const act = parseInt(candRow.dataset.candAction ?? "-1", 10);
    state.highlightAction = state.highlightAction === act ? null : act;
    update();
    return;
  }

  const threshEl = target.closest<HTMLElement>("[data-set-thresh]");
  if (threshEl) {
    state.errorThreshold = parseFloat(threshEl.dataset.setThresh ?? "0.2");
    update();
    return;
  }

  const btn = target.closest<HTMLButtonElement>("button");
  if (!btn) return;
  switch (btn.id) {
    case "handsBtn":
    case "optToggleHands":
      toggleAllOpponentHands();
      break;
    case "aiToggleBtn":
    case "optToggleAI":
    case "aiRevealInlineBtn":
      state.showAI = !state.showAI;
      update();
      break;
    case "dangerToggleBtn":
    case "optToggleDanger":
      state.showDanger = !state.showDanger;
      update();
      break;
    case "themeBtn":
    case "optToggleTheme":
      state.theme = state.theme === "light" ? "dark" : "light";
      update();
      break;
    case "resetWwydBtn":
      state.wwydAnswers = {};
      update();
      break;
    case "wwydNextBtn":
      stepDecision(1);
      break;
    case "undoBranchBtn":
      undoBranchStep();
      break;
    case "deleteBranchBtn":
      deleteActiveBranch();
      break;
    case "exitBranchBtn":
    case "exitSandboxBtn":
      clearBranchState(true);
      state.mode = "review";
      update();
      break;
    case "prevPlyBtn":
      stepTable(-1);
      break;
    case "nextPlyBtn":
      stepTable(1);
      break;
    case "prevDecBtn":
      stepDecision(-1);
      break;
    case "nextDecBtn":
      stepDecision(1);
      break;
    case "prevErrBtn":
      stepMistake(-1);
      break;
    case "nextErrBtn":
      stepMistake(1);
      break;
    case "prevRoundBtn":
      jumpRoundDir(-1);
      break;
    case "nextRoundBtn":
      jumpRoundDir(1);
      break;
    case "openRoundOverviewBtn":
      state.activeModal = "round-overview";
      update();
      break;
    case "openThisRoundBtn":
      state.activeModal = "this-round";
      update();
      break;
    case "openDangerBtn":
      state.activeModal = "danger";
      update();
      break;
    case "reportBtn":
    case "openReportBtn":
      state.activeModal = "report";
      update();
      break;
    case "importBtn":
      state.activeModal = "import";
      update();
      break;
    case "optionsBtn":
      state.activeModal = "options";
      update();
      break;
    case "helpBtn":
      state.activeModal = "help";
      update();
      break;
    case "importUrlBtn": {
      const urlInput = document.getElementById("importUrlInput") as HTMLInputElement | null;
      const ruleSel = document.getElementById("importRuleSelect") as HTMLSelectElement | null;
      const prog = document.getElementById("importProgressMsg");
      const url = urlInput?.value.trim() ?? "";
      if (!url) {
        showToast("请输入天凤牌谱链接", "warn");
        break;
      }
      if (prog) prog.textContent = "正在下载并由 Sovereign-Mahjong V6 GPU 评测...";
      void reviewReplayUrl(url, ruleSel?.value || "7d")
        .then((res) => {
          state.activeModal = null;
          loadReport(res.report, res.session_id);
          showToast(`评测完成：${res.report.rounds_total} 局`, "ok");
        })
        .catch((err) => {
          const msg = err instanceof Error ? err.message : String(err);
          if (prog) prog.textContent = `评测失败: ${msg}`;
          showToast(`评测失败: ${msg}`, "error", 3600);
        });
      break;
    }
  }
});

// Right-click on tabletop -> Tsumogiri (71) or Pass (84)
root.addEventListener("contextmenu", (e) => {
  const target = e.target as HTMLElement | null;
  if (!target || !target.closest("#kdBoard")) return;
  if (target.closest("[data-river-origin], button, input, select")) return;
  e.preventDefault();

  const { frames, cursor } = getActiveTimeline();
  const f = frames[cursor];
  if (!f || state.report === null) return;
  if (f.seat !== state.povSeat || f.event_kind === "round_end") {
    const actor = playerName(state.report?.player_names, f.seat);
    showToast(`当前为 ${actor} 行动帧，切回自家回合即可摸切`, "warn", 1500);
    return;
  }
  const legal = f.legal_actions ?? [];
  if (legal.includes(71)) {
    void handlePlayerAction(71);
  } else if (f.last_draw >= 0 && legal.includes(f.last_draw)) {
    void handlePlayerAction(f.last_draw);
  } else if (legal.includes(84)) {
    void handlePlayerAction(84);
  } else {
    showToast("当前请左键点击合法手牌或动作按钮", "info", 1600);
  }
});

root.addEventListener("change", (e) => {
  const target = e.target as HTMLInputElement | null;
  if (!target || !target.files?.length) return;
  const file = target.files[0];
  if (target.id === "importReviewJsonInput") {
    void file.text().then(ingestReviewJson).catch((err) => showToast(String(err), "error", 3200));
  } else if (target.id === "importRawReplayInput") {
    void handleRawFile(file);
  }
});

// Mouse wheel fast forward/backward
let wheelAccum = 0;
let lastWheelTime = 0;

window.addEventListener(
  "wheel",
  (e) => {
    if (!state.report || state.activeModal) return;
    const target = e.target as HTMLElement | null;
    if (target?.closest("select, textarea, input")) return;

    const scroller = target?.closest<HTMLElement>(".tab-scroll");
    if (scroller && scroller.scrollHeight > scroller.clientHeight + 2) return;

    e.preventDefault();
    const now = performance.now();
    if (now - lastWheelTime > 220 || (wheelAccum !== 0 && Math.sign(e.deltaY) !== Math.sign(wheelAccum))) {
      wheelAccum = 0;
    }
    lastWheelTime = now;
    wheelAccum += e.deltaY;

    if (Math.abs(wheelAccum) >= 24) {
      const dir: 1 | -1 = wheelAccum > 0 ? 1 : -1;
      wheelAccum = 0;
      if (state.mode === "wwyd") {
        if (e.shiftKey) stepTable(dir);
        else stepDecision(dir);
      } else {
        if (e.shiftKey) stepDecision(dir);
        else stepTable(dir);
      }
    }
  },
  { passive: false }
);

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  const tag = (e.target as HTMLElement | null)?.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

  if (e.key === "Escape") {
    if (state.activeModal) {
      state.activeModal = null;
      update();
      return;
    }
    if (state.branch.active || state.mode === "sandbox") {
      clearBranchState(true);
      state.mode = "review";
      update();
      return;
    }
  }

  if (state.activeModal) return;

  if (e.key === "ArrowLeft") {
    e.preventDefault();
    stepTable(-1);
  } else if (e.key === "ArrowRight" || e.key === " ") {
    e.preventDefault();
    stepTable(1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    stepDecision(-1);
  } else if (e.key === "ArrowDown") {
    e.preventDefault();
    stepDecision(1);
  } else if (e.key === "," || e.key === "PageUp") {
    e.preventDefault();
    stepMistake(-1);
  } else if (e.key === "." || e.key === "PageDown") {
    e.preventDefault();
    stepMistake(1);
  } else if (e.key === "[" || e.key === "Home") {
    e.preventDefault();
    jumpRoundDir(-1);
  } else if (e.key === "]" || e.key === "End") {
    e.preventDefault();
    jumpRoundDir(1);
  } else if (e.key === "h" || e.key === "H") {
    toggleAllOpponentHands();
  } else if (e.key === "m" || e.key === "M") {
    state.showAI = !state.showAI;
    update();
  } else if (e.key === "d" || e.key === "D") {
    state.showDanger = !state.showDanger;
    update();
  } else if (e.key === "q" || e.key === "Q") {
    setMode(state.mode === "wwyd" ? "review" : "wwyd");
  } else if (e.key === "z" || e.key === "Z") {
    state.activeModal = "danger";
    update();
  } else if (e.key === "g" || e.key === "G") {
    state.activeModal = "report";
    update();
  } else if (e.key === "r" || e.key === "R") {
    state.activeModal = "round-overview";
    update();
  } else if (e.key === "o" || e.key === "O") {
    state.activeModal = "options";
    update();
  } else if (e.key === "?") {
    state.activeModal = "help";
    update();
  } else if (["1", "2", "3", "4"].includes(e.key) && state.report) {
    const s = parseInt(e.key, 10) - 1;
    if (s !== state.povSeat) {
      if (state.branch.active) clearBranchState(true);
      state.povSeat = s;
      if (state.mode === "wwyd") snapToNearestPovDecision();
      update();
    }
  }
});

// Drag & drop replay or JSON
window.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("drag-over");
});
window.addEventListener("dragleave", () => document.body.classList.remove("drag-over"));
window.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file) void handleRawFile(file);
});

async function boot(): Promise<void> {
  const health = await checkHealth();
  state.backendOnline = health !== null;
  state.backendInfo = health;
  update();

  const params = new URLSearchParams(window.location.search);
  const fileUrl = params.get("file");

  if (fileUrl) {
    try {
      const resp = await fetch(fileUrl);
      if (resp.ok) {
        await ingestReviewJson(await resp.text());
        return;
      }
    } catch {
      // fall through
    }
  }

  if (state.backendOnline && health?.has_sample) {
    try {
      const res = await fetchSample();
      loadReport(res.report, res.session_id);
      return;
    } catch {
      // fall through
    }
  }

  try {
    const resp = await fetch("./sample_b70ecabe.json");
    if (resp.ok) {
      const rep = (await resp.json()) as ReviewReport;
      if (rep.frames && rep.seats) {
        let sid: string | null = null;
        if (state.backendOnline) {
          try {
            const up = await uploadReportForSession(rep);
            sid = up.session_id;
          } catch {
            sid = null;
          }
        }
        loadReport(rep, sid);
      }
    }
  } catch {
    showToast("未找到默认牌谱，请点击右上角「牌谱」导入", "info", 3200);
  }
}

void boot();

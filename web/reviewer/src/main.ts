import "./style.css";
import { fitBoard, getFilteredFrames, mountShell, patchView } from "./renderer";
import type { AppState, FilterMode, ReviewFrame, ReviewReport } from "./types";

const root = document.getElementById("app")!;

const state: AppState = {
  report: null,
  seat: 0,
  filter: "all",
  revealSeats: [false, false, false, false],
  quizMode: false,
  quizAnswers: {},
  cursor: 0,
};

function update(): void {
  patchView(state);
}

function loadReport(rep: ReviewReport): void {
  state.report = rep;
  const firstSeat = Object.keys(rep.seats)[0];
  state.seat = firstSeat !== undefined ? parseInt(firstSeat, 10) : 0;
  state.filter = "all";
  state.quizAnswers = {};
  state.cursor = 0;
  update();
}

function setCursor(idx: number): void {
  const frames = getFilteredFrames(state);
  if (!frames.length) return;
  state.cursor = Math.max(0, Math.min(idx, frames.length - 1));
  update();
}

function jumpBy(dir: 1 | -1, pred: (f: ReviewFrame) => boolean): boolean {
  const frames = getFilteredFrames(state);
  for (let i = state.cursor + dir; i >= 0 && i < frames.length; i += dir) {
    if (pred(frames[i])) {
      state.cursor = i;
      update();
      return true;
    }
  }
  return false;
}

function jumpBlunder(dir: 1 | -1): void {
  if (!jumpBy(dir, (f) => f.is_blunder)) jumpBy(dir, (f) => !f.matched_top1);
}

function jumpDiff(dir: 1 | -1): void {
  jumpBy(dir, (f) => !f.matched_top1);
}

function jumpCall(dir: 1 | -1): void {
  jumpBy(dir, (f) => !f.is_discard || f.best_action >= 72 || f.human_action >= 72);
}

function jumpRound(dir: 1 | -1): void {
  const frames = getFilteredFrames(state);
  if (!frames.length) return;
  const curR = frames[state.cursor].round_idx;
  jumpBy(dir, (f) => f.round_idx !== curR);
}

function toggleAllOpponentHands(): void {
  const oppSeats = [0, 1, 2, 3].filter((p) => p !== state.seat);
  const allOn = oppSeats.every((p) => state.revealSeats[p]);
  for (const p of oppSeats) state.revealSeats[p] = !allOn;
  update();
}

function toggleQuizMode(): void {
  state.quizMode = !state.quizMode;
  update();
}

function submitQuizAnswer(actionId: number): void {
  const frames = getFilteredFrames(state);
  const f = frames[state.cursor];
  if (!f || !state.quizMode) return;
  if (state.quizAnswers[f.idx] !== undefined) return; // Locked once answered!
  state.quizAnswers[f.idx] = actionId;
  update();
}

function handleFile(file: File): void {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = JSON.parse(String(reader.result)) as ReviewReport;
      if (!parsed.frames || !parsed.seats) {
        alert("无效的评测 JSON：缺少 frames 或 seats 字段");
        return;
      }
      loadReport(parsed);
    } catch (err) {
      alert(`JSON 解析失败: ${err}`);
    }
  };
  reader.readAsText(file, "utf-8");
}

// Mount static shell ONCE and bind event delegation ONCE -> zero DOM thrashing!
mountShell(root);
fitBoard();
window.addEventListener("resize", fitBoard);

root.addEventListener("click", (e) => {
  const target = e.target as HTMLElement | null;
  if (!target) return;

  const quizReset = target.closest<HTMLElement>("#resetQuizBtn");
  if (quizReset) {
    state.quizAnswers = {};
    update();
    return;
  }

  const seatBtn = target.closest<HTMLElement>("[data-seat]");
  if (seatBtn) {
    state.seat = parseInt(seatBtn.dataset.seat ?? "0", 10);
    state.cursor = 0;
    update();
    return;
  }

  const stepEl = target.closest<HTMLElement>("[data-step]");
  if (stepEl) {
    setCursor(parseInt(stepEl.dataset.step ?? "0", 10));
    return;
  }

  const jumpEl = target.closest<HTMLElement>("[data-jump-idx]");
  if (jumpEl) {
    const idx = parseInt(jumpEl.dataset.jumpIdx ?? "-1", 10);
    state.filter = "all";
    const frames = getFilteredFrames(state);
    const pos = frames.findIndex((f) => f.idx === idx);
    if (pos >= 0) setCursor(pos);
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

  const quizActEl = target.closest<HTMLElement>("[data-quiz-action]");
  if (quizActEl) {
    const act = parseInt(quizActEl.dataset.quizAction ?? "-1", 10);
    if (act >= 0) submitQuizAnswer(act);
    return;
  }

  const btn = target.closest<HTMLButtonElement>("button");
  if (!btn) return;
  switch (btn.id) {
    case "quizBtn": toggleQuizMode(); break;
    case "handsToggleBtn": toggleAllOpponentHands(); break;
    case "prevStepBtn": setCursor(state.cursor - 1); break;
    case "nextStepBtn": setCursor(state.cursor + 1); break;
    case "prevDiffBtn": jumpDiff(-1); break;
    case "nextDiffBtn": jumpDiff(1); break;
    case "prevBlunderBtn": jumpBlunder(-1); break;
    case "nextBlunderBtn": jumpBlunder(1); break;
    case "nextCallBtn": jumpCall(1); break;
    case "prevRoundBtn": jumpRound(-1); break;
    case "nextRoundBtn": jumpRound(1); break;
  }
});

root.addEventListener("change", (e) => {
  const target = e.target as HTMLElement | null;
  if (!target) return;
  if (target.id === "filterSelect") {
    state.filter = (target as HTMLSelectElement).value as FilterMode;
    state.cursor = 0;
    update();
  } else if (target.id === "fileInput") {
    const f = (target as HTMLInputElement).files?.[0];
    if (f) handleFile(f);
  }
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  const tag = (e.target as HTMLElement | null)?.tagName;
  if (tag === "SELECT" || tag === "INPUT") return;
  if (e.key === "ArrowLeft") setCursor(state.cursor - 1);
  else if (e.key === "ArrowRight" || e.key === " ") {
    e.preventDefault();
    setCursor(state.cursor + 1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    jumpBlunder(-1);
  } else if (e.key === "ArrowDown") {
    e.preventDefault();
    jumpBlunder(1);
  } else if (e.key === "[") jumpDiff(-1);
  else if (e.key === "]") jumpDiff(1);
  else if (e.key === "c" || e.key === "C") jumpCall(-1);
  else if (e.key === "v" || e.key === "V") jumpCall(1);
  else if (e.key === "h" || e.key === "H") toggleAllOpponentHands();
  else if (e.key === "q" || e.key === "Q") toggleQuizMode();
  else if (["1", "2", "3", "4"].includes(e.key) && state.report) {
    const s = parseInt(e.key, 10) - 1;
    if (state.report.seats[String(s)]) {
      state.seat = s;
      state.cursor = 0;
      update();
    }
  }
});

// Drag & drop JSON anywhere
window.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("drag-over");
});
window.addEventListener("dragleave", () => document.body.classList.remove("drag-over"));
window.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

async function boot(): Promise<void> {
  const params = new URLSearchParams(window.location.search);
  const fileUrl = params.get("file") || "./sample_b70ecabe.json";
  try {
    const resp = await fetch(fileUrl);
    if (resp.ok) {
      const rep = (await resp.json()) as ReviewReport;
      if (rep.frames && rep.seats) loadReport(rep);
    }
  } catch {
    // Wait for drag-and-drop JSON
  }
}

boot();

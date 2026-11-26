import "./style.css";
import { getFilteredFrames, renderApp } from "./renderer";
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

function jumpByPredicate(dir: 1 | -1, pred: (f: ReviewFrame) => boolean): boolean {
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
  if (!jumpByPredicate(dir, (f) => f.is_blunder)) {
    jumpByPredicate(dir, (f) => !f.matched_top1);
  }
}

function jumpDiff(dir: 1 | -1): void {
  jumpByPredicate(dir, (f) => !f.matched_top1);
}

function jumpCall(dir: 1 | -1): void {
  jumpByPredicate(dir, (f) => !f.is_discard || f.best_action >= 72 || f.human_action >= 72);
}

function jumpRound(dir: 1 | -1): void {
  const frames = getFilteredFrames(state);
  if (!frames.length) return;
  const curRound = frames[state.cursor].round_idx;
  jumpByPredicate(dir, (f) => f.round_idx !== curRound);
}

function toggleAllOpponentHands(): void {
  const oppSeats = [0, 1, 2, 3].filter((p) => p !== state.seat);
  const allOn = oppSeats.every((p) => state.revealSeats[p]);
  for (const p of oppSeats) {
    state.revealSeats[p] = !allOn;
  }
  update();
}

function toggleQuizMode(): void {
  state.quizMode = !state.quizMode;
  update();
}

/** Record answer once per frame.idx — cannot be re-answered or double-counted! */
function submitQuizAnswer(actionId: number): void {
  const frames = getFilteredFrames(state);
  const f = frames[state.cursor];
  if (!f || !state.quizMode) return;
  if (state.quizAnswers[f.idx] !== undefined) return; // Already answered -> locked!

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

function bindEvents(): void {
  root.querySelectorAll<HTMLButtonElement>("[data-seat]").forEach((btn) => {
    btn.onclick = () => {
      state.seat = parseInt(btn.dataset.seat ?? "0", 10);
      state.cursor = 0;
      update();
    };
  });

  root.querySelectorAll<HTMLElement>("[data-step]").forEach((el) => {
    el.onclick = () => setCursor(parseInt(el.dataset.step ?? "0", 10));
  });

  root.querySelectorAll<HTMLElement>("[data-jump-idx]").forEach((item) => {
    item.onclick = () => {
      const targetIdx = parseInt(item.dataset.jumpIdx ?? "-1", 10);
      state.filter = "all";
      const frames = getFilteredFrames(state);
      const pos = frames.findIndex((f) => f.idx === targetIdx);
      if (pos >= 0) {
        state.cursor = pos;
        update();
      }
    };
  });

  // Per-opponent hand reveal/hide click
  root.querySelectorAll<HTMLElement>("[data-toggle-seat]").forEach((zone) => {
    zone.onclick = (e) => {
      e.stopPropagation();
      const s = parseInt(zone.dataset.toggleSeat ?? "-1", 10);
      if (s >= 0 && s < 4) {
        state.revealSeats[s] = !state.revealSeats[s];
        update();
      }
    };
  });

  // Quiz mode clicks (hand tiles or call/riichi/PASS buttons)
  root.querySelectorAll<HTMLElement>("[data-quiz-action]").forEach((el) => {
    el.onclick = () => {
      const act = parseInt(el.dataset.quizAction ?? "-1", 10);
      if (act >= 0) submitQuizAnswer(act);
    };
  });

  const filterSel = root.querySelector<HTMLSelectElement>("#filterSelect");
  if (filterSel) {
    filterSel.onchange = () => {
      state.filter = filterSel.value as FilterMode;
      state.cursor = 0;
      update();
    };
  }

  const bindClick = (id: string, fn: () => void) => {
    const el = root.querySelector<HTMLButtonElement>(id);
    if (el) el.onclick = fn;
  };
  bindClick("#quizBtn", toggleQuizMode);
  bindClick("#handsToggleBtn", toggleAllOpponentHands);
  bindClick("#resetQuizBtn", () => {
    state.quizAnswers = {};
    update();
  });
  bindClick("#prevStepBtn", () => setCursor(state.cursor - 1));
  bindClick("#nextStepBtn", () => setCursor(state.cursor + 1));
  bindClick("#prevDiffBtn", () => jumpDiff(-1));
  bindClick("#nextDiffBtn", () => jumpDiff(1));
  bindClick("#prevBlunderBtn", () => jumpBlunder(-1));
  bindClick("#nextBlunderBtn", () => jumpBlunder(1));
  bindClick("#nextCallBtn", () => jumpCall(1));
  bindClick("#prevRoundBtn", () => jumpRound(-1));
  bindClick("#nextRoundBtn", () => jumpRound(1));

  root.querySelectorAll<HTMLInputElement>("#fileInput, #fileInputEmpty").forEach((inp) => {
    inp.onchange = () => {
      const f = inp.files?.[0];
      if (f) handleFile(f);
    };
  });
}

function update(): void {
  // Preserve scroll positions of scrollable lists to prevent visual jump
  const oldBlunderScroll = root.querySelector(".blunder-list")?.scrollTop ?? 0;
  const oldRoundScroll = root.querySelector(".round-strip")?.scrollLeft ?? 0;
  renderApp(root, state);
  bindEvents();
  const newBlunder = root.querySelector(".blunder-list");
  if (newBlunder) newBlunder.scrollTop = oldBlunderScroll;
  const newRound = root.querySelector(".round-strip");
  if (newRound) newRound.scrollLeft = oldRoundScroll;
}

// Keyboard shortcuts (KillerDucky / Mortal style)
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
window.addEventListener("dragleave", () => {
  document.body.classList.remove("drag-over");
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

async function boot(): Promise<void> {
  update();
  const params = new URLSearchParams(window.location.search);
  const fileUrl = params.get("file") || "./sample_b70ecabe.json";
  try {
    const resp = await fetch(fileUrl);
    if (resp.ok) {
      const rep = (await resp.json()) as ReviewReport;
      if (rep.frames && rep.seats) {
        loadReport(rep);
      }
    }
  } catch {
    // Wait for user file upload / drag-and-drop
  }
}

boot();

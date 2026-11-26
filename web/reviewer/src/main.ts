import "./style.css";
import { getFilteredFrames, renderApp } from "./renderer";
import type { AppState, FilterMode, ReviewReport } from "./types";

const root = document.getElementById("app")!;

const state: AppState = {
  report: null,
  seat: 0,
  filter: "all",
  xray: false,
  cursor: 0,
};

function loadReport(rep: ReviewReport): void {
  state.report = rep;
  const firstSeat = Object.keys(rep.seats)[0];
  state.seat = firstSeat !== undefined ? parseInt(firstSeat, 10) : 0;
  state.filter = "all";
  state.cursor = 0;
  update();
}

function setCursor(idx: number): void {
  const frames = getFilteredFrames(state);
  if (!frames.length) return;
  state.cursor = Math.max(0, Math.min(idx, frames.length - 1));
  update();
}

function jumpBlunder(dir: 1 | -1): void {
  const frames = getFilteredFrames(state);
  for (let i = state.cursor + dir; i >= 0 && i < frames.length; i += dir) {
    if (frames[i].is_blunder) {
      state.cursor = i;
      update();
      return;
    }
  }
  for (let i = state.cursor + dir; i >= 0 && i < frames.length; i += dir) {
    if (!frames[i].matched_top1) {
      state.cursor = i;
      update();
      return;
    }
  }
}

function jumpRound(dir: 1 | -1): void {
  const frames = getFilteredFrames(state);
  if (!frames.length) return;
  const curRound = frames[state.cursor].round_idx;
  for (let i = state.cursor + dir; i >= 0 && i < frames.length; i += dir) {
    if (frames[i].round_idx !== curRound) {
      state.cursor = i;
      update();
      return;
    }
  }
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

  root.querySelectorAll<HTMLElement>("[data-step]").forEach((dot) => {
    dot.onclick = () => setCursor(parseInt(dot.dataset.step ?? "0", 10));
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

  const roundSel = root.querySelector<HTMLSelectElement>("#roundSelect");
  if (roundSel) {
    roundSel.onchange = () => setCursor(parseInt(roundSel.value, 10) || 0);
  }

  const filterSel = root.querySelector<HTMLSelectElement>("#filterSelect");
  if (filterSel) {
    filterSel.onchange = () => {
      state.filter = filterSel.value as FilterMode;
      state.cursor = 0;
      update();
    };
  }

  const xrayBtn = root.querySelector<HTMLButtonElement>("#xrayBtn");
  if (xrayBtn) {
    xrayBtn.onclick = () => {
      state.xray = !state.xray;
      update();
    };
  }

  const bindClick = (id: string, fn: () => void) => {
    const el = root.querySelector<HTMLButtonElement>(id);
    if (el) el.onclick = fn;
  };
  bindClick("#prevStepBtn", () => setCursor(state.cursor - 1));
  bindClick("#nextStepBtn", () => setCursor(state.cursor + 1));
  bindClick("#prevBlunderBtn", () => jumpBlunder(-1));
  bindClick("#nextBlunderBtn", () => jumpBlunder(1));
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
  renderApp(root, state);
  bindEvents();
}

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
  } else if (e.key === "x" || e.key === "X") {
    state.xray = !state.xray;
    update();
  } else if (["1", "2", "3", "4"].includes(e.key) && state.report) {
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

// Initial load: `?file=...` or default `./sample_b70ecabe.json`
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

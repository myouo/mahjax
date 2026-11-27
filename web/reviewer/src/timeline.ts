import type { BranchState, ReviewFrame, ReviewReport, RoundMeta } from "./types";

function inferLegalActionsFromFrame(f: ReviewFrame): number[] {
  if (Array.isArray(f.legal_actions)) return f.legal_actions;
  const set = new Set<number>();
  for (const c of f.candidates ?? []) set.add(c.a);
  if (f.human_action >= 0) set.add(f.human_action);
  return Array.from(set);
}

/** Normalize both Schema v2 and legacy v1 reports into a unified internal representation. */
export function normalizeReport(raw: ReviewReport): ReviewReport {
  const frames = raw.frames ?? [];
  if (raw.schema_version === 2 && Array.isArray(raw.rounds) && raw.rounds.length > 0) {
    return raw;
  }

  // Legacy v1 fallback: synthesize event_kind, river originFrameIdx, and rounds[]
  const prevRiverLens = new Map<number, [number, number, number, number]>();
  const lastFrameInRound = new Map<number, number>();
  const roundMap = new Map<number, RoundMeta>();

  const normFrames: ReviewFrame[] = frames.map((f, i) => {
    const rIdx = f.round_idx;
    if (!prevRiverLens.has(rIdx)) {
      prevRiverLens.set(rIdx, [0, 0, 0, 0]);
    }
    const prevLens = prevRiverLens.get(rIdx)!;
    const prevOrigin = lastFrameInRound.get(rIdx) ?? i;

    const rivers = f.rivers.map((col, seat) =>
      col.map((tup, slotIdx) => {
        if (tup.length >= 5) return tup;
        const origin = slotIdx >= prevLens[seat] ? prevOrigin : Math.max(0, i - 1);
        return [tup[0], tup[1], tup[2], tup[3], origin] as [number, number, number, number, number];
      })
    ) as ReviewFrame["rivers"];

    for (let p = 0; p < 4; p++) prevLens[p] = rivers[p].length;
    lastFrameInRound.set(rIdx, i);

    const event_kind =
      f.event_kind ??
      (f.is_discard
        ? "discard"
        : f.human_action === 72
        ? "riichi"
        : f.human_action === 73 || f.human_action === 74
        ? "win"
        : f.human_action === 84
        ? "pass"
        : "call");

    const discard_tile =
      f.discard_tile ??
      (f.is_discard ? (f.human_action === 71 ? f.last_draw : f.human_action) : -1);

    const existing = roundMap.get(rIdx);
    if (!existing) {
      roundMap.set(rIdx, {
        round_idx: rIdx,
        round_num: rIdx,
        round_label: f.round_label,
        dealer: f.dealer,
        honba: f.honba,
        kyotaku: f.kyotaku,
        init_scores: [...f.scores] as [number, number, number, number],
        end_scores: [...f.scores] as [number, number, number, number],
        deltas: [0, 0, 0, 0],
        first_frame_idx: i,
        last_frame_idx: i,
        blunder_counts: [
          f.seat === 0 && f.is_blunder ? 1 : 0,
          f.seat === 1 && f.is_blunder ? 1 : 0,
          f.seat === 2 && f.is_blunder ? 1 : 0,
          f.seat === 3 && f.is_blunder ? 1 : 0,
        ],
        result: null,
      });
    } else {
      existing.last_frame_idx = i;
      existing.end_scores = [...f.scores] as [number, number, number, number];
      if (f.is_blunder) existing.blunder_counts[f.seat]++;
    }

    return {
      ...f,
      rivers,
      event_kind,
      discard_tile,
      is_decision: f.is_decision ?? true,
      has_ai: f.has_ai ?? (f.candidates?.length ?? 0) > 0,
      legal_actions: inferLegalActionsFromFrame(f),
    };
  });

  const rounds = Array.from(roundMap.values());
  for (let idx = 0; idx < rounds.length; idx++) {
    const r = rounds[idx];
    const nextInit = idx + 1 < rounds.length ? rounds[idx + 1].init_scores : r.end_scores;
    r.end_scores = [...nextInit] as [number, number, number, number];
    r.deltas = [0, 1, 2, 3].map((p) => r.end_scores[p] - r.init_scores[p]) as [number, number, number, number];
  }

  return {
    ...raw,
    schema_version: raw.schema_version ?? 1,
    rounds,
    frames: normFrames,
  };
}

export function isVisibleTableEvent(f: ReviewFrame, povSeat: number): boolean {
  const kind = f.event_kind ?? "discard";
  if (kind !== "pass") return true;
  return f.seat === povSeat && (f.is_decision ?? true);
}

/** Step +/-1 across all 4 players' visible table events (skips hidden opponent passes). */
export function stepTableEvent(
  frames: ReviewFrame[],
  cur: number,
  dir: 1 | -1,
  povSeat: number
): number {
  if (!frames.length) return 0;
  for (let i = cur + dir; i >= 0 && i < frames.length; i += dir) {
    if (isVisibleTableEvent(frames[i], povSeat)) return i;
  }
  return cur;
}

/** Step +/-1 to the next/prev decision made by `povSeat`. */
export function stepPovDecision(
  frames: ReviewFrame[],
  cur: number,
  dir: 1 | -1,
  povSeat: number
): number {
  if (!frames.length) return -1;
  for (let i = cur + dir; i >= 0 && i < frames.length; i += dir) {
    const f = frames[i];
    if (f.seat === povSeat && (f.is_decision ?? true)) return i;
  }
  return -1;
}

/** Step +/-1 to the next/prev error/blunder for `povSeat` below `threshold` (or `is_blunder`). */
export function stepError(
  frames: ReviewFrame[],
  cur: number,
  dir: 1 | -1,
  povSeat: number,
  threshold: number
): number {
  for (let i = cur + dir; i >= 0 && i < frames.length; i += dir) {
    const f = frames[i];
    if (f.seat !== povSeat) continue;
    if (!(f.is_decision ?? true) || f.has_ai === false) continue;
    if (f.matched_top1) continue;
    if (f.is_blunder || f.prob_ratio <= threshold) return i;
  }
  return -1;
}

/** Step +/-1 to any mismatch (`!matched_top1`) for `povSeat`. */
export function stepDiff(
  frames: ReviewFrame[],
  cur: number,
  dir: 1 | -1,
  povSeat: number
): number {
  for (let i = cur + dir; i >= 0 && i < frames.length; i += dir) {
    const f = frames[i];
    if (f.seat !== povSeat) continue;
    if (!(f.is_decision ?? true) || f.has_ai === false) continue;
    if (!f.matched_top1) return i;
  }
  return -1;
}

/** Jump to the first frame of the next/previous round. */
export function stepRound(
  report: ReviewReport,
  curFrameIdx: number,
  dir: 1 | -1
): number {
  const frames = report.frames;
  if (!frames.length) return 0;
  const rounds = report.rounds ?? [];
  const curRound = frames[Math.max(0, Math.min(curFrameIdx, frames.length - 1))].round_idx;
  if (rounds.length > 0) {
    const pos = rounds.findIndex((r) => r.round_idx === curRound);
    if (pos >= 0) {
      const nextPos = pos + dir;
      if (nextPos >= 0 && nextPos < rounds.length) {
        return rounds[nextPos].first_frame_idx;
      }
      return curFrameIdx;
    }
  }
  for (let i = curFrameIdx + dir; i >= 0 && i < frames.length; i += dir) {
    if (frames[i].round_idx !== curRound) {
      if (dir === 1) return i;
      const targetR = frames[i].round_idx;
      let start = i;
      while (start > 0 && frames[start - 1].round_idx === targetR) start--;
      return start;
    }
  }
  return curFrameIdx;
}

export type RiverJumpTarget =
  | { mode: "none" }
  | { mode: "replay"; frameIdx: number }
  | { mode: "branch"; branchPly: number };

export function resolveRiverJump(originCode: number, branch: BranchState): RiverJumpTarget {
  if (!Number.isFinite(originCode) || originCode === -1) return { mode: "none" };
  if (originCode >= 0) return { mode: "replay", frameIdx: originCode };
  if (originCode <= -2 && branch.active) {
    const ply = -originCode - 2;
    const activeBr = branch.branches.find((b) => b.id === branch.activeBranchId);
    if (activeBr && ply >= 0 && ply < activeBr.frames.length) {
      return { mode: "branch", branchPly: ply };
    }
  }
  return { mode: "none" };
}

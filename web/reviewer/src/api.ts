import type { BackendHealth, ReviewFrame, ReviewReport, RoundMeta } from "./types";

const BASE = "";

export interface BranchResponse {
  branch_id: string;
  fork_frame_idx: number;
  frames: ReviewFrame[];
  round_over: boolean;
}

async function parseJsonOrThrow<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // ignore non-json error body
    }
    throw new Error(detail);
  }
  return (await resp.json()) as T;
}

export async function checkHealth(): Promise<BackendHealth | null> {
  try {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), 1800);
    const resp = await fetch(`${BASE}/api/health`, { signal: ctrl.signal });
    window.clearTimeout(timer);
    if (!resp.ok) return null;
    return (await resp.json()) as BackendHealth;
  } catch {
    return null;
  }
}

export async function fetchSample(): Promise<{ session_id: string; report: ReviewReport }> {
  const resp = await fetch(`${BASE}/api/sample`);
  return parseJsonOrThrow(resp);
}

export async function uploadReportForSession(
  report: ReviewReport
): Promise<{ session_id: string; has_decks: boolean }> {
  const resp = await fetch(`${BASE}/api/load-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(report),
  });
  return parseJsonOrThrow(resp);
}

export async function reviewReplayUrl(
  url: string,
  rule: string
): Promise<{ session_id: string; report: ReviewReport }> {
  const resp = await fetch(`${BASE}/api/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, rule }),
  });
  return parseJsonOrThrow(resp);
}

export async function reviewReplayFile(
  content: string,
  filename: string,
  rule: string
): Promise<{ session_id: string; report: ReviewReport }> {
  const resp = await fetch(`${BASE}/api/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, filename, rule }),
  });
  return parseJsonOrThrow(resp);
}

export async function startBranch(params: {
  session_id: string | null;
  round_idx: number;
  step_in_round: number;
  fork_frame_idx: number;
  pov_seat: number;
  action: number;
  rule: string;
  round_seed?: RoundMeta;
}): Promise<BranchResponse> {
  const resp = await fetch(`${BASE}/api/branch/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJsonOrThrow(resp);
}

export async function stepBranch(params: {
  branch_id: string;
  from_branch_ply: number;
  action: number;
  fork_new?: boolean;
}): Promise<BranchResponse> {
  const resp = await fetch(`${BASE}/api/branch/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJsonOrThrow(resp);
}

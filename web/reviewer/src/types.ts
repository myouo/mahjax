export interface ActionCandidate {
  a: number;
  label: string;
  prob: number;
  du_ron: number;
}

export interface MeldInfo {
  kind: string;
  tiles: number[];
}

/** Compact river tile tuple: [tileId, isRiichi, isCalled, isTsumogiri] */
export type RiverTileTuple = [number, number, number, number];

export interface ReviewFrame {
  idx: number;
  round_idx: number;
  round_label: string;
  dealer: number;
  honba: number;
  kyotaku: number;
  wall_rem: number;
  dora: number[];
  turn: number;
  seat: number;
  scores: [number, number, number, number];
  riichi: [boolean, boolean, boolean, boolean];
  hands: [number[], number[], number[], number[]];
  rivers: [RiverTileTuple[], RiverTileTuple[], RiverTileTuple[], RiverTileTuple[]];
  melds: [MeldInfo[], MeldInfo[], MeldInfo[], MeldInfo[]];
  last_draw: number;
  target: number;
  is_discard: boolean;
  human_action: number;
  human_action_str: string;
  human_prob: number;
  best_action: number;
  best_action_str: string;
  best_prob: number;
  prob_ratio: number;
  matched_top1: boolean;
  is_blunder: boolean;
  candidates: ActionCandidate[];
  delta_u_ron_human: number;
  delta_u_ron_best: number;
  p_tenpai: [number, number, number];
}

export interface SeatSummary {
  seat: number;
  name: string;
  total_decisions: number;
  rating: number;
  geo_rating: number;
  overall_acc: number;
  top2_acc: number;
  top3_acc: number;
  discard_decisions: number;
  discard_acc: number;
  discard_rating: number;
  call_decisions: number;
  call_acc: number;
  call_rating: number;
  blunder_count: number;
  blunder_rate: number;
  avg_nll: number;
  perplexity: number;
}

export interface ReviewReport {
  replay_source: string;
  policy_source: string;
  rule: string;
  rounds_total: number;
  rounds_desynced: number;
  overall: SeatSummary;
  seats: Record<string, SeatSummary>;
  frames: ReviewFrame[];
}

export type FilterMode = "all" | "diff" | "blunder" | "calls";

export interface QuizStats {
  answered: number;
  matched: number;
  ratingSum: number;
}

export interface AppState {
  report: ReviewReport | null;
  seat: number;
  filter: FilterMode;
  revealSeats: [boolean, boolean, boolean, boolean];
  quizMode: boolean;
  quizAnsweredAction: number | null;
  quizStats: QuizStats;
  cursor: number;
}

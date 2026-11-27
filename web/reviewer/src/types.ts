export interface ActionCandidate {
  a: number;
  label: string;
  prob: number;
  du_ron: number;
}

export interface MeldInfo {
  kind: string;
  tiles: number[];
  called?: number | null;
  from?: number | null;
}

/**
 * River tile tuple:
 * - v2: [tileId, isRiichi, isCalled, isTsumogiri, originFrameIdx]
 * - v1 legacy: [tileId, isRiichi, isCalled, isTsumogiri]
 * Inside What-If branches, post-fork river tiles encode origin as -(branchPly + 2).
 */
export type RiverTileTuple =
  | [number, number, number, number]
  | [number, number, number, number, number];

export type EventKind =
  | "discard"
  | "riichi"
  | "call"
  | "pass"
  | "win"
  | "abort"
  | "round_end"
  | "pending";

export interface YakuEntry {
  name: string;
  nameEn?: string;
  nameKo?: string;
  nameZh?: string;
  han: number;
}

export interface WinEntry {
  seat: number;
  from: number | null;
  hand: number[];
  melds: MeldInfo[];
  winningTile: number;
  yaku: YakuEntry[];
  doraHan: number;
  akaHan: number;
  uraHan: number;
  kitaHan?: number;
  han: number;
  fu: number;
  yakuman: number;
  dora: number[];
  uraDora: number[];
  points: number;
}

export interface RoundResult {
  type: "ron" | "tsumo" | "draw" | "abortive" | "desync";
  reason: string | null;
  abortSeat?: number | null;
  round: { index: number; honba: number; kyotaku: number };
  winners: WinEntry[];
  tenpai: boolean[] | null;
  nagashiMangan?: boolean[] | null;
  deltas: [number, number, number, number];
  scores: [number, number, number, number];
  gameOver?: boolean;
  final?: unknown;
}

export interface RoundMeta {
  round_idx: number;
  round_num: number;
  round_label: string;
  dealer: number;
  honba: number;
  kyotaku: number;
  init_scores: [number, number, number, number];
  end_scores: [number, number, number, number];
  deltas: [number, number, number, number];
  first_frame_idx: number;
  last_frame_idx: number;
  blunder_counts: [number, number, number, number];
  result: RoundResult | null;
  deck?: number[];
  actions?: number[];
}

export interface ReviewFrame {
  idx: number;
  round_idx: number;
  ply?: number;
  step_in_round?: number;
  round_label: string;
  dealer: number;
  honba: number;
  kyotaku: number;
  wall_rem: number;
  dora: number[];
  ura_dora?: number[];
  turn: number;
  seat: number;
  event_kind?: EventKind;
  is_decision?: boolean;
  has_ai?: boolean;
  is_discard: boolean;
  discard_tile?: number;
  is_tsumogiri?: boolean;
  scores: [number, number, number, number];
  riichi: [boolean, boolean, boolean, boolean];
  riichi_declared?: [boolean, boolean, boolean, boolean];
  hands: [number[], number[], number[], number[]];
  rivers: [RiverTileTuple[], RiverTileTuple[], RiverTileTuple[], RiverTileTuple[]];
  melds: [MeldInfo[], MeldInfo[], MeldInfo[], MeldInfo[]];
  last_draw: number;
  target: number;
  last_discard?: { seat: number; index: number } | null;
  legal_actions?: number[];
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
  p_deal?: Record<string, [number, number, number]>;
  round_result?: RoundResult | null;
  is_branch?: boolean;
  branch_ply?: number;
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
  schema_version?: number;
  replay_source: string;
  policy_source: string;
  rule: string;
  player_names?: string[];
  rounds_total: number;
  rounds_desynced: number;
  overall: SeatSummary;
  seats: Record<string, SeatSummary>;
  rounds?: RoundMeta[];
  frames: ReviewFrame[];
}

export interface GitBranch {
  id: string;
  serverBranchId: string;
  name: string;
  colorIdx: number;
  parentBranchId: string | null;
  forkPly: number;
  frames: ReviewFrame[];
  roundOver: boolean;
}

export interface BranchState {
  active: boolean;
  forkFrameIdx: number;
  forkRoundIdx: number;
  forkLabel: string;
  branches: GitBranch[];
  activeBranchId: string; // "main" | "b1" | "b2" ...
  cursor: number;
  nextBranchSeq: number;
  loading: boolean;
}

export interface BackendHealth {
  status: string;
  device: string;
  policy_source: string;
  rules: string[];
  has_sample: boolean;
}

export type AppMode = "review" | "wwyd" | "sandbox";
export type ReviewTab = "analysis" | "mistakes" | "danger" | "stats";

export type ModalId =
  | null
  | "round-overview"
  | "this-round"
  | "danger"
  | "report"
  | "import"
  | "options"
  | "help";

export interface AppState {
  report: ReviewReport | null;
  sessionId: string | null;
  backendOnline: boolean;
  backendInfo: BackendHealth | null;
  mode: AppMode;
  reviewTab: ReviewTab;
  povSeat: number;
  frameIdx: number;
  revealSeats: [boolean, boolean, boolean, boolean];
  showAI: boolean;
  showDanger: boolean;
  /** Replay frame.idx -> chosen actionId (locked per question in WWYD mode) */
  wwydAnswers: Record<number, number>;
  branch: BranchState;
  theme: "light" | "dark";
  highlightAction: number | null;
  pulseRiverOrigin: number | null;
  activeModal: ModalId;
  errorThreshold: number;
  toast: { message: string; kind: "info" | "warn" | "error" | "ok" } | null;
}

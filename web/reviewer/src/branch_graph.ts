import { escHtml, playerName } from "./tiles";
import type { AppState, GitBranch, ReviewFrame, ReviewReport } from "./types";

const ROW_H = 36;
const LANE_W = 16;
const PAD_X = 12;

const GIT_COLORS = [
  "var(--git-main)",
  "var(--git-1)",
  "var(--git-2)",
  "var(--git-3)",
  "var(--git-4)",
  "var(--git-5)",
] as const;

export function gitLaneColor(colorIdx: number): string {
  if (colorIdx === 0) return GIT_COLORS[0];
  const paletteIdx = ((colorIdx - 1) % (GIT_COLORS.length - 1)) + 1;
  return GIT_COLORS[paletteIdx];
}

export function getActiveBranch(branch: AppState["branch"]): GitBranch | undefined {
  return branch.branches.find((b) => b.id === branch.activeBranchId);
}

export function getBranchDisplayFrame(
  branch: AppState["branch"],
  report: ReviewReport
): ReviewFrame | undefined {
  if (!branch.active) return report.frames[0];
  if (branch.activeBranchId === "main") {
    return report.frames[branch.forkFrameIdx];
  }
  const activeBr = getActiveBranch(branch) ?? branch.branches[0];
  if (!activeBr || activeBr.frames.length === 0) {
    return report.frames[branch.forkFrameIdx];
  }
  const idx = Math.max(0, Math.min(branch.cursor, activeBr.frames.length - 1));
  return activeBr.frames[idx];
}

export interface SubPlyItem {
  ply: number;
  seat: number;
  who: string;
  act: string;
  isCall: boolean;
  isHero: boolean;
}

interface GraphNode {
  id: string;
  kind: "root" | "main-actual" | "commit" | "pending" | "round_end";
  branchId: string;
  parentId: string | null;
  lane: number;
  colorIdx: number;
  startPly: number;
  endPly: number;
  turn: number;
  title: string;
  oppChip: string;
  refTag: string | null;
  subPlies: SubPlyItem[];
  childrenIds: string[];
}

function summarizeRoundEnd(fr: ReviewFrame, pov: number, names: string[]): string {
  const res = fr.round_result;
  if (!res) return "本局结束";
  const d = res.deltas?.[pov] ?? 0;
  const dStr = d === 0 ? "±0" : `${d > 0 ? "+" : ""}${d.toLocaleString()}`;
  if (res.type === "tsumo") {
    const w = res.winners?.[0];
    const who = w ? playerName(names, w.seat) : "自摸";
    return w && w.seat === pov ? `自家自摸 (${dStr})` : `${who} 自摸 (${dStr})`;
  }
  if (res.type === "ron") {
    const w = res.winners?.[0];
    const who = w ? playerName(names, w.seat) : "荣和";
    return w && w.seat === pov ? `自家荣和 (${dStr})` : `${who} 荣和 (${dStr})`;
  }
  if (res.type === "abortive") return `途中流局 (${dStr})`;
  return `荒牌流局 (${dStr})`;
}

function summarizeActualRound(
  state: AppState,
  names: string[],
  pov: number
): string {
  const rMeta = (state.report?.rounds ?? []).find(
    (r) => r.round_idx === state.branch.forkRoundIdx
  );
  if (!rMeta) return "返回实战主线";
  const d = rMeta.deltas?.[pov] ?? 0;
  const dStr = d === 0 ? "±0" : `${d > 0 ? "+" : ""}${d.toLocaleString()}`;
  const res = rMeta.result;
  if (!res) return `实战结果 (${dStr})`;
  if (res.type === "tsumo") {
    const w = res.winners?.[0];
    return w ? `${playerName(names, w.seat)} 自摸 (${dStr})` : `自摸 (${dStr})`;
  }
  if (res.type === "ron") {
    const w = res.winners?.[0];
    return w ? `${playerName(names, w.seat)} 荣和 (${dStr})` : `荣和 (${dStr})`;
  }
  if (res.type === "abortive") return `途中流局 (${dStr})`;
  return `荒牌流局 (${dStr})`;
}

function isCallEvent(fr: ReviewFrame): boolean {
  const ev = fr.event_kind;
  return ev === "call" || ev === "riichi" || ev === "win" || ev === "abort";
}

function buildBranchGraph(
  state: AppState,
  names: string[]
): {
  ordered: GraphNode[];
  nodesById: Map<string, GraphNode>;
  activeNodeId: string;
  activeNode: GraphNode | undefined;
} {
  const br = state.branch;
  const rep = state.report!;
  const pov = state.povSeat;
  const forkFrame = rep.frames[br.forkFrameIdx];
  const nodesById = new Map<string, GraphNode>();

  const rootAct = forkFrame?.human_action_str?.trim() || "当前手牌";
  const rootNode: GraphNode = {
    id: "root",
    kind: "root",
    branchId: "main",
    parentId: null,
    lane: 0,
    colorIdx: 0,
    startPly: 0,
    endPly: 0,
    turn: forkFrame?.turn ?? 1,
    title: `起点 · 实战 ${rootAct}`,
    oppChip: "",
    refTag: "main",
    subPlies: [],
    childrenIds: [],
  };
  nodesById.set("root", rootNode);

  // Track commits belonging to each branch so child branches hook to exact ancestor commit
  const branchCommits = new Map<string, GraphNode[]>();

  br.branches.forEach((b, bIdx) => {
    const lane = bIdx + 1;
    const commits: GraphNode[] = [];
    const frames = b.frames ?? [];
    const startI = Math.max(0, Math.min(b.forkPly, frames.length - 1));

    // Find commit boundary indices within b.frames starting at startI
    const boundaries: number[] = [];
    for (let i = startI; i < frames.length; i++) {
      const fr = frames[i];
      const opensCommit =
        i === startI ||
        fr.event_kind === "round_end" ||
        (fr.seat === pov && (fr.event_kind === "pending" || (fr.is_decision ?? true)));
      if (opensCommit) boundaries.push(i);
    }

    // Parent commit node of this branch's first commit
    let parentId = "root";
    if (b.parentBranchId) {
      const pList = branchCommits.get(b.parentBranchId) ?? [];
      for (const pc of pList) {
        if (pc.startPly < b.forkPly) parentId = pc.id;
      }
    }

    let prevId = parentId;
    boundaries.forEach((bStart, k) => {
      const bEnd = k + 1 < boundaries.length ? boundaries[k + 1] - 1 : frames.length - 1;
      const headFr = frames[bStart];
      const subPlies: SubPlyItem[] = [];
      let callChip = "";

      for (let j = bStart; j <= bEnd; j++) {
        const sub = frames[j];
        const who = sub.seat === pov ? "自家" : playerName(names, sub.seat);
        const act =
          sub.event_kind === "round_end"
            ? "本局结束"
            : sub.event_kind === "pending"
            ? "等待出牌"
            : sub.human_action_str || sub.event_kind || "";
        const isCall = sub.seat !== pov && isCallEvent(sub);
        if (isCall && !callChip) callChip = `${who.slice(0, 4)} ${act.split(" ")[0]}`;
        subPlies.push({
          ply: j,
          seat: sub.seat,
          who,
          act,
          isCall,
          isHero: sub.seat === pov,
        });
      }

      const isEnd = headFr.event_kind === "round_end";
      const isPending = headFr.event_kind === "pending";
      const kind: GraphNode["kind"] = isEnd ? "round_end" : isPending ? "pending" : "commit";
      const title = isEnd
        ? summarizeRoundEnd(headFr, pov, names)
        : isPending
        ? "等待出牌"
        : headFr.human_action_str || headFr.event_kind || "行动";

      const extraCount = bEnd - bStart;
      const oppChip = callChip || (extraCount > 0 && !isEnd && !isPending ? `+${extraCount}手` : "");

      const nodeId = `${b.id}@${bStart}`;
      const node: GraphNode = {
        id: nodeId,
        kind,
        branchId: b.id,
        parentId: prevId,
        lane,
        colorIdx: b.colorIdx,
        startPly: bStart,
        endPly: bEnd,
        turn: headFr.turn,
        title,
        oppChip,
        refTag: k === boundaries.length - 1 ? b.name : null,
        subPlies,
        childrenIds: [],
      };
      nodesById.set(nodeId, node);
      nodesById.get(prevId)?.childrenIds.push(nodeId);
      commits.push(node);
      prevId = nodeId;
    });

    branchCommits.set(b.id, commits);
  });

  // Add real-game main-actual reference leaf under root (appears at bottom of lane 0)
  const mainActualNode: GraphNode = {
    id: "main-actual",
    kind: "main-actual",
    branchId: "main-actual",
    parentId: "root",
    lane: 0,
    colorIdx: 0,
    startPly: 0,
    endPly: 0,
    turn: forkFrame?.turn ?? 1,
    title: `实战主线 · ${summarizeActualRound(state, names, pov)}`,
    oppChip: "原局",
    refTag: null,
    subPlies: [],
    childrenIds: [],
  };
  nodesById.set("main-actual", mainActualNode);
  rootNode.childrenIds.push("main-actual");

  // DFS order: continue same branch before visiting sibling/fork branches, with main-actual last
  const ordered: GraphNode[] = [];
  const visit = (node: GraphNode) => {
    ordered.push(node);
    const sameLane = node.childrenIds
      .map((cid) => nodesById.get(cid)!)
      .filter((c) => c && c.lane === node.lane && c.kind !== "main-actual");
    const forkChildren = node.childrenIds
      .map((cid) => nodesById.get(cid)!)
      .filter((c) => c && c.lane !== node.lane && c.kind !== "main-actual");
    const mainLeaves = node.childrenIds
      .map((cid) => nodesById.get(cid)!)
      .filter((c) => c && c.kind === "main-actual");

    for (const c of sameLane) visit(c);
    for (const c of forkChildren) visit(c);
    for (const c of mainLeaves) visit(c);
  };
  visit(rootNode);

  // Determine current HEAD node
  let activeNodeId = "root";
  if (br.activeBranchId !== "main") {
    const commits = branchCommits.get(br.activeBranchId) ?? [];
    for (const c of commits) {
      if (br.cursor >= c.startPly && br.cursor <= c.endPly) {
        activeNodeId = c.id;
        break;
      }
    }
    if (activeNodeId === "root" && commits.length > 0) {
      activeNodeId = commits[commits.length - 1].id;
    }
  }

  return { ordered, nodesById, activeNodeId, activeNode: nodesById.get(activeNodeId) };
}

export function renderGitGraphHtml(state: AppState, names: string[]): string {
  const { ordered, nodesById, activeNodeId, activeNode } = buildBranchGraph(state, names);
  const rowIndex = new Map<string, number>();
  let maxLane = 0;
  ordered.forEach((n, idx) => {
    rowIndex.set(n.id, idx);
    if (n.lane > maxLane) maxLane = n.lane;
  });

  const cx = (lane: number) => PAD_X + lane * LANE_W;
  const cy = (row: number) => row * ROW_H + ROW_H / 2;
  const svgW = PAD_X + (maxLane + 1) * LANE_W + 6;
  const svgH = ordered.length * ROW_H;

  let edgesSvg = "";
  let nodesSvg = "";

  // Draw edges bottom layer
  for (const node of ordered) {
    if (!node.parentId) continue;
    const parent = nodesById.get(node.parentId);
    if (!parent) continue;
    const r1 = rowIndex.get(parent.id)!;
    const r2 = rowIndex.get(node.id)!;
    const x1 = cx(parent.lane);
    const y1 = cy(r1);
    const x2 = cx(node.lane);
    const y2 = cy(r2);
    const col = gitLaneColor(node.colorIdx);

    if (node.kind === "main-actual") {
      edgesSvg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="1.6" stroke-dasharray="3 3" opacity="0.65"/>`;
    } else if (parent.lane === node.lane) {
      edgesSvg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="2.1" stroke-linecap="round"/>`;
    } else {
      const yTurn = Math.min(y2, y1 + ROW_H);
      const cp1y = y1 + ROW_H * 0.55;
      const cp2y = y1 + ROW_H * 0.22;
      edgesSvg += `<path d="M ${x1} ${y1} C ${x1} ${cp1y}, ${x2} ${cp2y}, ${x2} ${yTurn} L ${x2} ${y2}" fill="none" stroke="${col}" stroke-width="2.1" stroke-linecap="round"/>`;
    }
  }

  // Draw commit glyphs top layer
  for (const node of ordered) {
    const r = rowIndex.get(node.id)!;
    const x = cx(node.lane);
    const y = cy(r);
    const col = gitLaneColor(node.colorIdx);
    const isActive = node.id === activeNodeId;

    if (isActive) {
      nodesSvg += `<circle cx="${x}" cy="${y}" r="8" fill="${col}" opacity="0.22"/>`;
    }
    if (node.kind === "root") {
      nodesSvg += `<circle cx="${x}" cy="${y}" r="4.8" fill="var(--surface)" stroke="${col}" stroke-width="2.2"/>`;
    } else if (node.kind === "main-actual") {
      nodesSvg += `<circle cx="${x}" cy="${y}" r="3.8" fill="var(--surface)" stroke="${col}" stroke-width="1.6" stroke-dasharray="2 2"/>`;
    } else if (node.kind === "pending") {
      nodesSvg += `<circle cx="${x}" cy="${y}" r="4.4" fill="var(--surface)" stroke="${col}" stroke-width="2" stroke-dasharray="2.5 1.8"/>`;
    } else if (node.kind === "round_end") {
      nodesSvg += `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" rx="1.5" transform="rotate(45 ${x} ${y})" fill="${col}" stroke="var(--surface)" stroke-width="1.4"/>`;
    } else {
      nodesSvg += `<circle cx="${x}" cy="${y}" r="${isActive ? 5 : 4.2}" fill="${col}" stroke="var(--surface)" stroke-width="1.5"/>`;
    }
  }

  const rowsHtml = ordered
    .map((node) => {
      const col = gitLaneColor(node.colorIdx);
      const isActive = node.id === activeNodeId;
      const dataAttr =
        node.kind === "root"
          ? `data-git-root="1"`
          : node.kind === "main-actual"
          ? `data-git-main-actual="1"`
          : `data-git-branch="${escHtml(node.branchId)}" data-git-ply="${node.startPly}"`;

      const rowCls = [
        "git-row",
        isActive ? "is-active" : "",
        node.kind === "main-actual" ? "is-ghost" : "",
        node.kind === "pending" ? "is-pending" : "",
      ]
        .filter(Boolean)
        .join(" ");

      const refHtml = node.refTag
        ? `<span class="git-ref" style="--ref-col:${col}">${escHtml(node.refTag)}</span>`
        : "";
      const chipHtml = node.oppChip
        ? `<span class="git-opp-chip">${escHtml(node.oppChip)}</span>`
        : "";

      return `<div class="${rowCls}" style="padding-left:${svgW + 4}px" ${dataAttr}>
        <div class="git-row-main">
          ${refHtml}
          <span class="git-title">${escHtml(node.title)}</span>
        </div>
        <div class="git-row-meta">
          ${chipHtml}
          <span class="git-turn">${node.turn}巡</span>
        </div>
      </div>`;
    })
    .join("");

  let subPliesHtml = "";
  if (activeNode && activeNode.subPlies.length > 1) {
    const chips = activeNode.subPlies
      .map((sp) => {
        const cur = sp.ply === state.branch.cursor ? " active" : "";
        const cls = sp.isHero ? " is-hero" : sp.isCall ? " is-call" : "";
        return `<button class="subply-chip${cur}${cls}" data-branch-step="${sp.ply}">${escHtml(
          sp.who.slice(0, 4)
        )} ${escHtml(sp.act)}</button>`;
      })
      .join("");
    subPliesHtml = `<div class="git-subplies">${chips}</div>`;
  }

  const hintHtml =
    state.branch.branches.length === 1
      ? `<div class="git-hint">点击上方历史节点并改打其他牌，即可分叉出平行推演分支。</div>`
      : "";

  return `
    <div class="git-graph-wrap">
      <svg class="git-graph-svg" width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">${edgesSvg}${nodesSvg}</svg>
      <div class="git-rows">${rowsHtml}</div>
    </div>
    ${subPliesHtml}
    ${hintHtml}
  `;
}

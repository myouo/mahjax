import { calcDiscardUkeire, calcShanten, formatShantenCn, toTile34 } from "./shanten";
import { renderTileImg, seatWindCn } from "./tiles";
import type { ActionCandidate, AppState, MeldInfo, ReviewFrame, RiverTileTuple } from "./types";

function escapeHtml(s: string): string {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function getFilteredFrames(state: AppState): ReviewFrame[] {
  const frames = state.report?.frames ?? [];
  return frames.filter((f) => {
    if (f.seat !== state.seat) return false;
    if (state.filter === "diff") return !f.matched_top1;
    if (state.filter === "blunder") return f.is_blunder;
    if (state.filter === "calls") {
      return !f.is_discard || f.best_action >= 72 || f.human_action >= 72 || hasSpecialCandidates(f);
    }
    return true;
  });
}

function isSpecialAction(a: number): boolean {
  // Self-kans (37..70) or Riichi/Tsumo/Ron/Pon/Chi/Pass/Kyuushu (72..85)
  return (a >= 37 && a <= 70) || (a >= 72 && a <= 85);
}

function hasSpecialCandidates(f: ReviewFrame): boolean {
  return (f.candidates ?? []).some((c) => isSpecialAction(c.a));
}

function buildVisibleCounts34(f: ReviewFrame): Int8Array {
  const vis = new Int8Array(34);
  const add = (tid: number) => {
    const t = toTile34(tid);
    if (t >= 0 && t < 34) vis[t]++;
  };
  for (const t of f.hands[f.seat] ?? []) add(t);
  for (const d of f.dora ?? []) add(d);
  for (let p = 0; p < 4; p++) {
    for (const r of f.rivers[p] ?? []) add(r[0]);
    for (const m of f.melds[p] ?? []) {
      for (const t of m.tiles) add(t);
    }
  }
  return vis;
}

function renderPond(riverList: RiverTileTuple[], posCls: string): string {
  const tiles = (riverList ?? [])
    .map(([tid, isRiichi, isCalled, isTsumo]) => {
      const cls = [
        isRiichi ? "riichi-rot" : "",
        isCalled ? "called-dim" : "",
        isTsumo ? "tsumo-dim" : "",
      ]
        .filter(Boolean)
        .join(" ");
      return renderTileImg(tid, "xs", cls);
    })
    .join("");
  return `<div class="pond ${posCls}">${tiles}</div>`;
}

/** Render open/closed melds with authentic closed-kan backs and sideways called tile. */
function renderMeldsInline(meldsList: MeldInfo[], size: "xs" | "sm" = "xs"): string {
  if (!meldsList || meldsList.length === 0) return "";
  const groups = meldsList
    .map((m) => {
      const tiles = m.tiles ?? [];
      if (m.kind === "kan_closed" && tiles.length === 4) {
        // Classic Tenhou/Majsoul closed kan: outer two face-down, inner two face-up
        return `<div class="meld-grp" title="暗杠">
          ${renderTileImg(-1, size)}
          ${renderTileImg(tiles[1], size)}
          ${renderTileImg(tiles[2], size)}
          ${renderTileImg(-1, size)}
        </div>`;
      }
      // For chi, leftmost tile (idx 0) is called; for pon/open_kan, last tile is called
      const sideIdx = m.kind === "chi" ? 0 : tiles.length - 1;
      const inner = tiles
        .map((t, idx) => renderTileImg(t, size, idx === sideIdx ? "meld-side" : ""))
        .join("");
      return `<div class="meld-grp" title="${escapeHtml(m.kind)}">${inner}</div>`;
    })
    .join("");
  return `<div class="melds-inline">${groups}</div>`;
}

function renderOppZone(
  state: AppState,
  f: ReviewFrame,
  absSeat: number,
  relIdx: number,
  posCls: string
): string {
  const sInfo = state.report?.seats[String(absSeat)];
  const name = escapeHtml(sInfo?.name ?? `P${absSeat}`);
  const revealed = state.revealSeats[absSeat];
  const tp = f.p_tenpai?.[relIdx] ?? 0;
  const hand = f.hands[absSeat] ?? [];
  const melds = f.melds[absSeat] ?? [];
  const showBelief = !state.quizMode || state.quizAnswers[f.idx] !== undefined;

  const tilesHtml = hand
    .map((t) => renderTileImg(revealed ? t : -1, "xs"))
    .join("");
  const meldsHtml = renderMeldsInline(melds, "xs");

  return `<div class="opp-zone ${posCls}">
    <div class="opp-name-pill" data-toggle-seat="${absSeat}" title="点击单独展开/隐藏 P${absSeat} 手牌">
      <span>P${absSeat} ${name}</span>
      <span style="color:#94a3b8">${revealed ? "👁" : "🔒"}</span>
      ${showBelief && tp >= 0.35 ? `<span class="tenpai-chip">听 ${Math.round(tp * 100)}%</span>` : ""}
    </div>
    <div class="opp-row">
      <div class="opp-hand-strip" data-toggle-seat="${absSeat}" title="点击单独展开/隐藏 P${absSeat} 手牌">${tilesHtml}</div>
      ${meldsHtml}
    </div>
  </div>`;
}

function renderTenhouCenter(f: ReviewFrame): string {
  const seatBottom = f.seat;
  const seatRight = (f.seat + 1) % 4;
  const seatTop = (f.seat + 2) % 4;
  const seatLeft = (f.seat + 3) % 4;

  const edgeHtml = (absSeat: number, cls: string) => {
    const w = seatWindCn(absSeat, f.dealer);
    const isDealer = absSeat === f.dealer;
    const isTurn = absSeat === f.seat;
    const stick = f.riichi[absSeat] ? `<div class="riichi-stick ${cls}"></div>` : "";
    return `<div class="score-edge ${cls}${isTurn ? " is-turn" : ""}">
      <span class="w-tag${isDealer ? " dealer" : ""}">${w}</span>
      <span class="pts">${f.scores[absSeat].toLocaleString()}</span>
    </div>${stick}`;
  };

  const doraSlots: number[] = [];
  for (let i = 0; i < 5; i++) {
    doraSlots.push(i < f.dora.length ? f.dora[i] : -1);
  }
  const doraHtml = doraSlots.map((t) => renderTileImg(t, "xs")).join("");

  return `<div class="tenhou-center">
    ${edgeHtml(seatTop, "top")}
    ${edgeHtml(seatRight, "right")}
    ${edgeHtml(seatBottom, "bottom")}
    ${edgeHtml(seatLeft, "left")}
    <div class="center-core">
      <div class="center-round">${escapeHtml(f.round_label)}</div>
      <div class="center-wall">
        <span>余 <b>${f.wall_rem}</b></span>
        <span>供 <b>${f.kyotaku}</b></span>
        <span>第 <b>${f.turn}</b> 巡</span>
      </div>
      <div class="dora-bar" title="宝牌指示牌">${doraHtml}</div>
    </div>
  </div>`;
}

function callButtonKindClass(a: number): string {
  if (a === 72) return "kind-riichi";
  if (a === 73 || a === 74) return "kind-win";
  if (a === 84) return "kind-pass";
  return "kind-call";
}

/** Authentic floating on-table Call / Riichi / Kan / Win / PASS dock WITH weights! */
function renderTableActionDock(state: AppState, f: ReviewFrame): string {
  const specialCands: ActionCandidate[] = (f.candidates ?? []).filter((c) => isSpecialAction(c.a));
  if (specialCands.length === 0) return "";

  const userAns = state.quizMode ? state.quizAnswers[f.idx] : undefined;
  const hideAnswer = state.quizMode && userAns === undefined;

  const targetHtml =
    f.target >= 0
      ? `<div class="target-tile-box"><span>鸣牌对象</span>${renderTileImg(f.target, "sm")}</div>`
      : "";

  const btnsHtml = specialCands
    .map((c) => {
      const kindCls = callButtonKindClass(c.a);
      const isBest = c.a === f.best_action;
      const isHuman = c.a === f.human_action;
      const isUser = userAns !== undefined && userAns === c.a;

      let hlCls = "";
      let barCol = "#475569";
      let tagHtml = "";

      if (!hideAnswer) {
        if (isUser && isBest) {
          hlCls = "hl-both";
          barCol = "#34d399";
          tagHtml = `<span class="mini-tag ok">你✓AI</span>`;
        } else if (isUser) {
          hlCls = "hl-user";
          barCol = "#c084fc";
          tagHtml = `<span class="mini-tag usr">你选</span>`;
        } else if (isBest && isHuman) {
          hlCls = "hl-both";
          barCol = "#34d399";
          tagHtml = `<span class="mini-tag ok">一致✓</span>`;
        } else if (isBest) {
          hlCls = "hl-ai";
          barCol = "#38bdf8";
          tagHtml = `<span class="mini-tag ai">AI</span>`;
        } else if (isHuman) {
          hlCls = f.is_blunder ? "hl-blunder" : "hl-human";
          barCol = f.is_blunder ? "#f87171" : "#fbbf24";
          tagHtml = `<span class="mini-tag ${f.is_blunder ? "bad" : "hu"}">${f.is_blunder ? "恶手" : "实战"}</span>`;
        }
      }

      const clickAttr = hideAnswer ? `data-quiz-action="${c.a}"` : "";
      const clickCls = hideAnswer ? " clickable" : "";
      const metaHtml = hideAnswer
        ? `<div class="call-btn-meta" style="color:#94a3b8">点击作答</div>`
        : `<div class="call-btn-meta">${tagHtml}<span>${(c.prob * 100).toFixed(1)}%</span></div>`;
      const barHtml = hideAnswer
        ? ""
        : `<div class="call-btn-bar" style="width:${Math.max(3, c.prob * 100)}%;background:${barCol}"></div>`;

      return `<button class="call-btn ${kindCls} ${hlCls}${clickCls}" ${clickAttr}>
        <span>${escapeHtml(c.label)}</span>
        ${metaHtml}
        ${barHtml}
      </button>`;
    })
    .join("");

  return `<div class="table-action-dock">
    ${targetHtml}
    <div class="action-btn-group">${btnsHtml}</div>
  </div>`;
}

function renderEgoDock(state: AppState, f: ReviewFrame): string {
  const absSeat = f.seat;
  const userAns = state.quizMode ? state.quizAnswers[f.idx] : undefined;
  const hideAnswer = state.quizMode && userAns === undefined;

  const probMap: Record<number, number> = {};
  for (const c of f.candidates ?? []) {
    probMap[c.a] = c.prob;
  }

  const rawHand = [...(f.hands[absSeat] ?? [])];
  let drawnTile = -1;
  if (f.last_draw >= 0) {
    const pos = rawHand.indexOf(f.last_draw);
    if (pos >= 0) {
      rawHand.splice(pos, 1);
      drawnTile = f.last_draw;
    }
  }

  interface HandSlot {
    tile: number;
    actionId: number;
    isDraw: boolean;
  }
  const slots: HandSlot[] = rawHand.map((t) => ({
    tile: t,
    actionId: t,
    isDraw: false,
  }));
  if (drawnTile >= 0) {
    const drawAct = probMap[71] !== undefined ? 71 : drawnTile;
    slots.push({ tile: drawnTile, actionId: drawAct, isDraw: true });
  }

  const maxP = Math.max(f.best_prob || 0.01, 0.01);

  const slotsHtml = slots
    .map((sl) => {
      let p = probMap[sl.actionId];
      if (p === undefined && sl.isDraw) p = probMap[sl.tile];
      if (p === undefined) p = 0;

      const isBest =
        f.best_action === sl.actionId || (sl.isDraw && f.best_action === 71);
      const isHuman =
        f.human_action === sl.actionId || (sl.isDraw && f.human_action === 71);
      const isUser =
        userAns !== undefined &&
        (userAns === sl.actionId || (sl.isDraw && userAns === 71));

      let barCls = "other";
      let tileHl = "";
      let tagCls = "";
      let tagTxt = "";

      if (!hideAnswer) {
        if (isUser && isBest) {
          barCls = "both";
          tileHl = "hl-both";
          tagCls = "both";
          tagTxt = "你✓AI";
        } else if (isUser) {
          barCls = "user";
          tileHl = "hl-user";
          tagCls = "user";
          tagTxt = "你选";
        } else if (isBest && isHuman) {
          barCls = "both";
          tileHl = "hl-both";
          tagCls = "both";
          tagTxt = "一致✓";
        } else if (isBest) {
          barCls = "ai";
          tileHl = "hl-ai";
          tagCls = "ai";
          tagTxt = "AI";
        } else if (isHuman) {
          barCls = f.is_blunder ? "blunder" : "human";
          tileHl = f.is_blunder ? "hl-blunder" : "hl-human";
          tagCls = f.is_blunder ? "blunder" : "human";
          tagTxt = f.is_blunder ? "恶手" : "实战";
        }
      }

      const hPx =
        !hideAnswer && p > 0.004 ? Math.max(3, Math.round((p / maxP) * 36)) : 0;
      const pctTxt = !hideAnswer
        ? p >= 0.01
          ? `${Math.round(p * 100)}%`
          : p > 0.001
            ? "<1%"
            : ""
        : "";
      const pctColor =
        barCls === "both"
          ? "#34d399"
          : barCls === "ai"
            ? "#38bdf8"
            : barCls === "user"
              ? "#c084fc"
              : barCls === "blunder"
                ? "#f87171"
                : barCls === "human"
                  ? "#fbbf24"
                  : "#94a3b8";

      const canClickTile = hideAnswer && f.is_discard;
      const quizAttr = canClickTile ? `data-quiz-action="${sl.actionId}"` : "";
      const quizCls = canClickTile ? " quiz-clickable" : "";

      return `<div class="hand-slot${sl.isDraw ? " draw-gap" : ""}${quizCls}" ${quizAttr}>
        <div class="naga-bar-box">
          <div class="naga-pct" style="color:${pctColor}">${pctTxt}</div>
          ${hPx > 0 ? `<div class="naga-bar ${barCls}" style="height:${hPx}px"></div>` : ""}
        </div>
        ${renderTileImg(sl.tile, "lg", tileHl)}
        <div class="slot-tag ${tagCls}">${tagTxt}</div>
      </div>`;
    })
    .join("");

  const meldsHtml = renderMeldsInline(f.melds[absSeat] ?? [], "sm");
  const quizHint =
    hideAnswer && !hasSpecialCandidates(f)
      ? `<div class="quiz-hint-pill">🎯 何切挑战：直接点击下方手牌作答（每题仅计分一次）</div>`
      : "";

  return `<div class="ego-dock">
    ${quizHint}
    <div class="ego-strip">
      <div class="ego-hand-row">${slotsHtml}</div>
      ${meldsHtml}
    </div>
  </div>`;
}

function computeQuizStats(state: AppState): { answered: number; matched: number; avgRating: number } {
  const frames = state.report?.frames ?? [];
  const byIdx = new Map<number, ReviewFrame>();
  for (const fr of frames) byIdx.set(fr.idx, fr);

  let answered = 0;
  let matched = 0;
  let ratingSum = 0;

  for (const [idxStr, act] of Object.entries(state.quizAnswers)) {
    const fr = byIdx.get(parseInt(idxStr, 10));
    if (!fr) continue;
    answered++;
    if (act === fr.best_action) matched++;
    const cand = (fr.candidates ?? []).find((c) => c.a === act);
    const p = cand ? cand.prob : 0;
    ratingSum += Math.min(1, p / Math.max(fr.best_prob, 1e-6)) * 100;
  }

  return {
    answered,
    matched,
    avgRating: answered > 0 ? ratingSum / answered : 0,
  };
}

export function renderApp(root: HTMLElement, state: AppState): void {
  const report = state.report;
  if (!report) {
    root.innerHTML = `<div style="padding:56px;text-align:center;color:#94a3b8;">
      <h2 style="color:#38bdf8;">Sovereign-Mahjong V6 — KillerDucky / 天凤仿真复盘面板</h2>
      <p>正在加载示例牌谱，或直接拖入任意 <code>review_*.json</code> 文件到此窗口...</p>
      <label class="btn" style="display:inline-block;margin-top:12px;">
        📂 打开评测 JSON
        <input id="fileInputEmpty" type="file" accept=".json" style="display:none"/>
      </label>
    </div>`;
    return;
  }

  const frames = getFilteredFrames(state);
  if (state.cursor >= frames.length) state.cursor = Math.max(0, frames.length - 1);
  const f = frames[state.cursor];
  const curSeatInfo = report.seats[String(state.seat)];
  const userAns = f && state.quizMode ? state.quizAnswers[f.idx] : undefined;
  const hideAnswer = state.quizMode && userAns === undefined;

  // Seat tabs
  const seatTabsHtml = Object.keys(report.seats)
    .map((k) => {
      const s = report.seats[k];
      const active = parseInt(k, 10) === state.seat ? " active" : "";
      return `<button class="seat-btn${active}" data-seat="${k}">
        <span>P${k} ${escapeHtml(s.name)}</span>
        <span class="rt">${s.rating.toFixed(1)}</span>
        <span class="acc">${Math.round(s.overall_acc * 100)}%</span>
      </button>`;
    })
    .join("");

  // Round quick-jump strip
  const roundMap = new Map<number, { label: string; firstIdx: number; hasBlunder: boolean }>();
  frames.forEach((fr, idx) => {
    const existing = roundMap.get(fr.round_idx);
    if (!existing) {
      roundMap.set(fr.round_idx, {
        label: fr.round_label.replace(" 0本场", ""),
        firstIdx: idx,
        hasBlunder: fr.is_blunder,
      });
    } else if (fr.is_blunder) {
      existing.hasBlunder = true;
    }
  });
  const roundStripHtml = Array.from(roundMap.entries())
    .map(([rIdx, info]) => {
      const act = f && f.round_idx === rIdx ? " active" : "";
      return `<button class="round-pill${act}" data-step="${info.firstIdx}">
        <span>${escapeHtml(info.label)}</span>
        ${info.hasBlunder ? `<span class="b-dot" title="本局有恶手"></span>` : ""}
      </button>`;
    })
    .join("");

  const allOppRevealed = [0, 1, 2, 3]
    .filter((p) => p !== state.seat)
    .every((p) => state.revealSeats[p]);

  // Shanten & visible counts
  let shantenText = "";
  let visCounts: Int8Array | null = null;
  if (f) {
    const myMelds = (f.melds[f.seat] ?? []).length;
    shantenText = formatShantenCn(calcShanten(f.hands[f.seat] ?? [], myMelds));
    visCounts = buildVisibleCounts34(f);
  }

  // Tenhou Square Board
  const boardHtml = f
    ? `<div class="tenhou-board">
        ${renderOppZone(state, f, (f.seat + 2) % 4, 1, "opp-top")}
        ${renderOppZone(state, f, (f.seat + 3) % 4, 2, "opp-left")}
        ${renderOppZone(state, f, (f.seat + 1) % 4, 0, "opp-right")}
        ${renderTenhouCenter(f)}
        ${renderPond(f.rivers[(f.seat + 2) % 4], "pond-top")}
        ${renderPond(f.rivers[(f.seat + 3) % 4], "pond-left")}
        ${renderPond(f.rivers[(f.seat + 1) % 4], "pond-right")}
        ${renderPond(f.rivers[f.seat], "pond-bottom")}
        ${renderTableActionDock(state, f)}
        ${renderEgoDock(state, f)}
      </div>`
    : `<div class="tenhou-board" style="display:flex;align-items:center;justify-content:center;color:#94a3b8;">当前筛选条件下无匹配决策步</div>`;

  // Quiz card
  const qs = computeQuizStats(state);
  const quizAcc = qs.answered > 0 ? `${((qs.matched / qs.answered) * 100).toFixed(1)}%` : "-";
  const quizRt = qs.answered > 0 ? qs.avgRating.toFixed(1) : "-";
  const quizCardHtml = state.quizMode
    ? `<div class="card quiz-card">
        <h3><span>🎯 何切模式战绩（不可重复计分）</span><button class="btn" id="resetQuizBtn" style="padding:2px 8px;font-size:11px;">重置战绩</button></h3>
        <div class="stats-grid">
          <div class="stat-box"><div class="val purple">${qs.answered}</div><div class="lbl">已答题数</div></div>
          <div class="stat-box"><div class="val cyan">${quizAcc}</div><div class="lbl">AI一致率</div></div>
          <div class="stat-box"><div class="val">${quizRt}</div><div class="lbl">你的Rating</div></div>
        </div>
      </div>`
    : "";

  // Player stats
  const statsHtml = curSeatInfo
    ? `<div class="stat-box"><div class="val cyan">${curSeatInfo.rating.toFixed(2)}</div><div class="lbl">Rating</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.overall_acc * 100).toFixed(1)}%</div><div class="lbl">总一致率</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.discard_acc * 100).toFixed(1)}%</div><div class="lbl">弃牌一致</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.call_acc * 100).toFixed(1)}%</div><div class="lbl">鸣牌一致</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.top3_acc * 100).toFixed(1)}%</div><div class="lbl">Top-3</div></div>
       <div class="stat-box"><div class="val red">${curSeatInfo.blunder_count} (${(curSeatInfo.blunder_rate * 100).toFixed(1)}%)</div><div class="lbl">恶手</div></div>`
    : "";

  // Weight Table & Belief
  let weightTableHtml = "";
  let verdictHtml = "";
  let tenpaiHtml = "";
  if (f) {
    if (hideAnswer) {
      verdictHtml = `<span class="mini-tag usr">等待作答</span>`;
      weightTableHtml = `<div style="padding:20px 10px;text-align:center;color:#c084fc;font-size:13px;line-height:1.7;">
        🔒 答案已隐藏 — 请直接在左侧雀桌点击手牌或鸣牌/立直按钮作答<br/>
        <span style="color:#94a3b8;font-size:12px;">当前手牌：<b>${shantenText}</b></span>
      </div>`;
      tenpaiHtml = `<div style="font-size:12px;color:#94a3b8;padding:6px 0;">作答后揭晓 V6 Belief 对手听牌概率</div>`;
    } else {
      if (userAns !== undefined) {
        const uMatch = userAns === f.best_action;
        verdictHtml = uMatch
          ? `<span class="mini-tag ok">已答 · 与AI一致 ✓</span>`
          : `<span class="mini-tag usr">已锁定作答</span>`;
      } else if (f.matched_top1) {
        verdictHtml = `<span class="mini-tag ok">一致 ✓</span>`;
      } else if (f.is_blunder) {
        verdictHtml = `<span class="mini-tag bad">恶手 (${(f.prob_ratio * 100).toFixed(0)}%)</span>`;
      } else {
        verdictHtml = `<span class="mini-tag hu">分歧 (${(f.prob_ratio * 100).toFixed(0)}%)</span>`;
      }

      const myMelds = (f.melds[f.seat] ?? []).length;
      const myHand = f.hands[f.seat] ?? [];
      const rows = (f.candidates ?? [])
        .slice(0, 7)
        .map((c) => {
          const isAi = c.a === f.best_action;
          const isHu = c.a === f.human_action;
          const isUsr = userAns !== undefined && userAns === c.a;
          let color = "#475569";
          let tags = "";
          if (isUsr) tags += `<span class="mini-tag usr">你</span>`;
          if (isAi && isHu) {
            color = "#10b981";
            tags += `<span class="mini-tag ok">一致</span>`;
          } else {
            if (isAi) {
              color = "#38bdf8";
              tags += `<span class="mini-tag ai">AI</span>`;
            }
            if (isHu) {
              color = f.is_blunder ? "#ef4444" : "#f59e0b";
              tags += `<span class="mini-tag ${f.is_blunder ? "bad" : "hu"}">实战</span>`;
            }
          }
          if (isUsr && !isAi && !isHu) color = "#a855f7";

          let ukeStr = "-";
          if (visCounts && (c.a <= 36 || c.a === 71)) {
            const dTile = c.a === 71 ? f.last_draw : c.a;
            if (dTile >= 0) {
              const u = calcDiscardUkeire(myHand, dTile, myMelds, visCounts);
              ukeStr = `${formatShantenCn(u.shantenAfter)} ${u.ukeireCount}枚`;
            }
          }
          const relRatio = Math.min(100, (c.prob / Math.max(f.best_prob, 1e-6)) * 100);
          const riskCol = c.du_ron < -0.02 ? "#f87171" : c.du_ron < -0.005 ? "#fbbf24" : "#94a3b8";

          return `<tr>
            <td style="white-space:nowrap">${tags}${escapeHtml(c.label)}</td>
            <td class="bar-cell"><div class="cand-bar-bg"><div class="cand-bar-fill" style="width:${Math.max(2, c.prob * 100)}%;background:${color}"></div></div></td>
            <td style="text-align:right;font-weight:700">${(c.prob * 100).toFixed(1)}%</td>
            <td style="text-align:right;color:#94a3b8">${relRatio.toFixed(0)}</td>
            <td style="text-align:right;color:${riskCol}">${c.du_ron.toFixed(3)}</td>
            <td style="text-align:right;color:#cbd5e1;white-space:nowrap">${ukeStr}</td>
          </tr>`;
        })
        .join("");

      weightTableHtml = `<table class="weight-table">
        <thead><tr><th>动作</th><th>权重条</th><th style="text-align:right">P%</th><th style="text-align:right">比值</th><th style="text-align:right">ΔU放铳</th><th style="text-align:right">向听/进张</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

      const relNames = ["下家", "对家", "上家"];
      tenpaiHtml = [0, 1, 2]
        .map((i) => {
          const p = f.p_tenpai?.[i] ?? 0;
          const oppSeat = (f.seat + i + 1) % 4;
          return `<div class="tenpai-row">
            <span>${relNames[i]} P${oppSeat}</span>
            <div class="tenpai-bg"><div class="tenpai-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
            <span style="text-align:right;font-variant-numeric:tabular-nums">${Math.round(p * 100)}%</span>
          </div>`;
        })
        .join("");
    }
  }

  // Key move quick-jump list
  const seatAllFrames = report.frames.filter((x) => x.seat === state.seat);
  const rankedBlunders = seatAllFrames
    .filter((x) => !x.matched_top1)
    .sort((a, b) => b.best_prob - b.human_prob - (a.best_prob - a.human_prob))
    .slice(0, 18);

  const blunderHtml = rankedBlunders
    .map((x) => {
      const isCur = f && x.idx === f.idx ? " active" : "";
      const badCls = x.is_blunder ? " is-bad" : "";
      const masked = state.quizMode && state.quizAnswers[x.idx] === undefined;
      return `<div class="blunder-item${badCls}${isCur}" data-jump-idx="${x.idx}">
        <span>${escapeHtml(x.round_label.replace(" 0本场", ""))} 第${x.turn}巡</span>
        <span>${masked ? "🔒 未作答" : `${escapeHtml(x.human_action_str)} → <b style="color:#38bdf8">${escapeHtml(x.best_action_str)}</b>`}</span>
        <span style="color:${x.is_blunder ? "#f87171" : "#fbbf24"}">${masked ? "?" : `${Math.round(x.human_prob * 100)}%`}</span>
      </div>`;
    })
    .join("");

  const timelineHtml = frames
    .map((fr, i) => {
      const cls = fr.matched_top1 ? "ok" : fr.is_blunder ? "bad" : "diff";
      const cur = i === state.cursor ? " cur" : "";
      return `<div class="dot ${cls}${cur}" data-step="${i}" title="${escapeHtml(fr.round_label)} 第${fr.turn}巡"></div>`;
    })
    .join("");

  root.innerHTML = `<div class="app-shell">
    <div class="topbar">
      <span class="brand">Sovereign-Mahjong V6</span>
      <div class="seat-tabs">${seatTabsHtml}</div>
      <div class="controls-right">
        <button class="btn${state.quizMode ? " quiz-on" : ""}" id="quizBtn" title="切换何切练习模式 (Q)">
          🎯 何切模式: ${state.quizMode ? "开" : "关"}
        </button>
        <button class="btn${allOppRevealed ? " active" : ""}" id="handsToggleBtn" title="全开/全关三家手牌 (H)">
          👁 手牌: ${allOppRevealed ? "全开" : "隐藏"}
        </button>
        <select id="filterSelect" title="过滤决策步">
          <option value="all"${state.filter === "all" ? " selected" : ""}>筛选: 全部决策</option>
          <option value="diff"${state.filter === "diff" ? " selected" : ""}>仅看: 分歧手</option>
          <option value="blunder"${state.filter === "blunder" ? " selected" : ""}>仅看: 恶手</option>
          <option value="calls"${state.filter === "calls" ? " selected" : ""}>仅看: 鸣牌/立直</option>
        </select>
        <label class="btn" title="载入本地 review_*.json">
          📂 打开JSON
          <input id="fileInput" type="file" accept=".json" style="display:none" />
        </label>
      </div>
    </div>

    <div class="round-strip">
      <span class="round-strip-label">局次快跳:</span>
      ${roundStripHtml}
    </div>

    <div class="main">
      ${boardHtml}

      <div class="sidebar">
        ${quizCardHtml}

        <div class="card">
          <h3><span>P${state.seat} ${escapeHtml(curSeatInfo?.name ?? "")} (${escapeHtml(report.rule)})</span><span>${verdictHtml}</span></h3>
          <div class="stats-grid">${statsHtml}</div>
        </div>

        <div class="card">
          <h3><span>AI 权重与牌效明细 (${shantenText || "-"})</span><span style="color:#38bdf8">${f ? `${escapeHtml(f.round_label)} · 第${f.turn}巡` : ""}</span></h3>
          ${weightTableHtml}
        </div>

        <div class="card">
          <h3><span>V6 Belief 对手听牌概率</span><span style="font-size:11px;color:#94a3b8">点对手手牌可单独开关</span></h3>
          ${tenpaiHtml}
        </div>

        <div class="card">
          <h3><span>关键分歧 / 恶手快跳</span><span>Top ${rankedBlunders.length}</span></h3>
          <div class="blunder-list">${blunderHtml}</div>
        </div>
      </div>
    </div>

    <div class="nav-bar">
      <div class="nav-controls">
        <button class="btn" id="prevRoundBtn">⏮ 上一局</button>
        <button class="btn" id="prevBlunderBtn">⏪ 上一恶手 (↑)</button>
        <button class="btn" id="prevDiffBtn">‹ 上一分歧 ([)</button>
        <button class="btn" id="prevStepBtn">◀ 上一手 (←)</button>
        <span class="step-counter">${frames.length ? `${state.cursor + 1} / ${frames.length}` : "0 / 0"}</span>
        <button class="btn" id="nextStepBtn">下一手 (→) ▶</button>
        <button class="btn" id="nextDiffBtn">下一分歧 (]) ›</button>
        <button class="btn" id="nextBlunderBtn">下一恶手 (↓) ⏩</button>
        <button class="btn" id="nextCallBtn">下一鸣牌/立直 (V)</button>
        <button class="btn" id="nextRoundBtn">下一局 ⏭</button>
      </div>
      <div class="timeline">${timelineHtml}</div>
      <div class="kbd-hint">快捷键：←/→(Space) 切步 · ↑/↓ 恶手 · [/] 分歧 · C/V 鸣牌立直 · H 全开关手牌 · Q 何切模式 · 1~4 切视角 · 拖入 JSON 加载新牌谱</div>
    </div>
  </div>`;
}

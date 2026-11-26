import { calcDiscardUkeire, calcShanten, formatShantenCn, toTile34 } from "./shanten";
import { renderTileImg, seatWindCn } from "./tiles";
import type { ActionCandidate, AppState, MeldInfo, ReviewFrame, RiverTileTuple } from "./types";

function esc(s: string): string {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function isSpecialAction(a: number): boolean {
  return (a >= 37 && a <= 70) || (a >= 72 && a <= 85);
}

export function getFilteredFrames(state: AppState): ReviewFrame[] {
  const frames = state.report?.frames ?? [];
  return frames.filter((f) => {
    if (f.seat !== state.seat) return false;
    if (state.filter === "diff") return !f.matched_top1;
    if (state.filter === "blunder") return f.is_blunder;
    if (state.filter === "calls") {
      return !f.is_discard || f.best_action >= 72 || f.human_action >= 72 || (f.candidates ?? []).some((c) => isSpecialAction(c.a));
    }
    return true;
  });
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

function renderRiverTiles(riverList: RiverTileTuple[]): string {
  return (riverList ?? [])
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
}

function renderMeldGroups(meldsList: MeldInfo[], size: "xs" | "sm"): string {
  return (meldsList ?? [])
    .map((m) => {
      const tiles = m.tiles ?? [];
      if (m.kind === "kan_closed" && tiles.length === 4) {
        return `<div class="meld-grp" title="暗杠">${renderTileImg(-1, size)}${renderTileImg(tiles[1], size)}${renderTileImg(tiles[2], size)}${renderTileImg(-1, size)}</div>`;
      }
      const sideIdx = m.kind === "chi" ? 0 : tiles.length - 1;
      const inner = tiles
        .map((t, idx) => renderTileImg(t, size, idx === sideIdx ? "meld-side" : ""))
        .join("");
      return `<div class="meld-grp">${inner}</div>`;
    })
    .join("");
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
  return { answered, matched, avgRating: answered > 0 ? ratingSum / answered : 0 };
}

/** Mount the static locked shell ONCE so step navigation NEVER rebuilds or shifts the page! */
export function mountShell(root: HTMLElement): void {
  root.innerHTML = `<div class="shell">
    <div class="topbar">
      <span class="brand">Sovereign-Mahjong V6</span>
      <div class="seat-tabs" id="slotSeatTabs"></div>
      <div class="top-right">
        <div class="quiz-pill-stats hidden" id="slotQuizStats"></div>
        <button class="btn" id="quizBtn" title="何切练习模式 (Q)">🎯 何切: 关</button>
        <button class="btn" id="handsToggleBtn" title="全开/全关三家手牌 (H)">👁 手牌: 关</button>
        <select id="filterSelect" title="过滤决策步">
          <option value="all">全部决策</option>
          <option value="diff">仅分歧手</option>
          <option value="blunder">仅恶手</option>
          <option value="calls">仅鸣牌/立直</option>
        </select>
        <label class="btn" title="打开 review_*.json">
          📂 打开JSON
          <input id="fileInput" type="file" accept=".json" style="display:none" />
        </label>
      </div>
    </div>

    <div class="round-strip" id="slotRoundStrip"></div>

    <div class="workspace">
      <div class="board-stage" id="boardStage">
        <div class="tenhou-board" id="tenhouBoard">
          <!-- Fixed Opponent Slots -->
          <div class="opp-top-hand" id="slotTopHand" data-rel-opp="2"></div>
          <div class="opp-top-melds" id="slotTopMelds"></div>
          <div class="opp-top-tag" id="slotTopTag"></div>

          <div class="opp-left-hand" id="slotLeftHand" data-rel-opp="3"></div>
          <div class="opp-left-melds" id="slotLeftMelds"></div>
          <div class="opp-left-tag" id="slotLeftTag"></div>

          <div class="opp-right-hand" id="slotRightHand" data-rel-opp="1"></div>
          <div class="opp-right-melds" id="slotRightMelds"></div>
          <div class="opp-right-tag" id="slotRightTag"></div>

          <!-- Fixed Center Compass -->
          <div class="tenhou-center" id="slotCenter"></div>

          <!-- 4 Fixed Discard Ponds -->
          <div class="pond pond-top" id="slotPondTop"></div>
          <div class="pond pond-left" id="slotPondLeft"></div>
          <div class="pond pond-right" id="slotPondRight"></div>
          <div class="pond pond-bottom" id="slotPondBottom"></div>

          <!-- Fixed Tenhou Call / Riichi Action Bar -->
          <div class="tenhou-actions empty" id="slotTenhouActions"></div>

          <!-- Fixed-Anchor Ego Hand & Ego Melds -->
          <div class="ego-hand-fixed" id="slotEgoHand"></div>
          <div class="ego-melds-fixed" id="slotEgoMelds"></div>
        </div>
      </div>

      <div class="sidebar">
        <div class="card">
          <h3><span id="slotStatsTitle">玩家统计</span><span id="slotVerdict"></span></h3>
          <div class="stats-grid" id="slotStatsGrid"></div>
        </div>

        <div class="card">
          <h3><span id="slotWeightTitle">AI 权重与牌效</span><span id="slotStepMeta" style="color:#38bdf8"></span></h3>
          <div class="weight-wrap" id="slotWeightWrap"></div>
        </div>

        <div class="card">
          <h3><span>V6 Belief 对手听牌概率</span><span style="font-size:11px;color:#8a9eb8">点对手手牌可单独开关</span></h3>
          <div id="slotTenpai"></div>
        </div>

        <div class="card">
          <h3><span>关键分歧 / 恶手快跳</span><span id="slotBlunderCount"></span></h3>
          <div class="blunder-list" id="slotBlunderList"></div>
        </div>
      </div>
    </div>

    <div class="nav-bar">
      <div class="nav-controls">
        <button class="btn" id="prevRoundBtn">⏮ 上一局</button>
        <button class="btn" id="prevBlunderBtn">⏪ 上一恶手 (↑)</button>
        <button class="btn" id="prevDiffBtn">‹ 上一分歧 ([)</button>
        <button class="btn" id="prevStepBtn">◀ 上一手 (←)</button>
        <span class="step-counter" id="slotStepCounter">0 / 0</span>
        <button class="btn" id="nextStepBtn">下一手 (→) ▶</button>
        <button class="btn" id="nextDiffBtn">下一分歧 (]) ›</button>
        <button class="btn" id="nextBlunderBtn">下一恶手 (↓) ⏩</button>
        <button class="btn" id="nextCallBtn">下一鸣牌/立直 (V)</button>
        <button class="btn" id="nextRoundBtn">下一局 ⏭</button>
      </div>
      <div class="timeline" id="slotTimeline"></div>
    </div>
  </div>`;
}

/** Smoothly scale the fixed 760x760 board to fit the available stage without ever shifting internal coordinates. */
export function fitBoard(): void {
  const stage = document.getElementById("boardStage");
  const board = document.getElementById("tenhouBoard");
  if (!stage || !board) return;
  const availW = stage.clientWidth - 8;
  const availH = stage.clientHeight - 8;
  const scale = Math.min(1, availW / 760, availH / 760);
  board.style.transform = `scale(${Math.max(0.5, scale).toFixed(4)})`;
}

function tenhouBtnKind(a: number): string {
  if (a === 72) return "k-riichi";
  if (a === 73 || a === 74) return "k-win";
  if (a === 84) return "k-pass";
  return "";
}

function tenhouBtnShortLabel(c: ActionCandidate): string {
  if (c.a === 84) return "跳过";
  if (c.a === 72) return "立直";
  if (c.a === 73) return "自摸";
  if (c.a === 74) return "荣和";
  return c.label;
}

/** Patch ONLY dynamic inner slots — ZERO layout shift, ZERO scroll reset! */
export function patchView(state: AppState): void {
  const report = state.report;
  const $ = (id: string) => document.getElementById(id)!;

  if (!report) {
    $("slotCenter").innerHTML = `<div class="center-core"><div class="center-round">加载中...</div><div class="center-wall">可拖入 JSON</div></div>`;
    return;
  }

  const frames = getFilteredFrames(state);
  if (state.cursor >= frames.length) state.cursor = Math.max(0, frames.length - 1);
  const f = frames[state.cursor];
  const curSeatInfo = report.seats[String(state.seat)];
  const userAns = f && state.quizMode ? state.quizAnswers[f.idx] : undefined;
  const hideAnswer = state.quizMode && userAns === undefined;

  // 1. Topbar controls
  $("slotSeatTabs").innerHTML = Object.keys(report.seats)
    .map((k) => {
      const s = report.seats[k];
      const act = parseInt(k, 10) === state.seat ? " active" : "";
      return `<button class="seat-btn${act}" data-seat="${k}"><span>P${k} ${esc(s.name)}</span><span class="rt">${s.rating.toFixed(1)}</span></button>`;
    })
    .join("");

  const quizBtn = $("quizBtn");
  quizBtn.className = `btn${state.quizMode ? " quiz-on" : ""}`;
  quizBtn.textContent = `🎯 何切: ${state.quizMode ? "开" : "关"}`;

  const quizStatsEl = $("slotQuizStats");
  if (state.quizMode) {
    const qs = computeQuizStats(state);
    const acc = qs.answered > 0 ? `${((qs.matched / qs.answered) * 100).toFixed(0)}%` : "-";
    const rt = qs.answered > 0 ? qs.avgRating.toFixed(1) : "-";
    quizStatsEl.classList.remove("hidden");
    quizStatsEl.innerHTML = `<span>已答 <b>${qs.answered}</b></span><span>一致 <b>${acc}</b></span><span>Rating <b>${rt}</b></span><button class="btn" id="resetQuizBtn" style="height:20px;padding:0 6px;font-size:11px;">重置</button>`;
  } else {
    quizStatsEl.classList.add("hidden");
  }

  const allOppRevealed = [0, 1, 2, 3]
    .filter((p) => p !== state.seat)
    .every((p) => state.revealSeats[p]);
  const handsBtn = $("handsToggleBtn");
  handsBtn.className = `btn${allOppRevealed ? " active" : ""}`;
  handsBtn.textContent = `👁 手牌: ${allOppRevealed ? "开" : "关"}`;
  ($("filterSelect") as HTMLSelectElement).value = state.filter;

  // 2. Round Strip
  const roundMap = new Map<number, { label: string; firstIdx: number; hasBlunder: boolean }>();
  frames.forEach((fr, idx) => {
    const ex = roundMap.get(fr.round_idx);
    if (!ex) {
      roundMap.set(fr.round_idx, {
        label: fr.round_label.replace(" 0本场", ""),
        firstIdx: idx,
        hasBlunder: fr.is_blunder,
      });
    } else if (fr.is_blunder) {
      ex.hasBlunder = true;
    }
  });
  $("slotRoundStrip").innerHTML =
    `<span class="round-strip-label">局次:</span>` +
    Array.from(roundMap.entries())
      .map(([rIdx, info]) => {
        const act = f && f.round_idx === rIdx ? " active" : "";
        return `<button class="round-pill${act}" data-step="${info.firstIdx}">${esc(info.label)}${info.hasBlunder ? `<span class="b-dot"></span>` : ""}</button>`;
      })
      .join("");

  // 3. Board Slots
  if (!f) {
    $("slotCenter").innerHTML = `<div class="center-core"><div class="center-round">无匹配手</div></div>`;
    $("slotEgoHand").innerHTML = "";
    $("slotEgoMelds").innerHTML = "";
    $("slotTenhouActions").classList.add("empty");
    $("slotStepCounter").textContent = "0 / 0";
    $("slotTimeline").innerHTML = "";
    return;
  }

  const sBot = f.seat;
  const sRight = (f.seat + 1) % 4;
  const sTop = (f.seat + 2) % 4;
  const sLeft = (f.seat + 3) % 4;

  // Opponents (hand + melds + tag)
  const renderOpp = (absSeat: number, relIdx: number, handId: string, meldsId: string, tagId: string) => {
    const rev = state.revealSeats[absSeat];
    const tp = f.p_tenpai?.[relIdx] ?? 0;
    const sInfo = report.seats[String(absSeat)];
    const name = esc(sInfo?.name ?? `P${absSeat}`);
    $(handId).dataset.toggleSeat = String(absSeat);
    $(handId).innerHTML = (f.hands[absSeat] ?? []).map((t) => renderTileImg(rev ? t : -1, "xs")).join("");
    $(meldsId).innerHTML = renderMeldGroups(f.melds[absSeat] ?? [], "xs");
    $(tagId).innerHTML = `<div class="opp-pill" data-toggle-seat="${absSeat}">
      <span>P${absSeat} ${name}</span>
      <span style="color:#8a9eb8">${rev ? "👁" : "🔒"}</span>
      ${!hideAnswer && tp >= 0.35 ? `<span class="tenpai-chip">听 ${Math.round(tp * 100)}%</span>` : ""}
    </div>`;
  };
  renderOpp(sTop, 1, "slotTopHand", "slotTopMelds", "slotTopTag");
  renderOpp(sLeft, 2, "slotLeftHand", "slotLeftMelds", "slotLeftTag");
  renderOpp(sRight, 0, "slotRightHand", "slotRightMelds", "slotRightTag");

  // Center Compass
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
  const doraSlots = [0, 1, 2, 3, 4].map((i) => (i < f.dora.length ? f.dora[i] : -1));
  $("slotCenter").innerHTML = `
    ${edgeHtml(sTop, "top")}
    ${edgeHtml(sRight, "right")}
    ${edgeHtml(sBot, "bottom")}
    ${edgeHtml(sLeft, "left")}
    <div class="center-core">
      <div class="center-round">${esc(f.round_label)}</div>
      <div class="center-wall"><span>余 <b>${f.wall_rem}</b></span><span>供 <b>${f.kyotaku}</b></span><span>第 <b>${f.turn}</b> 巡</span></div>
      <div class="dora-bar" title="宝牌指示牌">${doraSlots.map((t) => renderTileImg(t, "xs")).join("")}</div>
    </div>`;

  // 4 Fixed Rivers
  $("slotPondTop").innerHTML = renderRiverTiles(f.rivers[sTop]);
  $("slotPondLeft").innerHTML = renderRiverTiles(f.rivers[sLeft]);
  $("slotPondRight").innerHTML = renderRiverTiles(f.rivers[sRight]);
  $("slotPondBottom").innerHTML = renderRiverTiles(f.rivers[sBot]);

  // Fixed Tenhou Call / Riichi / Kan / Win / PASS Action Bar
  const specialCands = (f.candidates ?? []).filter((c) => isSpecialAction(c.a));
  const actEl = $("slotTenhouActions");
  if (specialCands.length === 0) {
    actEl.classList.add("empty");
    actEl.innerHTML = "";
  } else {
    actEl.classList.remove("empty");
    const tgtChip =
      f.target >= 0
        ? `<div class="tenhou-target-chip"><span>鸣牌</span>${renderTileImg(f.target, "xs")}</div>`
        : "";
    const btns = specialCands
      .map((c) => {
        const isBest = c.a === f.best_action;
        const isHuman = c.a === f.human_action;
        const isUser = userAns !== undefined && userAns === c.a;
        let hl = "";
        let barCol = "#475569";
        let tag = "";
        if (!hideAnswer) {
          if (isUser && isBest) { hl = "hl-both"; barCol = "#34d399"; tag = "你✓AI"; }
          else if (isUser) { hl = "hl-user"; barCol = "#c084fc"; tag = "你选"; }
          else if (isBest && isHuman) { hl = "hl-both"; barCol = "#34d399"; tag = "一致✓"; }
          else if (isBest) { hl = "hl-ai"; barCol = "#38bdf8"; tag = "AI"; }
          else if (isHuman) { hl = f.is_blunder ? "hl-blunder" : "hl-human"; barCol = f.is_blunder ? "#f87171" : "#fbbf24"; tag = f.is_blunder ? "恶手" : "实战"; }
        }
        const clickAttr = hideAnswer ? `data-quiz-action="${c.a}"` : "";
        const sub = hideAnswer
          ? `<div class="sub-p">点击选择</div>`
          : `<div class="sub-p">${tag ? `<span>${tag}</span>` : ""}<span>${(c.prob * 100).toFixed(1)}%</span></div>`;
        const bar = hideAnswer
          ? ""
          : `<div class="w-bar" style="width:${Math.max(3, c.prob * 100)}%;background:${barCol}"></div>`;
        return `<button class="tenhou-btn ${tenhouBtnKind(c.a)} ${hl}${hideAnswer ? " clickable" : ""}" ${clickAttr}>
          <span>${esc(tenhouBtnShortLabel(c))}</span>${sub}${bar}
        </button>`;
      })
      .join("");
    actEl.innerHTML = `${tgtChip}${btns}`;
  }

  // Fixed-Anchor Ego Hand & Ego Melds
  const probMap: Record<number, number> = {};
  for (const c of f.candidates ?? []) probMap[c.a] = c.prob;

  const rawHand = [...(f.hands[sBot] ?? [])];
  let drawnTile = -1;
  if (f.last_draw >= 0) {
    const pos = rawHand.indexOf(f.last_draw);
    if (pos >= 0) {
      rawHand.splice(pos, 1);
      drawnTile = f.last_draw;
    }
  }
  const slots = rawHand.map((t) => ({ tile: t, actionId: t, isDraw: false }));
  if (drawnTile >= 0) {
    const drawAct = probMap[71] !== undefined ? 71 : drawnTile;
    slots.push({ tile: drawnTile, actionId: drawAct, isDraw: true });
  }
  const maxP = Math.max(f.best_prob || 0.01, 0.01);

  $("slotEgoHand").innerHTML = slots
    .map((sl) => {
      let p = probMap[sl.actionId];
      if (p === undefined && sl.isDraw) p = probMap[sl.tile];
      if (p === undefined) p = 0;

      const isBest = f.best_action === sl.actionId || (sl.isDraw && f.best_action === 71);
      const isHuman = f.human_action === sl.actionId || (sl.isDraw && f.human_action === 71);
      const isUser = userAns !== undefined && (userAns === sl.actionId || (sl.isDraw && userAns === 71));

      let barCls = "other", tileHl = "", tagCls = "", tagTxt = "";
      if (!hideAnswer) {
        if (isUser && isBest) { barCls = "both"; tileHl = "hl-both"; tagCls = "both"; tagTxt = "你✓AI"; }
        else if (isUser) { barCls = "user"; tileHl = "hl-user"; tagCls = "user"; tagTxt = "你选"; }
        else if (isBest && isHuman) { barCls = "both"; tileHl = "hl-both"; tagCls = "both"; tagTxt = "一致✓"; }
        else if (isBest) { barCls = "ai"; tileHl = "hl-ai"; tagCls = "ai"; tagTxt = "AI"; }
        else if (isHuman) {
          barCls = f.is_blunder ? "blunder" : "human";
          tileHl = f.is_blunder ? "hl-blunder" : "hl-human";
          tagCls = f.is_blunder ? "blunder" : "human";
          tagTxt = f.is_blunder ? "恶手" : "实战";
        }
      }

      const hPx = !hideAnswer && p > 0.004 ? Math.max(3, Math.round((p / maxP) * 32)) : 0;
      const pctTxt = !hideAnswer ? (p >= 0.01 ? `${Math.round(p * 100)}%` : p > 0.001 ? "<1%" : "") : "";
      const pctColor =
        barCls === "both" ? "#34d399" :
        barCls === "ai" ? "#38bdf8" :
        barCls === "user" ? "#c084fc" :
        barCls === "blunder" ? "#f87171" :
        barCls === "human" ? "#fbbf24" : "#8a9eb8";

      const canClick = hideAnswer && f.is_discard;
      return `<div class="hand-slot${sl.isDraw ? " draw-gap" : ""}${canClick ? " quiz-clickable" : ""}" ${canClick ? `data-quiz-action="${sl.actionId}"` : ""}>
        <div class="naga-bar-box">
          <div class="naga-pct" style="color:${pctColor}">${pctTxt}</div>
          ${hPx > 0 ? `<div class="naga-bar ${barCls}" style="height:${hPx}px"></div>` : ""}
        </div>
        ${renderTileImg(sl.tile, "lg", tileHl)}
        <div class="slot-tag ${tagCls}">${tagTxt}</div>
      </div>`;
    })
    .join("");

  $("slotEgoMelds").innerHTML = renderMeldGroups(f.melds[sBot] ?? [], "sm");

  // 4. Right Sidebar (Fixed Heights!)
  const myMelds = (f.melds[sBot] ?? []).length;
  const shantenText = formatShantenCn(calcShanten(f.hands[sBot] ?? [], myMelds));
  const visCounts = buildVisibleCounts34(f);

  $("slotStatsTitle").textContent = `P${state.seat} ${curSeatInfo?.name ?? ""} (${report.rule})`;
  $("slotStatsGrid").innerHTML = curSeatInfo
    ? `<div class="stat-box"><div class="val cyan">${curSeatInfo.rating.toFixed(2)}</div><div class="lbl">Rating</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.overall_acc * 100).toFixed(1)}%</div><div class="lbl">总一致率</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.discard_acc * 100).toFixed(1)}%</div><div class="lbl">弃牌一致</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.call_acc * 100).toFixed(1)}%</div><div class="lbl">鸣牌一致</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.top3_acc * 100).toFixed(1)}%</div><div class="lbl">Top-3</div></div>
       <div class="stat-box"><div class="val red">${curSeatInfo.blunder_count} (${(curSeatInfo.blunder_rate * 100).toFixed(1)}%)</div><div class="lbl">恶手</div></div>`
    : "";

  $("slotWeightTitle").textContent = `AI 权重与牌效 (${shantenText})`;
  $("slotStepMeta").textContent = `${f.round_label} · 第${f.turn}巡`;

  if (hideAnswer) {
    $("slotVerdict").innerHTML = `<span class="mini-tag usr">何切作答中</span>`;
    $("slotWeightWrap").innerHTML = `<div style="padding:36px 12px;text-align:center;color:#c084fc;font-size:13px;line-height:1.8;">
      🔒 答案已隐藏 — 请直接在左侧天凤桌点击手牌或右侧鸣牌/立直按钮作答<br/>
      <span style="color:#8a9eb8;font-size:12px;">每题锁定计分一次，作答后立即揭晓 AI 权重与牌效</span>
    </div>`;
    $("slotTenpai").innerHTML = `<div style="font-size:12px;color:#8a9eb8;padding:12px 0;">作答后显示 V6 Belief 三家听牌概率</div>`;
  } else {
    if (userAns !== undefined) {
      $("slotVerdict").innerHTML = userAns === f.best_action
        ? `<span class="mini-tag ok">已答 · 与AI一致 ✓</span>`
        : `<span class="mini-tag usr">已锁定作答</span>`;
    } else if (f.matched_top1) {
      $("slotVerdict").innerHTML = `<span class="mini-tag ok">一致 ✓</span>`;
    } else if (f.is_blunder) {
      $("slotVerdict").innerHTML = `<span class="mini-tag bad">恶手 (${(f.prob_ratio * 100).toFixed(0)}%)</span>`;
    } else {
      $("slotVerdict").innerHTML = `<span class="mini-tag hu">分歧 (${(f.prob_ratio * 100).toFixed(0)}%)</span>`;
    }

    const myHand = f.hands[sBot] ?? [];
    const rows = (f.candidates ?? [])
      .slice(0, 7)
      .map((c) => {
        const isAi = c.a === f.best_action;
        const isHu = c.a === f.human_action;
        const isUsr = userAns !== undefined && userAns === c.a;
        let color = "#475569";
        let tags = "";
        if (isUsr) tags += `<span class="mini-tag usr">你</span>`;
        if (isAi && isHu) { color = "#10b981"; tags += `<span class="mini-tag ok">一致</span>`; }
        else {
          if (isAi) { color = "#38bdf8"; tags += `<span class="mini-tag ai">AI</span>`; }
          if (isHu) { color = f.is_blunder ? "#ef4444" : "#f59e0b"; tags += `<span class="mini-tag ${f.is_blunder ? "bad" : "hu"}">实战</span>`; }
        }
        if (isUsr && !isAi && !isHu) color = "#a855f7";

        let ukeStr = "-";
        if (c.a <= 36 || c.a === 71) {
          const dTile = c.a === 71 ? f.last_draw : c.a;
          if (dTile >= 0) {
            const u = calcDiscardUkeire(myHand, dTile, myMelds, visCounts);
            ukeStr = `${formatShantenCn(u.shantenAfter)} ${u.ukeireCount}枚`;
          }
        }
        const rel = Math.min(100, (c.prob / Math.max(f.best_prob, 1e-6)) * 100);
        const rCol = c.du_ron < -0.02 ? "#f87171" : c.du_ron < -0.005 ? "#fbbf24" : "#8a9eb8";
        return `<tr>
          <td style="white-space:nowrap">${tags}${esc(c.label)}</td>
          <td class="bar-cell"><div class="cand-bar-bg"><div class="cand-bar-fill" style="width:${Math.max(2, c.prob * 100)}%;background:${color}"></div></div></td>
          <td style="text-align:right;font-weight:700">${(c.prob * 100).toFixed(1)}%</td>
          <td style="text-align:right;color:#8a9eb8">${rel.toFixed(0)}</td>
          <td style="text-align:right;color:${rCol}">${c.du_ron.toFixed(3)}</td>
          <td style="text-align:right;color:#cbd5e1;white-space:nowrap">${ukeStr}</td>
        </tr>`;
      })
      .join("");

    $("slotWeightWrap").innerHTML = `<table class="weight-table">
      <thead><tr><th>动作</th><th>权重条</th><th style="text-align:right">P%</th><th style="text-align:right">比值</th><th style="text-align:right">ΔU放铳</th><th style="text-align:right">向听/进张</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

    const relNames = ["下家", "对家", "上家"];
    $("slotTenpai").innerHTML = [0, 1, 2]
      .map((i) => {
        const p = f.p_tenpai?.[i] ?? 0;
        const oppSeat = (sBot + i + 1) % 4;
        return `<div class="tenpai-row">
          <span>${relNames[i]} P${oppSeat}</span>
          <div class="tenpai-bg"><div class="tenpai-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
          <span style="text-align:right;font-variant-numeric:tabular-nums">${Math.round(p * 100)}%</span>
        </div>`;
      })
      .join("");
  }

  // Key Moves List
  const ranked = report.frames
    .filter((x) => x.seat === state.seat && !x.matched_top1)
    .sort((a, b) => b.best_prob - b.human_prob - (a.best_prob - a.human_prob))
    .slice(0, 20);
  $("slotBlunderCount").textContent = `Top ${ranked.length}`;
  $("slotBlunderList").innerHTML = ranked
    .map((x) => {
      const isCur = x.idx === f.idx ? " active" : "";
      const bad = x.is_blunder ? " is-bad" : "";
      const masked = state.quizMode && state.quizAnswers[x.idx] === undefined;
      return `<div class="blunder-item${bad}${isCur}" data-jump-idx="${x.idx}">
        <span>${esc(x.round_label.replace(" 0本场", ""))} 第${x.turn}巡</span>
        <span>${masked ? "🔒 未作答" : `${esc(x.human_action_str)} → <b style="color:#38bdf8">${esc(x.best_action_str)}</b>`}</span>
        <span style="color:${x.is_blunder ? "#f87171" : "#fbbf24"}">${masked ? "?" : `${Math.round(x.human_prob * 100)}%`}</span>
      </div>`;
    })
    .join("");

  // 5. Bottom Nav & Timeline
  $("slotStepCounter").textContent = `${state.cursor + 1} / ${frames.length}`;
  $("slotTimeline").innerHTML = frames
    .map((fr, i) => {
      const cls = fr.matched_top1 ? "ok" : fr.is_blunder ? "bad" : "diff";
      return `<div class="dot ${cls}${i === state.cursor ? " cur" : ""}" data-step="${i}" title="${esc(fr.round_label)} 第${fr.turn}巡"></div>`;
    })
    .join("");
}

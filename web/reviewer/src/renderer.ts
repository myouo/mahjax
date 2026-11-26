import { renderTileImg, seatWindCn } from "./tiles";
import type { AppState, MeldInfo, ReviewFrame, RiverTileTuple } from "./types";

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
    return true;
  });
}

function renderRiver(riverList: RiverTileTuple[]): string {
  if (!riverList || riverList.length === 0) return `<div class="river"></div>`;
  const tiles = riverList
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
  return `<div class="river">${tiles}</div>`;
}

function renderMelds(meldsList: MeldInfo[]): string {
  if (!meldsList || meldsList.length === 0) return "";
  return `<div class="melds-row">${meldsList
    .map(
      (m) =>
        `<div class="meld-grp">${m.tiles
          .map((t) => renderTileImg(t, "xs"))
          .join("")}</div>`
    )
    .join("")}</div>`;
}

function renderOppSeat(
  state: AppState,
  f: ReviewFrame,
  absSeat: number,
  relIdx: number
): string {
  const sInfo = state.report?.seats[String(absSeat)];
  const name = escapeHtml(sInfo?.name ?? `Seat ${absSeat}`);
  const wind = seatWindCn(absSeat, f.dealer);
  const sc = f.scores[absSeat];
  const isRiichi = f.riichi[absSeat];
  const tp = f.p_tenpai?.[relIdx] ?? 0;
  const hand = f.hands[absSeat] ?? [];

  const handHtml = `<div class="opp-hand">${hand
    .map((t) => renderTileImg(state.xray ? t : -1, "xs"))
    .join("")}</div>`;

  const tagHtml = `<div class="player-tag">
    <span class="wind">${wind}</span>
    <span>P${absSeat} ${name}</span>
    <span class="sc">${sc.toLocaleString()}</span>
    ${isRiichi ? `<span class="riichi-pill">立直</span>` : ""}
    ${tp >= 0.35 ? `<span class="tenpai-pill">听牌 ${Math.round(tp * 100)}%</span>` : ""}
  </div>`;

  return `${tagHtml}${handHtml}${renderMelds(f.melds[absSeat])}${renderRiver(f.rivers[absSeat])}`;
}

function renderEgoSeat(state: AppState, f: ReviewFrame): string {
  const absSeat = f.seat;
  const sInfo = state.report?.seats[String(absSeat)];
  const name = escapeHtml(sInfo?.name ?? `Seat ${absSeat}`);
  const wind = seatWindCn(absSeat, f.dealer);
  const sc = f.scores[absSeat];
  const isRiichi = f.riichi[absSeat];

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

      let barCls = "other";
      let tileHl = "";
      let tagCls = "";
      let tagTxt = "";
      if (isBest && isHuman) {
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

      const hPx = p > 0.004 ? Math.max(3, Math.round((p / maxP) * 44)) : 0;
      const pctTxt =
        p >= 0.01 ? `${Math.round(p * 100)}%` : p > 0.001 ? "<1%" : "";
      const pctColor =
        barCls === "both"
          ? "#34d399"
          : barCls === "ai"
            ? "#38bdf8"
            : barCls === "blunder"
              ? "#f87171"
              : barCls === "human"
                ? "#fbbf24"
                : "#94a3b8";

      return `<div class="hand-slot ${sl.isDraw ? "draw-gap" : ""}">
        <div class="naga-bar-box">
          <div class="naga-pct" style="color:${pctColor}">${pctTxt}</div>
          ${hPx > 0 ? `<div class="naga-bar ${barCls}" style="height:${hPx}px"></div>` : ""}
        </div>
        ${renderTileImg(sl.tile, "lg", tileHl)}
        <div class="slot-tag ${tagCls}">${tagTxt}</div>
      </div>`;
    })
    .join("");

  let callBanner = "";
  if (!f.is_discard || f.best_action >= 72 || f.human_action >= 72) {
    const tgtHtml =
      f.target >= 0
        ? `<span>对象牌:</span>${renderTileImg(f.target, "sm")}`
        : "";
    const huCol = f.matched_top1
      ? "#34d399"
      : f.is_blunder
        ? "#f87171"
        : "#fbbf24";
    callBanner = `<div class="call-banner">
      ${tgtHtml}
      <span>实战: <b style="color:${huCol}">${escapeHtml(f.human_action_str)} (${(f.human_prob * 100).toFixed(1)}%)</b></span>
      <span>|</span>
      <span>AI推荐: <b style="color:#38bdf8">${escapeHtml(f.best_action_str)} (${(f.best_prob * 100).toFixed(1)}%)</b></span>
    </div>`;
  }

  const tagHtml = `<div class="player-tag">
    <span class="wind">${wind}</span>
    <span>P${absSeat} ${name} (当前视角)</span>
    <span class="sc">${sc.toLocaleString()}</span>
    ${isRiichi ? `<span class="riichi-pill">立直</span>` : ""}
  </div>`;

  return `${renderRiver(f.rivers[absSeat])}${callBanner}<div class="ego-hand-wrap">${slotsHtml}</div>${renderMelds(f.melds[absSeat])}${tagHtml}`;
}

export function renderApp(root: HTMLElement, state: AppState): void {
  const report = state.report;
  if (!report) {
    root.innerHTML = `<div style="padding:48px;text-align:center;color:#94a3b8;">
      <h2 style="color:#38bdf8;">Sovereign-Mahjong V6 — KillerDucky / NAGA 复盘面板</h2>
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

  // Seat tabs
  const seatTabsHtml = Object.keys(report.seats)
    .map((k) => {
      const s = report.seats[k];
      const active = parseInt(k, 10) === state.seat ? " active" : "";
      return `<button class="seat-btn${active}" data-seat="${k}">
        <span>P${k} ${escapeHtml(s.name)}</span>
        <span class="rt">${s.rating.toFixed(1)}</span>
        <span style="color:#94a3b8">${Math.round(s.overall_acc * 100)}%</span>
      </button>`;
    })
    .join("");

  // Round selector
  const roundSeen = new Map<number, { label: string; firstIdx: number }>();
  frames.forEach((fr, idx) => {
    if (!roundSeen.has(fr.round_idx)) {
      roundSeen.set(fr.round_idx, { label: fr.round_label, firstIdx: idx });
    }
  });
  const roundOpts = Array.from(roundSeen.entries())
    .map(([rIdx, info]) => {
      const sel = f && f.round_idx === rIdx ? " selected" : "";
      return `<option value="${info.firstIdx}"${sel}>${escapeHtml(info.label)}</option>`;
    })
    .join("");

  // Compass
  const compassHtml = f
    ? `<div class="round-title">${escapeHtml(f.round_label)}</div>
       <div class="wall-meta">
         <span>余牌 <b>${f.wall_rem}</b></span>
         <span>供托 <b>${f.kyotaku}</b></span>
         <span>第 <b>${f.turn}</b> 巡</span>
       </div>
       <div class="dora-row"><span>宝牌指示:</span>${f.dora.map((t) => renderTileImg(t, "xs")).join("")}</div>`
    : `<div class="round-title">无匹配决策步</div>`;

  // Sidebar stats
  const statsHtml = curSeatInfo
    ? `<div class="stat-box"><div class="val cyan">${curSeatInfo.rating.toFixed(2)}</div><div class="lbl">Mortal Rating</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.overall_acc * 100).toFixed(1)}%</div><div class="lbl">总一致率</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.discard_acc * 100).toFixed(1)}%</div><div class="lbl">纯弃牌一致</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.call_acc * 100).toFixed(1)}%</div><div class="lbl">鸣牌/立直一致</div></div>
       <div class="stat-box"><div class="val">${(curSeatInfo.top3_acc * 100).toFixed(1)}%</div><div class="lbl">Top-3 一致</div></div>
       <div class="stat-box"><div class="val red">${curSeatInfo.blunder_count} (${(curSeatInfo.blunder_rate * 100).toFixed(1)}%)</div><div class="lbl">恶手数</div></div>`
    : "";

  // Verdict & candidates
  let verdictHtml = "";
  let candHtml = "";
  let riskHtml = "";
  let tenpaiHtml = "";
  if (f) {
    if (f.matched_top1) {
      verdictHtml = `<span class="mini-tag ok">一致 ✓ (r=100%)</span>`;
    } else if (f.is_blunder) {
      verdictHtml = `<span class="mini-tag bad">恶手 (r=${(f.prob_ratio * 100).toFixed(1)}%)</span>`;
    } else {
      verdictHtml = `<span class="mini-tag hu">分歧 (r=${(f.prob_ratio * 100).toFixed(1)}%)</span>`;
    }

    candHtml = (f.candidates ?? [])
      .slice(0, 6)
      .map((c) => {
        const isAi = c.a === f.best_action;
        const isHu = c.a === f.human_action;
        let color = "#475569";
        let tags = "";
        if (isAi && isHu) {
          color = "#10b981";
          tags = `<span class="mini-tag ok">一致</span>`;
        } else if (isAi) {
          color = "#38bdf8";
          tags = `<span class="mini-tag ai">AI</span>`;
        } else if (isHu) {
          color = f.is_blunder ? "#ef4444" : "#f59e0b";
          tags = `<span class="mini-tag ${f.is_blunder ? "bad" : "hu"}">实战</span>`;
        }
        return `<div class="cand-row">
          <div class="cand-label">${tags}<span>${escapeHtml(c.label)}</span></div>
          <div class="cand-bar-bg"><div class="cand-bar-fill" style="width:${Math.max(1, c.prob * 100)}%;background:${color}"></div></div>
          <div style="text-align:right;font-variant-numeric:tabular-nums">${(c.prob * 100).toFixed(1)}%</div>
        </div>`;
      })
      .join("");

    riskHtml = `放铳期望 ΔU_ron — 实战: <b>${f.delta_u_ron_human.toFixed(4)}</b> | AI推荐: <b>${f.delta_u_ron_best.toFixed(4)}</b>`;

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

  // Blunder quick-jump list
  const seatAllFrames = report.frames.filter((x) => x.seat === state.seat);
  const rankedBlunders = seatAllFrames
    .filter((x) => !x.matched_top1)
    .sort((a, b) => b.best_prob - b.human_prob - (a.best_prob - a.human_prob))
    .slice(0, 15);

  const blunderHtml = rankedBlunders
    .map((x) => {
      const isCur = f && x.idx === f.idx ? " active" : "";
      const badCls = x.is_blunder ? " is-bad" : "";
      return `<div class="blunder-item${badCls}${isCur}" data-jump-idx="${x.idx}">
        <span>${escapeHtml(x.round_label)} 第${x.turn}巡</span>
        <span>${escapeHtml(x.human_action_str)} → <b style="color:#38bdf8">${escapeHtml(x.best_action_str)}</b></span>
        <span style="color:${x.is_blunder ? "#f87171" : "#fbbf24"}">${Math.round(x.human_prob * 100)}%</span>
      </div>`;
    })
    .join("");

  // Timeline dots
  const timelineHtml = frames
    .map((fr, i) => {
      const cls = fr.matched_top1 ? "ok" : fr.is_blunder ? "bad" : "diff";
      const cur = i === state.cursor ? " cur" : "";
      return `<div class="dot ${cls}${cur}" data-step="${i}" title="${escapeHtml(fr.round_label)} 第${fr.turn}巡: ${escapeHtml(fr.human_action_str)} vs ${escapeHtml(fr.best_action_str)}"></div>`;
    })
    .join("");

  root.innerHTML = `<div class="app-shell">
    <div class="topbar">
      <span class="brand">Sovereign-Mahjong V6</span>
      <span class="badge">规则: ${escapeHtml(report.rule)}</span>
      <span class="badge" title="${escapeHtml(report.replay_source)}">${escapeHtml(report.replay_source)}</span>
      <div class="seat-tabs">${seatTabsHtml}</div>
      <div class="controls-right">
        <select id="roundSelect" title="跳转局次">${roundOpts}</select>
        <select id="filterSelect" title="过滤决策步">
          <option value="all"${state.filter === "all" ? " selected" : ""}>显示: 全部决策</option>
          <option value="diff"${state.filter === "diff" ? " selected" : ""}>仅看: 分歧手</option>
          <option value="blunder"${state.filter === "blunder" ? " selected" : ""}>仅看: 恶手</option>
        </select>
        <button class="btn${state.xray ? " active" : ""}" id="xrayBtn">👁 透视: ${state.xray ? "开" : "关"}</button>
        <label class="btn" title="载入本地 review_*.json">
          📂 打开JSON
          <input id="fileInput" type="file" accept=".json" style="display:none" />
        </label>
      </div>
    </div>

    <div class="main">
      <div class="table-wrap">
        <div class="seat-top">${f ? renderOppSeat(state, f, (f.seat + 2) % 4, 1) : ""}</div>
        <div class="table-mid">
          <div class="seat-left">${f ? renderOppSeat(state, f, (f.seat + 3) % 4, 2) : ""}</div>
          <div class="compass">${compassHtml}</div>
          <div class="seat-right">${f ? renderOppSeat(state, f, (f.seat + 1) % 4, 0) : ""}</div>
        </div>
        <div class="seat-bottom">${f ? renderEgoSeat(state, f) : `<div style="padding:32px;color:#94a3b8">当前筛选条件下无决策步</div>`}</div>
      </div>

      <div class="sidebar">
        <div class="card">
          <h3><span>P${state.seat} ${escapeHtml(curSeatInfo?.name ?? "")} 总览</span><span>${verdictHtml}</span></h3>
          <div class="stats-grid">${statsHtml}</div>
        </div>

        <div class="card">
          <h3><span>AI 候选分布 (Top-6)</span><span style="color:#38bdf8">${f ? `${escapeHtml(f.round_label)} · 第${f.turn}巡` : ""}</span></h3>
          <div>${candHtml}</div>
          <div style="margin-top:8px;font-size:12px;color:#94a3b8;">${riskHtml}</div>
        </div>

        <div class="card">
          <h3><span>V6 Belief 对手听牌概率</span><span style="font-size:11px;color:#94a3b8">B_EMA</span></h3>
          <div>${tenpaiHtml}</div>
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
        <button class="btn" id="prevStepBtn">◀ 上一手 (←)</button>
        <span class="step-counter">${frames.length ? `${state.cursor + 1} / ${frames.length}` : "0 / 0"}</span>
        <button class="btn" id="nextStepBtn">下一手 (→) ▶</button>
        <button class="btn" id="nextBlunderBtn">下一恶手 (↓) ⏩</button>
        <button class="btn" id="nextRoundBtn">下一局 ⏭</button>
      </div>
      <div class="timeline">${timelineHtml}</div>
      <div class="kbd-hint">快捷键：←/→ 切换步数 · ↑/↓ 跳转恶手 · 1~4 切换视角 · X 开关透视 · 支持直接拖入 review_*.json</div>
    </div>
  </div>`;
}

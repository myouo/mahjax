import { buildVisibleCounts34 } from "./board";
import { renderGitGraphHtml } from "./branch_graph";
import { calcDiscardUkeire, formatShantenCn } from "./shanten";
import { escHtml, playerName, tileShortLabel, tileSvgFilename } from "./tiles";
import type { ActionCandidate, AppState, ReviewFrame } from "./types";

const OPP_DISPLAY = [
  { idx: 2, relOffset: 3, label: "上家", short: "上", cssVar: "var(--opp-kami)" },
  { idx: 1, relOffset: 2, label: "对家", short: "对", cssVar: "var(--opp-toim)" },
  { idx: 0, relOffset: 1, label: "下家", short: "下", cssVar: "var(--opp-shim)" },
] as const;

function isCallOrSpecial(a: number): boolean {
  return (a >= 37 && a <= 70) || (a >= 72 && a <= 85);
}

function shortCallLabel(c: ActionCandidate): string {
  if (c.a === 84) return "跳过";
  if (c.a === 72) return "立直";
  if (c.a === 73) return "自摸";
  if (c.a === 74) return "荣和";
  if (c.a === 75 || c.a === 76) return "碰";
  if (c.a === 77) return "明杠";
  if (c.a >= 78 && c.a <= 83) return "吃";
  if (c.a === 85) return "九种";
  if (c.a >= 37 && c.a <= 70) return "杠";
  return c.label.slice(0, 4);
}

function renderConditionalCallSvg(f: ReviewFrame): string {
  const hasSpecial = (f.candidates ?? []).some((c) => isCallOrSpecial(c.a));
  if (!hasSpecial && f.matched_top1) return "";

  const shown: ActionCandidate[] = [];
  const seen = new Set<number>();
  for (const c of f.candidates ?? []) {
    if (isCallOrSpecial(c.a) || (!f.matched_top1 && (c.a === f.best_action || c.a === f.human_action))) {
      if (!seen.has(c.a)) {
        shown.push(c);
        seen.add(c.a);
      }
    }
    if (shown.length >= 4) break;
  }
  if (shown.length === 0) return "";

  const w = 286;
  const h = 76;
  const slotW = 54;
  const maxBarH = 26;
  let inner = "";
  shown.forEach((c, idx) => {
    const cx = 12 + idx * slotW + slotW / 2;
    const barH = c.prob > 0.004 ? Math.max(3, Math.round(c.prob * maxBarH)) : 2;
    const isBest = c.a === f.best_action;
    const isHuman = f.human_action >= 0 && c.a === f.human_action;
    const fill = isBest ? "var(--ai)" : isHuman ? (f.is_blunder ? "var(--bad)" : "var(--gold)") : "var(--ink-3)";
    const yBar = 35 - barH;

    inner += `<rect x="${cx - 9}" y="${yBar}" width="18" height="${barH}" rx="2.5" fill="${fill}"/>`;
    inner += `<text x="${cx}" y="${Math.max(10, yBar - 3)}" text-anchor="middle" fill="${fill}" font-size="9.5" font-weight="700">${(c.prob * 100).toFixed(0)}%</text>`;

    const lbl = shortCallLabel(c);
    if (isHuman) {
      inner += `<rect x="${cx - 18}" y="38" width="36" height="14" rx="3.5" fill="none" stroke="${f.is_blunder ? "var(--bad)" : "var(--gold)"}" stroke-width="1.3"/>`;
    }
    inner += `<text class="call-svg-lbl" x="${cx}" y="48.5" text-anchor="middle" font-size="9.5" font-weight="600">${escHtml(lbl)}</text>`;

    let iconTid = -1;
    if (c.a <= 36) iconTid = c.a;
    else if (c.a === 71) iconTid = f.last_draw;
    else if (c.a >= 37 && c.a <= 70) iconTid = c.a - 37;
    else if (isCallOrSpecial(c.a) && c.a !== 84 && c.a !== 72 && c.a !== 73 && f.target >= 0) iconTid = f.target;
    if (iconTid >= 0) {
      inner += `<rect x="${cx - 8}" y="54" width="16" height="20" rx="2" fill="var(--tile-face)" stroke="var(--line)" stroke-width="0.6"/>`;
      inner += `<image href="./tiles/${tileSvgFilename(iconTid)}" x="${cx - 7}" y="55" width="14" height="18"/>`;
    }
  });

  if (f.human_action >= 0 && !f.matched_top1) {
    const vx = 12 + shown.length * slotW + 8;
    const verdict = f.is_blunder ? "恶手" : f.prob_ratio >= 0.25 ? "接近" : "分歧";
    const vCol = f.is_blunder ? "var(--bad)" : "var(--gold)";
    if (vx < w - 14) {
      inner += `<text x="${vx}" y="34" fill="${vCol}" font-size="11" font-weight="700">${verdict}</text>`;
    }
  }
  return `<div class="call-svg-wrap"><svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${inner}</svg></div>`;
}

function renderCandidateTable(
  f: ReviewFrame,
  pov: number,
  userAns: number | undefined,
  includeHumanMark: boolean
): string {
  const myHand = f.hands[pov] ?? [];
  const myMelds = (f.melds[pov] ?? []).length;
  const vis = buildVisibleCounts34(f, pov);

  const rows = (f.candidates ?? [])
    .slice(0, 7)
    .map((c) => {
      const isAi = c.a === f.best_action;
      const isHu = includeHumanMark && f.human_action >= 0 && c.a === f.human_action;
      const isUsr = userAns !== undefined && userAns === c.a;
      let color = "var(--ink-3)";
      let marks = "";
      if (isAi) {
        color = "var(--ai)";
        marks += `<span class="mk mk-ai">AI</span>`;
      }
      if (isHu) {
        if (!isAi) color = f.is_blunder ? "var(--bad)" : "var(--gold)";
        marks += `<span class="mk ${f.is_blunder ? "mk-bad" : "mk-hu"}">实战</span>`;
      }
      if (isUsr) {
        if (!isAi && !isHu) color = "var(--ink)";
        marks += `<span class="mk mk-usr">你</span>`;
      }

      let ukeStr = "-";
      if (c.a <= 36 || c.a === 71) {
        const dTile = c.a === 71 ? f.last_draw : c.a;
        if (dTile >= 0) {
          const u = calcDiscardUkeire(myHand, dTile, myMelds, vis);
          ukeStr = `${formatShantenCn(u.shantenAfter)} ${u.ukeireCount}`;
        }
      }
      const rCol = c.du_ron < -0.02 ? "var(--bad)" : c.du_ron < -0.005 ? "var(--gold)" : "var(--ink-3)";
      return `<tr class="cand-row" data-cand-action="${c.a}">
        <td class="cand-act">${escHtml(c.label)}${marks}</td>
        <td class="cand-bar-td"><div class="cand-bar-bg"><div class="cand-bar-fill" style="width:${Math.max(2, c.prob * 100)}%;background:${color}"></div></div></td>
        <td class="num bold">${(c.prob * 100).toFixed(1)}%</td>
        <td class="num" style="color:${rCol}">${c.du_ron.toFixed(3)}</td>
        <td class="num muted">${ukeStr}</td>
      </tr>`;
    })
    .join("");

  return `<table class="cand-table">
    <thead><tr><th>动作</th><th>权重</th><th class="num">概率</th><th class="num">ΔU铳</th><th class="num">进张</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

const ICONS = {
  prevRound: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 3.5v9M11.5 4L6.5 8l5 4"/></svg>`,
  nextRound: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5v9M4.5 4l5 4-5 4"/></svg>`,
  prevErr: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4L3 8l4 4"/><path d="M11.5 4.2v4.3"/><circle cx="11.5" cy="11.4" r="0.8" fill="currentColor" stroke="none"/></svg>`,
  nextErr: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 4l4 4-4 4"/><path d="M4.5 4.2v4.3"/><circle cx="4.5" cy="11.4" r="0.8" fill="currentColor" stroke="none"/></svg>`,
  prevDec: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 4L5 8l4 4"/><path d="M12.5 5v6"/></svg>`,
  nextDec: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4l4 4-4 4"/><path d="M3.5 5v6"/></svg>`,
  prevPly: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 4L6 8l4 4"/></svg>`,
  nextPly: `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 4l4 4-4 4"/></svg>`,
};

function renderReviewInspector(
  f: ReviewFrame,
  state: AppState,
  names: string[]
): string {
  const report = state.report!;
  const pov = state.povSeat;
  const frames = report.frames ?? [];
  const rounds = report.rounds ?? [];
  const rMeta = rounds.find((r) => r.round_idx === f.round_idx);
  const startIdx = rMeta ? rMeta.first_frame_idx : 0;
  const endIdx = rMeta ? rMeta.last_frame_idx : Math.max(0, frames.length - 1);
  const roundLen = Math.max(1, endIdx - startIdx + 1);
  const relPos = Math.max(0, state.frameIdx - startIdx);
  const pct = roundLen > 1 ? (relPos / (roundLen - 1)) * 100 : 0;

  let scrubTicks = "";
  for (let i = startIdx; i <= endIdx; i++) {
    const fr = frames[i];
    if (fr.seat === pov && fr.has_ai !== false && (fr.is_decision ?? true) && !fr.matched_top1) {
      const x = roundLen > 1 ? ((i - startIdx) / (roundLen - 1)) * 100 : 0;
      scrubTicks += `<span class="scrub-tick ${fr.is_blunder ? "bad" : "diff"}" style="left:${x.toFixed(2)}%"></span>`;
    }
  }

  const navAndScrub = `
    <div class="insp-section">
      <div class="scrub-header">
        <button class="scrub-rnd-btn" id="openRoundOverviewBtn">
          <span>${escHtml(f.round_label)}</span>
          <span class="caret">▾</span>
        </button>
        <span class="scrub-pos">${relPos + 1} / ${roundLen} · ${f.turn}巡</span>
      </div>
      <div class="scrub-track" id="roundScrubTrack" data-scrub-start="${startIdx}" data-scrub-end="${endIdx}">
        <div class="scrub-fill" style="width:${pct.toFixed(2)}%"></div>
        ${scrubTicks}
        <div class="scrub-thumb" style="left:${pct.toFixed(2)}%"></div>
      </div>
      <div class="transport-bar" role="toolbar" aria-label="时间轴导航">
        <button class="tp-btn" id="prevRoundBtn" aria-label="上一局">${ICONS.prevRound}<span>局</span></button>
        <button class="tp-btn is-err" id="prevErrBtn" aria-label="上一恶手">${ICONS.prevErr}<span>恶</span></button>
        <button class="tp-btn" id="prevDecBtn" aria-label="上一决策">${ICONS.prevDec}<span>决</span></button>
        <button class="tp-btn" id="prevPlyBtn" aria-label="上一帧">${ICONS.prevPly}<span>帧</span></button>
        <span class="tp-sep"></span>
        <button class="tp-btn" id="nextPlyBtn" aria-label="下一帧">${ICONS.nextPly}<span>帧</span></button>
        <button class="tp-btn" id="nextDecBtn" aria-label="下一决策">${ICONS.nextDec}<span>决</span></button>
        <button class="tp-btn is-err" id="nextErrBtn" aria-label="下一恶手">${ICONS.nextErr}<span>恶</span></button>
        <button class="tp-btn" id="nextRoundBtn" aria-label="下一局">${ICONS.nextRound}<span>局</span></button>
      </div>
    </div>
    <div class="tab-bar">
      <button class="tab-btn${state.reviewTab === "analysis" ? " active" : ""}" data-review-tab="analysis">分析</button>
      <button class="tab-btn${state.reviewTab === "mistakes" ? " active" : ""}" data-review-tab="mistakes">恶手</button>
      <button class="tab-btn${state.reviewTab === "danger" ? " active" : ""}" data-review-tab="danger">危险</button>
      <button class="tab-btn${state.reviewTab === "stats" ? " active" : ""}" data-review-tab="stats">统计</button>
    </div>
  `;

  let tabBody = "";
  const isPov = f.seat === pov;

  if (state.reviewTab === "analysis") {
    let summaryLine = "";
    let mainContent = "";
    if (f.event_kind === "round_end") {
      summaryLine = `<div class="summary-strip"><span class="muted">本局已结束</span><button class="link-btn" id="openThisRoundBtn">查看结算明细</button></div>`;
    } else if (!isPov || f.has_ai === false || !(f.candidates?.length)) {
      summaryLine = `<div class="summary-strip"><span>${escHtml(playerName(names, f.seat))}: <b>${escHtml(f.human_action_str || f.event_kind || "")}</b></span></div>`;
      mainContent = `<div class="quiet-note">当前为他家行动帧。滚动「决」或 Shift+滚轮可直达自家决策帧；点击牌河任意牌可直接跳回该切牌帧。</div>`;
    } else if (!state.showAI) {
      summaryLine = `<div class="summary-strip"><span>自家回合 · AI 评估已隐藏</span><button class="link-btn" id="aiRevealInlineBtn">显示 (M)</button></div>`;
      mainContent = `<div class="quiet-note">按 M 或点击牌桌 AI 柱区显隐评估；左键手牌或右键摸切可分叉推演。</div>`;
    } else {
      const ratioPct = Math.round(f.prob_ratio * 100);
      const verdictCls = f.matched_top1 ? "ok" : f.is_blunder ? "bad" : "hu";
      const verdictTxt = f.matched_top1 ? "一致" : f.is_blunder ? `恶手 ${ratioPct}%` : `分歧 ${ratioPct}%`;
      summaryLine = `<div class="summary-strip">
        <span>实战 <b>${escHtml(f.human_action_str)}</b> (${(f.human_prob * 100).toFixed(1)}%) · AI <b>${escHtml(f.best_action_str)}</b> (${(f.best_prob * 100).toFixed(1)}%)</span>
        <span class="verdict-txt ${verdictCls}">${verdictTxt}</span>
      </div>`;
      mainContent = `${renderConditionalCallSvg(f)}${renderCandidateTable(f, pov, undefined, true)}`;
    }
    tabBody = `<div class="tab-pane">
      ${summaryLine}
      <div class="tab-scroll">${mainContent}</div>
    </div>`;
  } else if (state.reviewTab === "mistakes") {
    const ranked = frames
      .filter((x) => x.seat === pov && (x.is_decision ?? true) && x.has_ai !== false && !x.matched_top1)
      .sort((a, b) => b.best_prob - b.human_prob - (a.best_prob - a.human_prob));
    const listHtml = ranked.length
      ? ranked
          .map((x) => {
            const active = x.idx === f.idx ? " active" : "";
            const cls = x.is_blunder ? " is-bad" : "";
            return `<div class="mistake-row${cls}${active}" data-jump-frame="${x.idx}">
              <span class="mr-rnd">${escHtml(x.round_label.replace(" 0本场", ""))} ${x.turn}巡</span>
              <span class="mr-act">${escHtml(x.human_action_str)} → <b>${escHtml(x.best_action_str)}</b></span>
              <span class="mr-pct ${x.is_blunder ? "bad" : "hu"}">${Math.round(x.human_prob * 100)}%</span>
            </div>`;
          })
          .join("")
      : `<div class="quiet-note">${escHtml(playerName(names, pov))} 在本局中无明显分歧或恶手。</div>`;
    tabBody = `<div class="tab-pane"><div class="tab-scroll">${listHtml}</div></div>`;
  } else if (state.reviewTab === "danger") {
    const tenpaiHtml =
      isPov && f.has_ai !== false
        ? OPP_DISPLAY.map((o) => {
            const p = f.p_tenpai?.[o.idx] ?? 0;
            const oppSeat = (pov + o.relOffset) % 4;
            return `<div class="tenpai-row">
              <span class="tp-lbl"><i class="tp-dot" style="background:${o.cssVar}"></i>${o.short} · ${escHtml(playerName(names, oppSeat))}</span>
              <div class="tenpai-bg"><div class="tenpai-fill" style="width:${(p * 100).toFixed(1)}%;background:${o.cssVar}"></div></div>
              <span class="tp-val">${Math.round(p * 100)}%</span>
            </div>`;
          }).join("")
        : `<div class="quiet-note">需在自家决策帧查看 V6 Belief 对手听牌估计。</div>`;

    const riskEntries = Object.entries(f.p_deal ?? {})
      .map(([k, vec]) => {
        const comb = 1 - (1 - vec[0]) * (1 - vec[1]) * (1 - vec[2]);
        return { t34: parseInt(k, 10), vec, comb };
      })
      .filter((x) => x.comb >= 0.015)
      .sort((a, b) => b.comb - a.comb)
      .slice(0, 8);

    const riskTable =
      riskEntries.length > 0
        ? `<table class="cand-table" style="margin-top:8px">
            <thead><tr><th>高危牌</th><th class="num">上家</th><th class="num">对家</th><th class="num">下家</th><th class="num">综合</th></tr></thead>
            <tbody>${riskEntries
              .map(
                (r) => `<tr>
                  <td><b>${tileShortLabel(r.t34)}</b></td>
                  <td class="num">${(r.vec[2] * 100).toFixed(1)}%</td>
                  <td class="num">${(r.vec[1] * 100).toFixed(1)}%</td>
                  <td class="num">${(r.vec[0] * 100).toFixed(1)}%</td>
                  <td class="num bold">${(r.comb * 100).toFixed(1)}%</td>
                </tr>`
              )
              .join("")}</tbody>
          </table>`
        : `<div class="quiet-note" style="margin-top:8px">当前场面无显著高于 1.5% 的危险牌。</div>`;

    tabBody = `<div class="tab-pane">
      <div class="tab-scroll">
        <div class="sub-head"><span>V6 Belief 对手听牌概率</span><button class="link-btn" id="openDangerBtn">34 牌完整表 (Z)</button></div>
        ${tenpaiHtml}
        ${isPov && f.has_ai !== false ? riskTable : ""}
      </div>
    </div>`;
  } else {
    const order = [pov, (pov + 3) % 4, (pov + 2) % 4, (pov + 1) % 4];
    const relTag = (s: number) =>
      s === pov ? "自家" : s === (pov + 3) % 4 ? "上家" : s === (pov + 2) % 4 ? "对家" : "下家";
    const rows = order
      .map((p) => {
        const s = report.seats[String(p)];
        if (!s) return "";
        const act = p === pov ? " is-cur" : "";
        return `<tr class="stat-row${act}" data-seat="${p}">
          <td><span class="rel-pill ${p === pov ? "is-self" : ""}">${relTag(p)}</span>${escHtml(playerName(names, p))}</td>
          <td class="num bold">${s.rating.toFixed(1)}</td>
          <td class="num">${(s.overall_acc * 100).toFixed(1)}%</td>
          <td class="num">${(s.discard_acc * 100).toFixed(1)}%</td>
          <td class="num">${(s.call_acc * 100).toFixed(1)}%</td>
          <td class="num">${s.blunder_count}</td>
        </tr>`;
      })
      .join("");
    tabBody = `<div class="tab-pane">
      <div class="tab-scroll">
        <div class="sub-head"><span>全员简要评分</span><button class="link-btn" id="openReportBtn">打开完整报告表 ›</button></div>
        <table class="cand-table">
          <thead><tr><th>玩家</th><th class="num">Rating</th><th class="num">总一致</th><th class="num">弃牌</th><th class="num">鸣牌</th><th class="num">恶手</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div class="quiet-note" style="margin-top:10px">规则预设: ${escHtml(report.rule)} · 共 ${report.rounds_total} 局 (失步 ${report.rounds_desynced})</div>
      </div>
    </div>`;
  }

  return `${navAndScrub}${tabBody}`;
}

function renderWwydInspector(
  f: ReviewFrame,
  state: AppState,
  userAns: number | undefined,
  names: string[]
): string {
  const frames = state.report?.frames ?? [];
  const byIdx = new Map<number, ReviewFrame>();
  for (const fr of frames) byIdx.set(fr.idx, fr);
  let answered = 0;
  let matched = 0;
  let sum = 0;
  for (const [k, act] of Object.entries(state.wwydAnswers)) {
    const fr = byIdx.get(parseInt(k, 10));
    if (!fr) continue;
    answered++;
    if (act === fr.best_action) matched++;
    const cand = (fr.candidates ?? []).find((c) => c.a === act);
    sum += Math.min(1, (cand ? cand.prob : 0) / Math.max(fr.best_prob, 1e-6)) * 100;
  }
  const acc = answered > 0 ? `${Math.round((matched / answered) * 100)}%` : "-";
  const rt = answered > 0 ? (sum / answered).toFixed(1) : "-";
  const isPovDec = f.seat === state.povSeat && (f.is_decision ?? true) && f.event_kind !== "round_end";
  const hideAnswer = isPovDec && userAns === undefined;

  const header = `
    <div class="insp-section">
      <div class="quiz-score-bar">
        <span>已答 <b>${answered}</b></span>
        <span>一致 <b>${acc}</b></span>
        <span>Rating <b>${rt}</b></span>
        <button class="link-btn" id="resetWwydBtn">重置</button>
      </div>
      <div class="transport-bar two-col">
        <button class="tp-btn wide" id="prevDecBtn">${ICONS.prevDec}<span>上一题</span></button>
        <button class="tp-btn wide" id="nextDecBtn"><span>下一题</span>${ICONS.nextDec}</button>
      </div>
      <div class="scrub-header">
        <span class="scrub-rnd-static">${escHtml(f.round_label)} · ${f.turn}巡</span>
        <span class="scrub-pos">${isPovDec ? (hideAnswer ? "待作答" : "已揭晓") : "非自家决策"}</span>
      </div>
    </div>
  `;

  if (!isPovDec) {
    return `${header}<div class="tab-pane"><div class="quiet-note">当前帧非 ${escHtml(playerName(names, state.povSeat))} 的决策。点击「下一题」跳转到下一手。</div></div>`;
  }

  if (hideAnswer) {
    return `${header}<div class="tab-pane">
      <div class="tab-scroll">
        <div class="wwyd-prompt-box">
          <div class="wp-title">盲测思考中</div>
          <div class="wp-desc">左键点击牌桌手牌切出，或右键点击桌面/手牌执行摸切/跳过；作答后即刻对比 AI 与实战选择。</div>
        </div>
      </div>
    </div>`;
  }

  const userCand = (f.candidates ?? []).find((c) => c.a === userAns);
  const userLabel = userCand ? userCand.label : `#${userAns}`;
  const isMatch = userAns === f.best_action;
  const verdictLine = `<div class="summary-strip">
    <span>你选 <b>${escHtml(userLabel)}</b> · AI <b>${escHtml(f.best_action_str)}</b> · 实战 <b>${escHtml(f.human_action_str)}</b></span>
    <span class="verdict-txt ${isMatch ? "ok" : "hu"}">${isMatch ? "一致" : "差异"}</span>
  </div>`;

  const forkActionBtn = state.backendOnline
    ? `<button class="btn" id="wwydForkBtn" data-wwyd-fork="${userAns}">以此手进入推演</button>`
    : "";

  return `${header}<div class="tab-pane">
    ${verdictLine}
    <div class="tab-scroll">
      ${renderCandidateTable(f, state.povSeat, userAns, true)}
      <div class="wwyd-post-actions">
        <button class="btn primary" id="wwydNextBtn">下一题 ›</button>
        ${forkActionBtn}
      </div>
    </div>
  </div>`;
}

function renderSandboxInspector(f: ReviewFrame, state: AppState, names: string[]): string {
  const br = state.branch;
  const pov = state.povSeat;

  if (!br.active) {
    const isPovTurn = f.seat === pov && f.event_kind !== "round_end" && (f.legal_actions?.length ?? 0) > 0;
    return `<div class="insp-section">
      <div class="sandbox-head">
        <span>推演就绪</span>
        <button class="link-btn" id="exitSandboxBtn">返回复盘 (Esc)</button>
      </div>
      <div class="transport-bar two-col">
        <button class="tp-btn wide" id="prevDecBtn">${ICONS.prevDec}<span>上一决策</span></button>
        <button class="tp-btn wide" id="nextDecBtn"><span>下一决策</span>${ICONS.nextDec}</button>
      </div>
    </div>
    <div class="tab-pane">
      <div class="tab-scroll">
        <div class="quiet-note">${
          isPovTurn
            ? "基于原局真实牌山，三家由 V6 AI 实时应对；左键手牌或右键摸切即可分叉推演。"
            : "当前非自家回合，请先点「上一决策」或「下一决策」切换至自家回合。"
        }</div>
        ${isPovTurn && state.showAI && (f.candidates?.length ?? 0) > 0 ? renderCandidateTable(f, pov, undefined, false) : ""}
      </div>
    </div>`;
  }

  const isPovTurn = f.seat === pov && f.event_kind !== "round_end";
  const liveAiHtml =
    !isPovTurn || !(f.candidates?.length)
      ? ""
      : !state.showAI
      ? `<div class="quiet-note">AI 评估已隐藏 (按 M 显隐)</div>`
      : renderCandidateTable(f, pov, undefined, false);

  return `<div class="insp-section">
    <div class="sandbox-head">
      <span>推演分支图 · ${escHtml(br.forkLabel)}</span>
      <span class="scrub-pos">${br.branches.length} 条分支</span>
    </div>
    <div class="sandbox-controls">
      <button class="btn" id="undoBranchBtn">撤销上步</button>
      <button class="btn" id="deleteBranchBtn">删除分支</button>
      <button class="btn" id="exitBranchBtn">返回复盘</button>
    </div>
  </div>
  <div class="tab-pane">
    <div class="tab-scroll">
      ${renderGitGraphHtml(state, names)}
      ${liveAiHtml ? `<div class="sub-head" style="margin-top:10px"><span>实时 V6 建议</span></div>${liveAiHtml}` : ""}
    </div>
  </div>`;
}

export function renderSidebar(
  f: ReviewFrame,
  state: AppState,
  _hideAnswer: boolean,
  userAns: number | undefined
): void {
  const rep = state.report!;
  const names =
    rep.player_names ?? [0, 1, 2, 3].map((p) => rep.seats[String(p)]?.name || `玩家${p + 1}`);
  const host = document.getElementById("sidebarHost")!;

  if (state.mode === "wwyd") {
    host.innerHTML = renderWwydInspector(f, state, userAns, names);
  } else if (state.mode === "sandbox") {
    host.innerHTML = renderSandboxInspector(f, state, names);
  } else {
    host.innerHTML = renderReviewInspector(f, state, names);
  }
}

import { escHtml, playerName, renderMeldGroup, renderTileImg, seatWindCn, tileShortLabel } from "./tiles";
import type { AppState, ReviewFrame, RoundMeta, RoundResult } from "./types";

const REASON_CN: Record<string, string> = {
  kyuushu: "九种九牌",
  four_winds: "四风连打",
  four_kans: "四杠散了",
  four_riichi: "四家立直",
  triple_ron: "三家和了",
};

const RULE_LABEL_CN: Record<string, string> = {
  "7d": "天凤凤桌 7段",
  "10d": "天凤凤桌 10段",
  majsoul: "雀魂王座",
  mleague: "M-League",
};

function getNames(state: AppState): string[] {
  const rep = state.report!;
  return [0, 1, 2, 3].map((p) => rep.player_names?.[p] || rep.seats[String(p)]?.name || `玩家${p + 1}`);
}

function relSeatTag(seat: number, pov: number): { text: string; isSelf: boolean } {
  if (seat === pov) return { text: "自家", isSelf: true };
  if (seat === (pov + 3) % 4) return { text: "上家", isSelf: false };
  if (seat === (pov + 2) % 4) return { text: "对家", isSelf: false };
  return { text: "下家", isSelf: false };
}

function roundOutcomeBadge(r: RoundMeta, names: string[]): string {
  const res = r.result;
  if (!res) {
    const maxDelta = Math.max(...r.deltas);
    return maxDelta > 0 ? "和了" : "流局";
  }
  if (res.type === "ron") return res.winners.map((w) => `${playerName(names, w.seat)} 荣和`).join(" / ");
  if (res.type === "tsumo") return res.winners.map((w) => `${playerName(names, w.seat)} 自摸`).join(" / ");
  if (res.type === "abortive") return REASON_CN[res.reason ?? ""] ?? "途中流局";
  if (res.type === "desync") return "失步";
  return "荒牌流局";
}

function renderRoundOverviewModal(state: AppState): { title: string; body: string } {
  const rep = state.report!;
  const rounds = rep.rounds ?? [];
  const names = getNames(state);
  const curRound = state.branch.active
    ? state.branch.forkRoundIdx
    : rep.frames[state.frameIdx]?.round_idx ?? 0;

  const rows = rounds
    .map((r) => {
      const active = r.round_idx === curRound ? " is-cur" : "";
      const outcome = roundOutcomeBadge(r, names);
      const cells = [0, 1, 2, 3]
        .map((p) => {
          const sc = r.init_scores[p];
          const d = r.deltas[p];
          const dCls = d > 0 ? "pos" : d < 0 ? "neg" : "zero";
          const dStr = d === 0 ? "±0" : `${d > 0 ? "+" : ""}${d}`;
          const bl = r.blunder_counts[p] ?? 0;
          return `<td><div class="ro-sc">${sc.toLocaleString()}</div><div class="ro-dl ${dCls}">${dStr}${bl > 0 ? ` <span class="ro-bl">${bl}恶</span>` : ""}</div></td>`;
        })
        .join("");
      return `<tr class="ro-row${active}" data-jump-round="${r.round_idx}">
        <td class="ro-lbl">${escHtml(r.round_label)}</td>
        <td><span class="ro-badge">${escHtml(outcome)}</span></td>
        ${cells}
      </tr>`;
    })
    .join("");

  const lastR = rounds[rounds.length - 1];
  const finalCells = lastR
    ? [0, 1, 2, 3].map((p) => `<td><b>${lastR.end_scores[p].toLocaleString()}</b></td>`).join("")
    : "<td>-</td><td>-</td><td>-</td><td>-</td>";

  return {
    title: "全半庄小局筛选与总览（点击任意局跳转）",
    body: `<div class="modal-scroll">
      <table class="ro-table">
        <thead>
          <tr>
            <th>局次</th>
            <th>结果</th>
            ${[0, 1, 2, 3].map((p) => `<th>${escHtml(playerName(names, p))}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${rows}
          <tr class="ro-final-row">
            <td colspan="2"><b>终局得点</b></td>
            ${finalCells}
          </tr>
        </tbody>
      </table>
    </div>`,
  };
}

function renderThisRoundModal(f: ReviewFrame, state: AppState): { title: string; body: string } {
  const rep = state.report!;
  const names = getNames(state);
  const rMeta = (rep.rounds ?? []).find((r) => r.round_idx === f.round_idx);
  const res: RoundResult | null = f.round_result ?? rMeta?.result ?? null;

  if (!res) {
    return {
      title: `${f.round_label} 结算明细`,
      body: `<div class="quiet-note">旧版 JSON 缺少该局结算元数据；通过本地后端重新评测即可查看完整和牌役种与结算明细。</div>`,
    };
  }

  let detailHtml = "";
  if (res.type === "ron" || res.type === "tsumo") {
    detailHtml = res.winners
      .map((w) => {
        const handTiles = (w.hand ?? []).map((t) => renderTileImg(t, "sm")).join("");
        const meldsHtml = (w.melds ?? []).map((m) => renderMeldGroup(m, w.seat, "xs")).join("");
        const winTile = renderTileImg(w.winningTile, "sm", "win-tile-gap");
        const yakuRows = (w.yaku ?? [])
          .map((y) => `<span class="yaku-pill">${escHtml(y.nameZh || y.name)} <b>${y.han}${w.yakuman ? "倍" : "飜"}</b></span>`)
          .join("");
        const extraHan = [
          w.doraHan > 0 ? `<span class="yaku-pill dora">宝牌 <b>${w.doraHan}飜</b></span>` : "",
          w.akaHan > 0 ? `<span class="yaku-pill dora">赤宝牌 <b>${w.akaHan}飜</b></span>` : "",
          w.uraHan > 0 ? `<span class="yaku-pill dora">里宝牌 <b>${w.uraHan}飜</b></span>` : "",
        ].join("");
        const doraStrip = (w.dora ?? []).length
          ? `<div class="res-dora-row"><span>表宝指示:</span>${w.dora.map((d) => renderTileImg(d, "xs")).join("")}${
              w.uraDora?.length
                ? `<span style="margin-left:10px">里宝指示:</span>${w.uraDora.map((d) => renderTileImg(d, "xs")).join("")}`
                : ""
            }</div>`
          : "";
        return `<div class="winner-card">
          <div class="winner-head">
            <span><b>${escHtml(playerName(names, w.seat))}</b> · ${
              w.from !== null ? `荣和 ${escHtml(playerName(names, w.from))}` : "自摸"
            }</span>
            <span class="win-pts">${w.yakuman ? `${w.yakuman}倍役满` : `${w.han}飜 ${w.fu}符`} · <b>+${w.points.toLocaleString()}</b></span>
          </div>
          <div class="winner-hand-row">${handTiles}${winTile}${meldsHtml ? `<span class="res-melds">${meldsHtml}</span>` : ""}</div>
          ${doraStrip}
          <div class="yaku-grid">${yakuRows}${extraHan}</div>
        </div>`;
      })
      .join("");
  } else {
    const title = res.type === "abortive" ? REASON_CN[res.reason ?? ""] ?? "途中流局" : "荒牌流局";
    const tenpaiList = [0, 1, 2, 3]
      .map((p) => {
        const tp = res.tenpai?.[p] ?? false;
        return `<div class="draw-seat-row">
          <span>${seatWindCn(p, f.dealer)}家 · ${escHtml(playerName(names, p))}</span>
          <span class="${tp ? "pos" : "muted"}">${tp ? "听牌" : "未听"}</span>
        </div>`;
      })
      .join("");
    detailHtml = `<div class="winner-card"><div class="winner-head"><span>${title}</span></div>${tenpaiList}</div>`;
  }

  const deltaTable = `<table class="res-score-table">
    <thead><tr><th>风位</th><th>玩家</th><th>结算前</th><th>变动</th><th>结算后</th></tr></thead>
    <tbody>
      ${[0, 1, 2, 3]
        .map((p) => {
          const after = res.scores[p];
          const d = res.deltas[p];
          const before = after - d;
          const dCls = d > 0 ? "pos" : d < 0 ? "neg" : "zero";
          return `<tr>
            <td>${seatWindCn(p, f.dealer)}家</td>
            <td>${escHtml(playerName(names, p))}</td>
            <td class="num">${before.toLocaleString()}</td>
            <td class="num ${dCls}">${d >= 0 ? "+" : ""}${d.toLocaleString()}</td>
            <td class="num"><b>${after.toLocaleString()}</b></td>
          </tr>`;
        })
        .join("")}
    </tbody>
  </table>`;

  return {
    title: `${f.round_label} · 单局结算明细`,
    body: `<div class="modal-scroll">${detailHtml}${deltaTable}</div>`,
  };
}

function renderDangerModal(f: ReviewFrame, state: AppState): { title: string; body: string } {
  const pov = state.povSeat;
  const names = getNames(state);
  const pDeal = f.p_deal ?? {};
  const suits: Array<{ label: string; start: number; count: number }> = [
    { label: "万子 (m)", start: 0, count: 9 },
    { label: "筒子 (p)", start: 9, count: 9 },
    { label: "索子 (s)", start: 18, count: 9 },
    { label: "字牌 (z)", start: 27, count: 7 },
  ];

  const tables = suits
    .map((s) => {
      const rows = Array.from({ length: s.count }, (_, i) => {
        const t34 = s.start + i;
        const vec = pDeal[String(t34)];
        const d0 = vec ? vec[0] : 0; // shimocha
        const d1 = vec ? vec[1] : 0; // toimen
        const d2 = vec ? vec[2] : 0; // kamicha
        const comb = 1 - (1 - d0) * (1 - d1) * (1 - d2);
        const bg =
          comb >= 0.12
            ? "rgba(192, 62, 53, 0.15)"
            : comb >= 0.05
            ? "rgba(184, 122, 30, 0.13)"
            : "transparent";
        const fmt = (v: number) => (vec ? `${(v * 100).toFixed(1)}%` : "<1%");
        return `<tr style="background:${bg}">
          <td><div class="d-tile-cell">${renderTileImg(t34, "mini")}<span>${tileShortLabel(t34)}</span></div></td>
          <td class="num" style="color:var(--opp-kami)">${fmt(d2)}</td>
          <td class="num" style="color:var(--opp-toim)">${fmt(d1)}</td>
          <td class="num" style="color:var(--opp-shim)">${fmt(d0)}</td>
          <td class="num bold">${vec ? `${(comb * 100).toFixed(1)}%` : "<1%"}</td>
        </tr>`;
      }).join("");
      return `<div class="danger-suit-card">
        <h4>${s.label}</h4>
        <table class="danger-tbl">
          <thead><tr>
            <th>牌</th>
            <th>上 · ${escHtml(playerName(names, (pov + 3) % 4).slice(0, 6))}</th>
            <th>对 · ${escHtml(playerName(names, (pov + 2) % 4).slice(0, 6))}</th>
            <th>下 · ${escHtml(playerName(names, (pov + 1) % 4).slice(0, 6))}</th>
            <th>综合</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
    })
    .join("");

  return {
    title: `34 牌实时放铳危险度明细 (${f.round_label} · 第${f.turn}巡)`,
    body: `<div class="modal-scroll"><div class="danger-grid">${tables}</div></div>`,
  };
}

function renderReportModal(state: AppState): { title: string; body: string } {
  const rep = state.report!;
  const pov = state.povSeat;
  const names = getNames(state);
  const rounds = rep.rounds ?? [];
  const frames = rep.frames ?? [];

  const wins = [0, 0, 0, 0];
  const tsumo = [0, 0, 0, 0];
  const ron = [0, 0, 0, 0];
  const dealIns = [0, 0, 0, 0];
  const winPtsSum = [0, 0, 0, 0];
  const riichiRoundSets: Array<Set<number>> = [new Set(), new Set(), new Set(), new Set()];
  const openRoundSets: Array<Set<number>> = [new Set(), new Set(), new Set(), new Set()];
  const worstBlunder: Array<{ frame: ReviewFrame; gap: number } | null> = [null, null, null, null];

  for (const r of rounds) {
    const res = r.result;
    if (res && (res.type === "ron" || res.type === "tsumo")) {
      for (const w of res.winners) {
        if (w.seat >= 0 && w.seat < 4) {
          wins[w.seat]++;
          winPtsSum[w.seat] += w.points;
          if (w.from !== null && w.from >= 0 && w.from < 4) {
            ron[w.seat]++;
            dealIns[w.from]++;
          } else {
            tsumo[w.seat]++;
          }
        }
      }
    }
  }

  for (const fr of frames) {
    for (let p = 0; p < 4; p++) {
      if (fr.riichi?.[p]) riichiRoundSets[p].add(fr.round_idx);
      if ((fr.melds?.[p] ?? []).some((m) => m.kind !== "kan_closed")) {
        openRoundSets[p].add(fr.round_idx);
      }
    }
    const s = fr.seat;
    if (s >= 0 && s < 4 && fr.has_ai !== false && (fr.is_decision ?? true) && !fr.matched_top1) {
      const gap = fr.best_prob - fr.human_prob;
      if (!worstBlunder[s] || gap > worstBlunder[s]!.gap) {
        worstBlunder[s] = { frame: fr, gap };
      }
    }
  }

  const lastR = rounds[rounds.length - 1];
  const lastFrame = frames[frames.length - 1];
  const endScores: [number, number, number, number] = lastR
    ? lastR.end_scores
    : lastFrame
    ? lastFrame.scores
    : [25000, 25000, 25000, 25000];
  const rankedSeats = [0, 1, 2, 3].sort((a, b) => endScores[b] - endScores[a] || a - b);
  const rankMap: Record<number, number> = {};
  rankedSeats.forEach((s, i) => {
    rankMap[s] = i + 1;
  });

  const order = [pov, (pov + 3) % 4, (pov + 2) % 4, (pov + 1) % 4];
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  const aiRows = order
    .map((p) => {
      const s = rep.seats[String(p)];
      if (!s) return "";
      const rel = relSeatTag(p, pov);
      const rk = rankMap[p] ?? "-";
      const sc = endScores[p] ?? 0;
      const barW = Math.max(4, Math.min(100, s.rating));
      return `<tr class="stat-row${rel.isSelf ? " is-cur" : ""}" data-seat="${p}">
        <td>
          <div class="rep-player-cell">
            <span class="rel-pill${rel.isSelf ? " is-self" : ""}">${rel.text}</span>
            <b>${escHtml(playerName(names, p))}</b>
          </div>
          <div class="rt-bar-wrap"><span class="rt-bar" style="width:${barW.toFixed(1)}%"></span></div>
        </td>
        <td class="num"><b>${rk}位</b><div class="muted">${sc.toLocaleString()}</div></td>
        <td class="num"><b>${s.rating.toFixed(1)}</b></td>
        <td class="num"><b>${pct(s.overall_acc)}</b><div class="muted">共 ${s.total_decisions} 手</div></td>
        <td class="num">${pct(s.discard_acc)}<div class="muted">${s.discard_decisions} 手</div></td>
        <td class="num">${s.call_decisions > 0 ? pct(s.call_acc) : "-"}<div class="muted">${s.call_decisions} 次</div></td>
        <td class="num"><b class="${s.blunder_count > 0 ? "neg" : ""}">${s.blunder_count} 次</b><div class="muted">${pct(s.blunder_rate)}</div></td>
      </tr>`;
    })
    .join("");

  const playRows = order
    .map((p) => {
      const rel = relSeatTag(p, pov);
      const w = wins[p];
      const avgWin = w > 0 ? Math.round(winPtsSum[p] / w).toLocaleString() : "-";
      const wb = worstBlunder[p]?.frame;
      const wbCell = wb
        ? `<button class="link-btn report-jump-btn" data-jump-frame="${wb.idx}">${escHtml(
            wb.round_label.replace(" 0本场", "")
          )} ${wb.turn}巡 · ${escHtml(wb.human_action_str)} → <b>${escHtml(wb.best_action_str)}</b> (${Math.round(
            wb.human_prob * 100
          )}%)</button>`
        : `<span class="muted">无明显失误</span>`;
      return `<tr class="${rel.isSelf ? "is-self-row" : ""}">
        <td><span class="rel-pill${rel.isSelf ? " is-self" : ""}">${rel.text}</span><b>${escHtml(playerName(names, p))}</b></td>
        <td class="num"><b>${w}</b> <span class="muted">(${tsumo[p]}/${ron[p]})</span></td>
        <td class="num${dealIns[p] > 0 ? " neg" : ""}">${dealIns[p]}</td>
        <td class="num">${riichiRoundSets[p].size} 局</td>
        <td class="num">${openRoundSets[p].size} 局</td>
        <td class="num">${avgWin}</td>
        <td>${wbCell}</td>
      </tr>`;
    })
    .join("");

  const ruleCn = RULE_LABEL_CN[rep.rule?.toLowerCase() ?? ""] ?? rep.rule;

  return {
    title: `全员 AI 复盘与三家实战对照 · ${escHtml(playerName(names, pov))} 视角`,
    body: `<div class="modal-scroll">
      <div class="report-meta">
        <span>对局规则 <b>${escHtml(ruleCn)}</b></span>
        <span>总局数 <b>${rep.rounds_total} 局</b></span>
        <span>全员均分 <b>${rep.overall.rating.toFixed(1)}</b></span>
        <span>平均一致率 <b>${pct(rep.overall.overall_acc)}</b></span>
      </div>

      <div class="report-sec">
        <h4>全员 AI 复盘评分（点击任意行切换主视角）</h4>
        <table class="report-table">
          <thead>
            <tr>
              <th>玩家 (相对方位)</th>
              <th class="num">终局顺位 / 点数</th>
              <th class="num">AI 评分</th>
              <th class="num">AI 一致率</th>
              <th class="num">弃牌一致</th>
              <th class="num">鸣牌一致</th>
              <th class="num">恶手</th>
            </tr>
          </thead>
          <tbody>${aiRows}</tbody>
        </table>
      </div>

      <div class="report-sec">
        <h4>实战攻防统计（点击最大失误直达该手）</h4>
        <table class="report-table">
          <thead>
            <tr>
              <th>玩家</th>
              <th class="num">和牌 (自摸/荣和)</th>
              <th class="num">放铳</th>
              <th class="num">立直</th>
              <th class="num">副露</th>
              <th class="num">平均打点</th>
              <th>最大失误 / 分歧</th>
            </tr>
          </thead>
          <tbody>${playRows}</tbody>
        </table>
      </div>
    </div>`,
  };
}

function renderImportModal(state: AppState): { title: string; body: string } {
  const statusTxt = state.backendOnline
    ? `<span class="pos">本地 V6 GPU 后端在线 (${escHtml(state.backendInfo?.device ?? "CUDA")})</span>`
    : `<span class="neg">离线模式（仅可加载已导出的 review_*.json；启动 review_server.py 后可在线拉取天凤牌谱与实时推演）</span>`;
  return {
    title: "导入牌谱 / 实时 V6 评测",
    body: `<div class="import-box">
      <div class="import-status">${statusTxt}</div>
      <div class="import-section">
        <label>1. 天凤牌谱链接（例如 https://tenhou.net/0/?log=...）</label>
        <div class="import-row">
          <input id="importUrlInput" type="text" placeholder="https://tenhou.net/0/?log=..." />
          <select id="importRuleSelect">
            <option value="7d">凤桌 7段 (Tenhou 7d)</option>
            <option value="10d">凤桌 10段 (Tenhou 10d)</option>
            <option value="majsoul">雀魂王座 (Majsoul Throne)</option>
            <option value="mleague">M-League</option>
          </select>
          <button class="btn primary" id="importUrlBtn" ${state.backendOnline ? "" : "disabled"}>评测并加载</button>
        </div>
      </div>
      <div class="import-section">
        <label>2. 上传本地牌谱 (.mjlog / .json / .jsonl) 或已导出的 review_*.json</label>
        <div class="import-row">
          <label class="btn primary ${state.backendOnline ? "" : "disabled"}">
            选择原始牌谱评测
            <input id="importRawReplayInput" type="file" accept=".mjlog,.xml,.json,.jsonl,.txt" style="display:none" ${state.backendOnline ? "" : "disabled"} />
          </label>
          <label class="btn">
            打开 review_*.json
            <input id="importReviewJsonInput" type="file" accept=".json" style="display:none" />
          </label>
        </div>
      </div>
      <div id="importProgressMsg" class="import-progress"></div>
    </div>`,
  };
}

function renderOptionsModal(state: AppState): { title: string; body: string } {
  const allOpp = [0, 1, 2, 3].filter((p) => p !== state.povSeat).every((p) => state.revealSeats[p]);
  const isLight = state.theme === "light";
  return {
    title: "显示选项",
    body: `<div class="options-grid">
      <button class="opt-btn" id="optToggleTheme">
        <span class="opt-info"><span class="opt-t">界面配色主题</span><span class="opt-s">和纸浅色 / 茶室深色</span></span>
        <span class="opt-pill on">${isLight ? "和纸浅色" : "茶室深色"}</span>
      </button>
      <button class="opt-btn" id="optToggleHands">
        <span class="opt-info"><span class="opt-t">三家手牌透视 (H)</span><span class="opt-s">掀开三家暗手牌对比读牌</span></span>
        <span class="opt-pill ${allOpp ? "on" : "off"}">${allOpp ? "开启" : "关闭"}</span>
      </button>
      <button class="opt-btn" id="optToggleAI">
        <span class="opt-info"><span class="opt-t">牌桌上 AI 评估柱 (M)</span><span class="opt-s">在手牌上方显示推荐度柱</span></span>
        <span class="opt-pill ${state.showAI ? "on" : "off"}">${state.showAI ? "开启" : "关闭"}</span>
      </button>
      <button class="opt-btn" id="optToggleDanger">
        <span class="opt-info"><span class="opt-t">三家放铳危险条 (D)</span><span class="opt-s">左中右对应 上家 / 对家 / 下家</span></span>
        <span class="opt-pill ${state.showDanger ? "on" : "off"}">${state.showDanger ? "开启" : "关闭"}</span>
      </button>
      <div class="opt-thresh">
        <span>恶手跳转阈值</span>
        <div class="thresh-pills">
          ${[0.05, 0.15, 0.25, 1.0]
            .map((v) => `<button class="btn${Math.abs(state.errorThreshold - v) < 0.01 ? " active" : ""}" data-set-thresh="${v}">${v >= 1.0 ? "全部分歧" : `< ${(v * 100).toFixed(0)}%`}</button>`)
            .join("")}
        </div>
      </div>
    </div>`,
  };
}

function renderHelpModal(): { title: string; body: string } {
  return {
    title: "操作指南与快捷键",
    body: `<div class="help-grid">
      <div class="help-card">
        <h4>鼠标交互</h4>
        <ul>
          <li><b>左键点击手牌</b>：复盘模式中直接进入「推演」分叉；何切模式中提交作答</li>
          <li><b>右键点击桌面 / 手牌</b>：快速 <b>摸切</b>（吃碰杠询问时自动 <b>跳过</b>）</li>
          <li><b>点击任意牌河中的牌</b>：直接跳转到该家切出那张牌的事件帧</li>
          <li><b>鼠标滚轮 上 / 下</b>：后退 / 前进逐帧事件（<code>Shift + 滚轮</code> 按自家决策步跳转）</li>
          <li><b>点击对手手牌</b>：单独开关该家的明牌透视</li>
          <li><b>点击 AI 柱区 / 危险条</b>：切换隐藏 AI 柱 / 打开 34 牌放铳明细</li>
          <li><b>点击桌面中央局名</b>：打开小局筛选与全半庄总览</li>
        </ul>
      </div>
      <div class="help-card">
        <h4>键盘快捷键</h4>
        <ul>
          <li><code>←</code> / <code>→</code>：上一帧 / 下一帧</li>
          <li><code>↑</code> / <code>↓</code>：上一自家决策 / 下一自家决策</li>
          <li><code>,</code> / <code>.</code> 或 <code>PgUp</code> / <code>PgDn</code>：上一恶手 / 下一恶手</li>
          <li><code>[</code> / <code>]</code> 或 <code>Home</code> / <code>End</code>：上一局 / 下一局</li>
          <li><code>1</code> ~ <code>4</code>：切换观察视角 1 ~ 4</li>
          <li><code>H</code>：三家手牌透视开关</li>
          <li><code>M</code>：AI 评估柱开关</li>
          <li><code>D</code>：三家放铳危险条开关</li>
          <li><code>G</code> / <code>Z</code> / <code>R</code>：全员报告表 / 34 牌明细 / 小局总览</li>
          <li><code>Esc</code>：关闭弹窗，或退出「推演」回到原牌谱</li>
        </ul>
      </div>
    </div>`,
  };
}

export function renderModal(state: AppState, activeFrame: ReviewFrame | undefined): void {
  const host = document.getElementById("modalHost")!;
  if (!state.activeModal) {
    host.className = "modal-backdrop hidden";
    host.innerHTML = "";
    return;
  }

  let content: { title: string; body: string } = { title: "", body: "" };
  if (state.activeModal === "round-overview" && state.report) {
    content = renderRoundOverviewModal(state);
  } else if (state.activeModal === "this-round" && state.report && activeFrame) {
    content = renderThisRoundModal(activeFrame, state);
  } else if (state.activeModal === "danger" && state.report && activeFrame) {
    content = renderDangerModal(activeFrame, state);
  } else if (state.activeModal === "report" && state.report) {
    content = renderReportModal(state);
  } else if (state.activeModal === "import") {
    content = renderImportModal(state);
  } else if (state.activeModal === "options") {
    content = renderOptionsModal(state);
  } else if (state.activeModal === "help") {
    content = renderHelpModal();
  }

  const isWide =
    state.activeModal === "report" ||
    state.activeModal === "round-overview" ||
    state.activeModal === "danger";

  host.className = "modal-backdrop";
  host.innerHTML = `<div class="modal-dialog${isWide ? " modal-wide" : ""}" role="dialog" aria-modal="true">
    <div class="modal-head">
      <h3>${escHtml(content.title)}</h3>
      <button class="modal-close" id="modalCloseBtn">×</button>
    </div>
    <div class="modal-body">${content.body}</div>
  </div>`;
}

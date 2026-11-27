import { toTile34 } from "./shanten";
import { escHtml, playerName, renderMeldGroup, renderTileImg, seatWindCn } from "./tiles";
import type { ActionCandidate, AppState, ReviewFrame, RiverTileTuple } from "./types";

const SLOT_W = 42;
const DRAW_GAP = 14;
const AI_SVG_H = 46;
const OPP_DISPLAY = [
  { idx: 2, short: "上", color: "#a57fc4" },
  { idx: 1, short: "对", color: "#5fa3de" },
  { idx: 0, short: "下", color: "#e5655b" },
] as const;

function isSpecialAction(a: number): boolean {
  return (a >= 37 && a <= 70) || (a >= 72 && a <= 85);
}

export function buildVisibleCounts34(f: ReviewFrame, povSeat: number): Int8Array {
  const vis = new Int8Array(34);
  const add = (tid: number) => {
    const t = toTile34(tid);
    if (t >= 0 && t < 34) vis[t]++;
  };
  for (const t of f.hands[povSeat] ?? []) add(t);
  for (const d of f.dora ?? []) add(d);
  for (let p = 0; p < 4; p++) {
    for (const r of f.rivers[p] ?? []) add(r[0]);
    for (const m of f.melds[p] ?? []) {
      for (const t of m.tiles) add(t);
    }
  }
  return vis;
}

function getHeroHandSlots(
  f: ReviewFrame,
  povSeat: number
): { slots: Array<{ tile: number; actionId: number; isDraw: boolean }>; legalSet: Set<number> } {
  const rawHand = [...(f.hands[povSeat] ?? [])];
  let drawnTile = -1;
  if (f.seat === povSeat && f.last_draw >= 0 && f.event_kind !== "round_end") {
    const pos = rawHand.indexOf(f.last_draw);
    if (pos >= 0) {
      rawHand.splice(pos, 1);
      drawnTile = f.last_draw;
    }
  }
  const legalSet = new Set<number>(
    f.seat === povSeat && f.event_kind !== "round_end" ? f.legal_actions ?? [] : []
  );

  const slots = rawHand.map((t) => ({
    tile: t,
    actionId: legalSet.has(t) ? t : -1,
    isDraw: false,
  }));
  if (drawnTile >= 0) {
    const drawAct = legalSet.has(71) ? 71 : legalSet.has(drawnTile) ? drawnTile : -1;
    slots.push({ tile: drawnTile, actionId: drawAct, isDraw: true });
  }
  return { slots, legalSet };
}

function renderRiverTiles(
  riverList: RiverTileTuple[],
  seat: number,
  f: ReviewFrame,
  pulseOrigin: number | null
): string {
  const list = riverList ?? [];
  if (list.length === 0) return "";
  const ld = f.last_discard;
  const rows: Array<{ start: number; items: RiverTileTuple[] }> = [
    { start: 0, items: list.slice(0, 6) },
    { start: 6, items: list.slice(6, 12) },
    { start: 12, items: list.slice(12) },
  ];

  return rows
    .filter((r) => r.items.length > 0)
    .map((r) => {
      const tilesHtml = r.items
        .map((tup, offset) => {
          const idx = r.start + offset;
          const [tid, isRiichi, isCalled, isTsumo] = tup;
          const origin = tup.length >= 5 ? tup[4] : -1;
          const isLast = ld && ld.seat === seat && ld.index === idx;
          const isPulse = pulseOrigin !== null && origin === pulseOrigin;
          const originAttr = origin !== -1 ? `data-river-origin="${origin}"` : "";

          if (isRiichi) {
            const boxCls = [
              "river-tile",
              "river-riichi-box",
              isCalled ? "called-dim" : "",
              isTsumo ? "tsumo-dim" : "",
              isLast ? "is-last-discard" : "",
              isPulse ? "is-pulse" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return `<span class="${boxCls}" ${originAttr}>${renderTileImg(tid, "xs", "river-riichi-inner")}</span>`;
          }

          const cls = [
            "river-tile",
            isCalled ? "called-dim" : "",
            isTsumo ? "tsumo-dim" : "",
            isLast ? "is-last-discard" : "",
            isPulse ? "is-pulse" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return renderTileImg(tid, "xs", cls, originAttr);
        })
        .join("");
      return `<div class="pond-row">${tilesHtml}</div>`;
    })
    .join("");
}

function actionBtnKind(a: number): string {
  if (a === 72) return "k-riichi";
  if (a === 73 || a === 74) return "k-win";
  if (a === 84) return "k-pass";
  return "";
}

function actionPriority(a: number): number {
  if (a === 73 || a === 74) return 0;
  if (a === 72) return 1;
  if ((a >= 37 && a <= 70) || a === 77) return 2;
  if (a === 75 || a === 76) return 3;
  if (a >= 78 && a <= 83) return 4;
  if (a === 85) return 5;
  if (a === 84) return 99;
  return 50;
}

function actionFallbackLabel(a: number): string {
  if (a === 84) return "跳过";
  if (a === 72) return "立直";
  if (a === 73) return "自摸";
  if (a === 74) return "荣和";
  if (a === 85) return "九种";
  if (a === 75 || a === 76) return "碰";
  if (a === 77) return "明杠";
  if (a >= 78 && a <= 83) return "吃";
  if (a >= 37 && a <= 70) return "杠";
  return `#${a}`;
}

function renderActionButtonInner(c: ActionCandidate, f: ReviewFrame): string {
  if (c.a === 84) return `<span class="act-verb">跳过</span>`;
  if (c.a === 73) return `<span class="act-verb">自摸</span>`;
  if (c.a === 74) {
    return `<span class="act-verb">荣和</span>${f.target >= 0 ? renderTileImg(f.target, "micro") : ""}`;
  }
  if (c.a === 72) return `<span class="act-verb">立直</span><span class="act-riichi-stick"></span>`;
  if (c.a === 85) return `<span class="act-verb">九种</span>`;
  if (c.a >= 37 && c.a <= 70) {
    const verb = c.label.includes("加杠") ? "加杠" : "暗杠";
    return `<span class="act-verb">${verb}</span>${renderTileImg(c.a - 37, "micro")}`;
  }
  if (c.a === 77) {
    return `<span class="act-verb">明杠</span>${f.target >= 0 ? renderTileImg(f.target, "micro") : ""}`;
  }
  if (c.a === 75 || c.a === 76) {
    return `<span class="act-verb">碰</span>${f.target >= 0 ? renderTileImg(f.target, "micro") : ""}`;
  }
  if (c.a >= 78 && c.a <= 83) {
    const m = c.label.match(/配([0-9mps]+)/);
    const pairTxt = m ? m[1] : "";
    return `<span class="act-verb">吃</span>${f.target >= 0 ? renderTileImg(f.target, "micro") : ""}${
      pairTxt ? `<span class="act-detail">配${escHtml(pairTxt)}</span>` : ""
    }`;
  }
  return `<span class="act-verb">${escHtml(c.label)}</span>`;
}

export function fitBoard(): void {
  const stage = document.getElementById("boardStage");
  const board = document.getElementById("kdBoard");
  if (!stage || !board) return;
  const availW = stage.clientWidth - 16;
  const availH = stage.clientHeight - 16;
  const scale = Math.min(1.06, availW / 764, availH / 752);
  board.style.transform = `scale(${Math.max(0.45, scale).toFixed(4)})`;
}

export function renderBoard(
  f: ReviewFrame,
  state: AppState,
  reportNames: string[],
  hideAnswer: boolean,
  userAns: number | undefined
): void {
  const $ = (id: string) => document.getElementById(id)!;
  const pov = state.povSeat;
  const sBot = pov;
  const sRight = (pov + 1) % 4;
  const sTop = (pov + 2) % 4;
  const sLeft = (pov + 3) % 4;

  // 1. Center Compass
  const edgeHtml = (absSeat: number, posCls: string) => {
    const w = seatWindCn(absSeat, f.dealer);
    const isDealer = absSeat === f.dealer;
    const isTurn = absSeat === f.seat && f.event_kind !== "round_end";
    const stick = f.riichi[absSeat] ? `<div class="riichi-stick ${posCls}"></div>` : "";
    return `<div class="gi-edge ${posCls}${isTurn ? " is-turn" : ""}">
      <span class="gi-wind${isDealer ? " is-dealer" : ""}">${w}</span>
      <span class="gi-score">${f.scores[absSeat].toLocaleString()}</span>
    </div>${stick}`;
  };

  const doraSlots = [0, 1, 2, 3, 4].map((i) => (i < f.dora.length ? f.dora[i] : -1));
  const roundResultBtn =
    f.event_kind === "round_end" && f.round_result
      ? `<button class="center-res-btn" id="openThisRoundBtn">本局结算</button>`
      : "";

  $("slotCenter").innerHTML = `
    ${edgeHtml(sTop, "top")}
    ${edgeHtml(sRight, "right")}
    ${edgeHtml(sBot, "bottom")}
    ${edgeHtml(sLeft, "left")}
    <div class="gi-core">
      <button class="info-round-btn" id="openRoundOverviewBtn">
        <span>${escHtml(f.round_label)}</span>
        <span class="caret">▾</span>
      </button>
      <div class="gi-meta">
        <span>余 <b>${f.wall_rem}</b></span>
        <span>供 <b>${f.kyotaku}</b></span>
        <span>${f.turn}巡</span>
      </div>
      <div class="gi-doras">
        ${doraSlots.map((t) => renderTileImg(t, "mini")).join("")}
      </div>
      ${roundResultBtn}
    </div>
  `;

  // 2. Opponent Hands, Bottom-Aligned Melds & Clean Name Badges
  const renderOpp = (
    absSeat: number,
    relIdx: number,
    slotId: string,
    orient: "top" | "left" | "right"
  ) => {
    const rev = state.revealSeats[absSeat] || f.event_kind === "round_end";
    const showTenpai =
      state.mode !== "wwyd" &&
      !hideAnswer &&
      state.showAI &&
      f.has_ai !== false &&
      f.seat === pov;
    const tp = showTenpai ? (f.p_tenpai?.[relIdx] ?? 0) : 0;
    const pName = escHtml(playerName(reportNames, absSeat));
    const wind = seatWindCn(absSeat, f.dealer);

    const rawHand = [...(f.hands[absSeat] ?? [])];
    let oppDrawn = -1;
    if (f.seat === absSeat && f.last_draw >= 0 && f.event_kind !== "round_end") {
      const idx = rawHand.indexOf(f.last_draw);
      if (idx >= 0) {
        rawHand.splice(idx, 1);
        oppDrawn = f.last_draw;
      }
    }
    const closedHtml =
      rawHand.map((t) => renderTileImg(rev ? t : -1, "sm")).join("") +
      (oppDrawn >= 0 ? renderTileImg(rev ? oppDrawn : -1, "sm", "opp-drawn-gap") : "");
    const meldsHtml = (f.melds[absSeat] ?? []).map((m) => renderMeldGroup(m, absSeat, "xs")).join("");
    const activeCls = f.seat === absSeat && f.event_kind !== "round_end" ? " is-acting" : "";

    $(slotId).innerHTML = `
      <div class="opp-rot-wrap opp-${orient}">
        <div class="opp-hand-strip${activeCls}" data-toggle-seat="${absSeat}">
          <div class="opp-closed">${closedHtml}</div>
          ${meldsHtml ? `<div class="opp-melds">${meldsHtml}</div>` : ""}
        </div>
      </div>
      <div class="opp-label opp-label-${orient}" data-toggle-seat="${absSeat}">
        <span class="ol-w">${wind}</span>
        <span class="ol-n">${pName}</span>
        ${tp >= 0.35 ? `<span class="ol-tp">听${Math.round(tp * 100)}%</span>` : ""}
      </div>
    `;
  };

  renderOpp(sTop, 1, "slotOppTop", "top");
  renderOpp(sLeft, 2, "slotOppLeft", "left");
  renderOpp(sRight, 0, "slotOppRight", "right");

  // 3. Four Symmetric Rivers with dedicated Riichi space & 3-row layout
  $("slotPondBottom").innerHTML = `<div class="pond-grid">${renderRiverTiles(f.rivers[sBot], sBot, f, state.pulseRiverOrigin)}</div>`;
  $("slotPondTop").innerHTML = `<div class="pond-rot rot-top"><div class="pond-grid">${renderRiverTiles(f.rivers[sTop], sTop, f, state.pulseRiverOrigin)}</div></div>`;
  $("slotPondLeft").innerHTML = `<div class="pond-rot rot-left"><div class="pond-grid">${renderRiverTiles(f.rivers[sLeft], sLeft, f, state.pulseRiverOrigin)}</div></div>`;
  $("slotPondRight").innerHTML = `<div class="pond-rot rot-right"><div class="pond-grid">${renderRiverTiles(f.rivers[sRight], sRight, f, state.pulseRiverOrigin)}</div></div>`;

  // 4. Sculpted Floating Call / Riichi / Win / Pass Action Cards with Weight Visualization
  const isPovTurn = f.seat === pov && f.event_kind !== "round_end";
  const legalSet = new Set<number>(isPovTurn ? f.legal_actions ?? [] : []);
  const actEl = $("slotActionBar");
  const probMap: Record<number, number> = {};
  for (const c of f.candidates ?? []) probMap[c.a] = c.prob;

  const specialActionsToShow: ActionCandidate[] = [];
  if (isPovTurn) {
    const seen = new Set<number>();
    for (const c of f.candidates ?? []) {
      if (isSpecialAction(c.a) && (legalSet.size === 0 || legalSet.has(c.a))) {
        specialActionsToShow.push(c);
        seen.add(c.a);
      }
    }
    for (const la of f.legal_actions ?? []) {
      if (isSpecialAction(la) && !seen.has(la)) {
        specialActionsToShow.push({
          a: la,
          label: actionFallbackLabel(la),
          prob: probMap[la] ?? 0,
          du_ron: 0,
        });
      }
    }
    specialActionsToShow.sort((a, b) => {
      const pa = actionPriority(a.a);
      const pb = actionPriority(b.a);
      if (pa !== pb) return pa - pb;
      return b.prob - a.prob;
    });
  }

  if (!isPovTurn || specialActionsToShow.length === 0) {
    actEl.className = "action-bar empty";
    actEl.innerHTML = "";
  } else {
    actEl.className = "action-bar";
    const showHints = !hideAnswer && state.showAI && f.has_ai !== false;
    const btns = specialActionsToShow
      .map((c) => {
        const isBest = showHints && c.a === f.best_action;
        const isHuman = !hideAnswer && state.mode !== "sandbox" && f.human_action >= 0 && c.a === f.human_action;
        const isUser = userAns !== undefined && userAns === c.a;
        let hl = "";
        let toneCls = "tone-muted";
        let barCol = "#8a7f70";

        if (isBest) {
          hl = "hl-ai";
          toneCls = "tone-ai";
          barCol = "#2e5c8a";
        } else if (isHuman) {
          hl = f.is_blunder ? "hl-blunder" : "hl-human";
          toneCls = f.is_blunder ? "tone-bad" : "tone-human";
          barCol = f.is_blunder ? "#c03e35" : "#b57b1c";
        } else if (isUser) {
          hl = "hl-user";
          toneCls = "tone-user";
          barCol = "#4a3f33";
        }
        if (c.a === 73 || c.a === 74) barCol = "#fff3dc";

        let tagHtml = "";
        if (isBest) tagHtml = `<span class="act-tag ai">AI</span>`;
        else if (isHuman) tagHtml = `<span class="act-tag ${f.is_blunder ? "bad" : "hu"}">实战</span>`;
        else if (isUser) tagHtml = `<span class="act-tag usr">你</span>`;

        const clickable = legalSet.has(c.a);
        const pctVal = Math.max(0, Math.min(100, c.prob * 100));
        const pctText = c.prob > 0 && c.prob < 0.095 ? `${pctVal.toFixed(1)}%` : `${Math.round(pctVal)}%`;
        const meterW = c.prob > 0.002 ? Math.max(3, pctVal) : 0;

        return `<button class="kd-act-btn ${actionBtnKind(c.a)} ${toneCls} ${hl}${clickable ? " is-clickable" : ""}" ${clickable ? `data-special-action="${c.a}"` : ""}>
          ${showHints ? `<span class="act-weight-fill" style="width:${pctVal.toFixed(1)}%"></span>` : ""}
          <span class="act-content">${renderActionButtonInner(c, f)}${showHints ? `<span class="act-pct">${pctText}</span>${tagHtml}` : ""}</span>
          ${showHints ? `<span class="act-meter"><span class="act-meter-bar" style="width:${meterW.toFixed(1)}%;background:${barCol}"></span></span>` : ""}
        </button>`;
      })
      .join("");
    actEl.innerHTML = btns;
  }

  // 5. Fixed Left-Anchored Hero Hand + AI SVG Bars + Flush Bottom-Aligned Melds + Vertical 3-Bar Danger Strip
  const { slots } = getHeroHandSlots(f, pov);
  const hasDraw = slots.some((s) => s.isDraw);
  const svgWidth = Math.max(SLOT_W, slots.length * SLOT_W + (hasDraw ? DRAW_GAP : 0));
  const maxP = Math.max(f.best_prob || 0.01, 0.01);
  const hasAiOnThisFrame = isPovTurn && f.has_ai !== false && (f.candidates?.length ?? 0) > 0;

  let aiSvgInner = "";
  if (hideAnswer && isPovTurn) {
    aiSvgInner = `<rect x="1" y="8" width="${svgWidth - 2}" height="${AI_SVG_H - 12}" rx="7" fill="rgba(16, 30, 22, 0.52)" stroke="rgba(244, 239, 228, 0.24)" stroke-dasharray="4 3"/>
      <text x="${svgWidth / 2}" y="29" text-anchor="middle" fill="#efe9dc" font-size="11.5" font-weight="600">何切思考中 · 左键切牌 / 右键摸切</text>`;
  } else if (!state.showAI) {
    aiSvgInner = `<rect x="1" y="10" width="${svgWidth - 2}" height="${AI_SVG_H - 14}" rx="7" fill="rgba(16, 30, 22, 0.36)" stroke="rgba(244, 239, 228, 0.16)" stroke-dasharray="3 3"/>
      <text x="${svgWidth / 2}" y="30" text-anchor="middle" fill="rgba(244, 239, 228, 0.72)" font-size="11">AI 评估已隐藏 · 点击或按 M 显示</text>`;
  } else if (hasAiOnThisFrame) {
    let closedCount = 0;
    for (const s of slots) if (!s.isDraw) closedCount++;

    slots.forEach((sl, idx) => {
      const cx = (sl.isDraw ? closedCount * SLOT_W + DRAW_GAP : idx * SLOT_W) + SLOT_W / 2;
      let p = probMap[sl.actionId];
      if (p === undefined && sl.isDraw) p = probMap[sl.tile] ?? probMap[71];
      if (p === undefined) p = probMap[sl.tile] ?? 0;

      const isBest =
        f.best_action === sl.actionId ||
        (sl.isDraw && f.best_action === 71) ||
        (!hasDraw && f.best_action === sl.tile);
      const isHuman =
        state.mode !== "sandbox" &&
        f.human_action >= 0 &&
        (f.human_action === sl.actionId ||
          (sl.isDraw && f.human_action === 71) ||
          (!sl.isDraw && f.human_action === sl.tile));
      const isUser =
        userAns !== undefined &&
        (userAns === sl.actionId || (sl.isDraw && userAns === 71));

      let barFill = "rgba(244, 239, 228, 0.44)";
      let textFill = "rgba(244, 239, 228, 0.80)";
      if (isBest) {
        barFill = "#7bc4e2";
        textFill = "#cbe9f7";
      } else if (isUser) {
        barFill = "#f6f0e4";
        textFill = "#ffffff";
      } else if (isHuman) {
        barFill = f.is_blunder ? "#e56b5d" : "#e0b453";
        textFill = f.is_blunder ? "#ffb2a8" : "#f6d887";
      }

      const barH = p > 0.004 ? Math.max(3, Math.round((p / maxP) * 26)) : 0;
      const pctStr = p >= 0.02 ? `${Math.round(p * 100)}%` : "";
      const yTop = AI_SVG_H - 3 - barH;
      if (barH > 0) {
        aiSvgInner += `<rect x="${cx - 7}" y="${yTop}" width="14" height="${barH}" rx="2.5" fill="${barFill}"/>`;
      }
      if (pctStr) {
        aiSvgInner += `<text x="${cx}" y="${Math.max(11, yTop - 3)}" text-anchor="middle" fill="${textFill}" font-size="9.5" font-weight="700">${pctStr}</text>`;
      }
    });
  }

  let dangerStripHtml = "";
  if (state.showDanger && !(state.mode === "wwyd" && hideAnswer)) {
    const hasDealData = isPovTurn && f.p_deal && Object.keys(f.p_deal).length > 0;
    const anyThreat =
      f.riichi.some((r, idx) => idx !== pov && r) ||
      (f.p_tenpai ?? []).some((x) => x >= 0.15);

    const legendHtml = `<div class="danger-legend">
      ${OPP_DISPLAY.map(
        (o) => `<span class="dl-item"><i style="background:${o.color}"></i>${o.short}</span>`
      ).join("")}
    </div>`;

    const colsHtml = slots
      .map((sl) => {
        const cls = `danger-slot${sl.isDraw ? " is-draw" : ""}`;
        if (!hasDealData) {
          return `<div class="${cls} is-empty"></div>`;
        }
        const t34 = toTile34(sl.tile);
        const vec = f.p_deal?.[String(t34)] ?? [0, 0, 0];
        const maxD = Math.max(vec[0], vec[1], vec[2]);

        const bars = OPP_DISPLAY.map((opp) => {
          const p = vec[opp.idx] ?? 0;
          const hPct = p >= 0.003 ? Math.min(100, Math.max(12, Math.round((p / 0.18) * 100))) : 0;
          return `<div class="dv-track"><div class="dv-fill" style="height:${hPct}%;background:${opp.color}"></div></div>`;
        }).join("");

        let readout = `<span class="ds-pct"></span>`;
        if (maxD >= 0.01) {
          const level = maxD >= 0.12 ? " hot" : maxD >= 0.045 ? " warn" : "";
          readout = `<span class="ds-pct${level}">${Math.round(maxD * 100)}%</span>`;
        } else if (anyThreat && maxD < 0.002) {
          readout = `<span class="ds-pct safe">安</span>`;
        }

        return `<div class="${cls}"><div class="dv-bars">${bars}</div>${readout}</div>`;
      })
      .join("");

    dangerStripHtml = `<div class="danger-strip-wrap" id="dangerBarsStrip">${legendHtml}<div class="danger-slots-row">${colsHtml}</div></div>`;
  }

  const tilesHtml = slots
    .map((sl) => {
      const isHuman =
        !hideAnswer &&
        state.mode !== "sandbox" &&
        isPovTurn &&
        f.human_action >= 0 &&
        (f.human_action === sl.actionId ||
          (sl.isDraw && f.human_action === 71) ||
          (!sl.isDraw && f.human_action === sl.tile));
      const isUser =
        userAns !== undefined &&
        (userAns === sl.actionId || (sl.isDraw && userAns === 71));
      const isHlRow =
        state.highlightAction !== null &&
        (state.highlightAction === sl.actionId ||
          (sl.isDraw && state.highlightAction === 71) ||
          (!sl.isDraw && state.highlightAction === sl.tile));

      let tileCls = "";
      if (!hideAnswer && isPovTurn) {
        if (isUser) tileCls = "mark-user";
        else if (isHuman) tileCls = f.is_blunder ? "mark-blunder" : "mark-human";
      }
      if (isHlRow) tileCls += " hl-hover-row";

      const isLegalDiscard = isPovTurn && sl.actionId >= 0;
      const isDimmedIllegal = isPovTurn && legalSet.size > 0 && sl.actionId < 0;
      const slotCls = [
        "ego-slot",
        sl.isDraw ? "is-draw" : "",
        isLegalDiscard ? "is-clickable" : "",
        isDimmedIllegal ? "is-illegal" : "",
      ]
        .filter(Boolean)
        .join(" ");

      const actionAttr = isLegalDiscard ? `data-hand-action="${sl.actionId}"` : "";

      return `<div class="${slotCls}" ${actionAttr}>
        ${renderTileImg(sl.tile, "lg", tileCls)}
      </div>`;
    })
    .join("");

  const egoMeldsHtml = (f.melds[pov] ?? []).map((m) => renderMeldGroup(m, pov, "sm")).join("");
  $("slotEgoArea").innerHTML = `
    <div class="ego-hand-stack">
      <div class="ego-ai-row">
        <svg class="kd-ai-svg" id="aiBarsSvg" width="${svgWidth}" height="${AI_SVG_H}" viewBox="0 0 ${svgWidth} ${AI_SVG_H}">${aiSvgInner}</svg>
      </div>
      <div class="ego-tiles-line">
        <div class="ego-tiles-row" id="egoTilesRow">${tilesHtml}</div>
        ${egoMeldsHtml ? `<div class="ego-melds-col">${egoMeldsHtml}</div>` : ""}
      </div>
      ${dangerStripHtml}
    </div>
  `;
}

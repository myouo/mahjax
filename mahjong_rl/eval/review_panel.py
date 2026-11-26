"""KillerDucky / mjai-reviewer / NAGA-style interactive HTML review panel for Sovereign-Mahjong V6.

Generates a single self-contained HTML file (zero external CDN dependencies) with:
- Interactive 4-player Mahjong table (scores, winds, dora indicators, riichi sticks,
  6-column rivers with sideways riichi tiles & dimmed tsumogiri, open melds, optional X-Ray)
- NAGA-style vertical probability bars rendered directly above each tile in hand
- Mortal / KillerDucky-style Top-K horizontal probability breakdown + Deal-in risk (Delta_U_ron)
- Sovereign-Mahjong V6 live opponent tenpai probability gauges (Shimocha / Toimen / Kamicha)
- Round selector, seat switcher, All / Divergences / Blunders filter, clickable step timeline,
  and full keyboard navigation (Left/Right, Up/Down for blunders, 1-4 for seats).
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict


def render_naga_panel_html(report: Dict[str, Any]) -> str:
    """Render a full interactive KillerDucky / mjai-reviewer / NAGA-style HTML review panel."""
    # Safely embed report JSON inside <script type="application/json"> (escape </script>)
    report_json = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    title_src = html.escape(str(report.get("replay_source", "Mahjong Replay")))
    rule_str = html.escape(str(report.get("rule", "7d")))
    pol_src = html.escape(str(report.get("policy_source", "")))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sovereign-Mahjong V6 — NAGA / KillerDucky 牌谱复盘面板</title>
<style>
  :root {{
    --bg: #0a0f1c;
    --panel: #111827;
    --panel2: #1e293b;
    --border: #2b3952;
    --felt: #0d2b22;
    --felt-edge: #154234;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --emerald: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --tile-bg: #fbf8ee;
    --tile-border: #cbd5e1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    min-height: 100vh; display: flex; flex-direction: column;
  }}

  /* Top toolbar */
  .topbar {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border);
  }}
  .brand {{ font-weight: 800; font-size: 15px; color: var(--cyan); letter-spacing: .3px; margin-right: 6px; }}
  .badge {{ font-size: 12px; padding: 3px 8px; border-radius: 6px; background: var(--panel2); color: var(--muted); border: 1px solid var(--border); }}
  .seat-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .seat-btn {{
    cursor: pointer; border: 1px solid var(--border); background: var(--panel2); color: var(--text);
    padding: 5px 10px; border-radius: 8px; font-size: 12px; transition: .15s;
    display: flex; align-items: center; gap: 6px;
  }}
  .seat-btn:hover {{ border-color: var(--cyan); }}
  .seat-btn.active {{ background: #0c4a6e; border-color: var(--cyan); color: #fff; }}
  .seat-btn .rt {{ color: var(--cyan); font-weight: 700; }}
  .controls-right {{ margin-left: auto; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  select, .btn {{
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    border-radius: 7px; padding: 5px 10px; font-size: 12px; cursor: pointer;
  }}
  .btn:hover {{ border-color: var(--cyan); }}
  .btn.active {{ background: #1e3a5f; border-color: var(--cyan); color: #fff; }}

  /* Main workspace */
  .main {{
    display: grid; grid-template-columns: minmax(640px, 1.55fr) minmax(350px, 1fr);
    gap: 12px; padding: 12px 16px; flex: 1;
  }}
  @media (max-width: 1100px) {{
    .main {{ grid-template-columns: 1fr; }}
  }}

  /* Mahjong Table */
  .table-wrap {{
    background: radial-gradient(circle at center, #123c2f 0%, var(--felt) 70%, #091d17 100%);
    border: 2px solid var(--felt-edge); border-radius: 14px;
    padding: 12px; display: flex; flex-direction: column; justify-content: space-between;
    min-height: 590px; position: relative; box-shadow: inset 0 0 40px rgba(0,0,0,0.45);
  }}
  .seat-top {{ display: flex; flex-direction: column; align-items: center; gap: 6px; }}
  .table-mid {{
    display: grid; grid-template-columns: 1fr 220px 1fr; align-items: center; gap: 8px; margin: 6px 0;
  }}
  .seat-left {{ display: flex; flex-direction: column; align-items: flex-start; gap: 6px; }}
  .seat-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }}
  .seat-bottom {{ display: flex; flex-direction: column; align-items: center; gap: 8px; margin-top: 6px; }}

  .player-tag {{
    font-size: 12px; background: rgba(15, 23, 42, 0.78); padding: 3px 9px; border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.25); display: inline-flex; align-items: center; gap: 6px;
  }}
  .player-tag .wind {{ color: var(--amber); font-weight: 700; }}
  .player-tag .sc {{ color: #f8fafc; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .player-tag .riichi-pill {{ background: #991b1b; color: #fecaca; font-size: 10px; padding: 1px 5px; border-radius: 4px; font-weight: 700; }}
  .player-tag .tenpai-pill {{ background: #78350f; color: #fde68a; font-size: 10px; padding: 1px 5px; border-radius: 4px; font-weight: 700; }}

  /* Center compass */
  .compass {{
    background: rgba(9, 15, 28, 0.88); border: 1px solid #334155; border-radius: 10px;
    padding: 10px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 6px;
  }}
  .round-title {{ font-size: 15px; font-weight: 800; color: #f8fafc; }}
  .wall-meta {{ font-size: 12px; color: var(--muted); display: flex; gap: 10px; }}
  .dora-row {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--amber); }}

  /* Tiles */
  .tile {{
    display: inline-flex; flex-direction: column; align-items: center; justify-content: center;
    background: linear-gradient(180deg, #ffffff 0%, var(--tile-bg) 85%, #e7e1ce 100%);
    border: 1px solid #94a3b8; border-bottom: 3px solid #64748b; border-radius: 5px;
    font-weight: 800; user-select: none; position: relative; line-height: 1.05;
    box-shadow: 0 2px 4px rgba(0,0,0,0.35);
  }}
  .tile.lg {{ width: 38px; height: 52px; font-size: 16px; }}
  .tile.sm {{ width: 26px; height: 35px; font-size: 11px; border-bottom-width: 2px; border-radius: 4px; }}
  .tile.xs {{ width: 22px; height: 30px; font-size: 10px; border-bottom-width: 2px; border-radius: 3px; }}
  .tile.back {{
    background: linear-gradient(180deg, #1d4ed8 0%, #1e3a8a 100%);
    border-color: #60a5fa; border-bottom-color: #172554; color: transparent;
  }}
  .tile .suit-m {{ color: #b91c1c; }}
  .tile .suit-p {{ color: #1d4ed8; }}
  .tile .suit-s {{ color: #15803d; }}
  .tile .suit-z {{ color: #1e293b; }}
  .tile .suit-z-g {{ color: #15803d; }}
  .tile .suit-z-r {{ color: #dc2626; }}
  .tile .suit-z-w {{ color: #475569; border: 1.5px solid #94a3b8; padding: 1px 3px; border-radius: 2px; font-size: 11px; }}
  .tile .sub {{ font-size: 9px; opacity: 0.75; margin-top: 1px; }}
  .tile.red {{ border-color: #ef4444; box-shadow: 0 0 0 1px rgba(239,68,68,0.5), 0 2px 4px rgba(0,0,0,0.35); }}
  .tile.red .num {{ color: #dc2626 !important; }}
  .tile.tsumo-dim {{ opacity: 0.72; }}
  .tile.called-dim {{ opacity: 0.42; border-style: dashed; }}
  .tile.riichi-rot {{ transform: rotate(90deg); margin: 0 5px; border-color: #f59e0b; }}

  /* Rivers */
  .river {{
    display: grid; grid-template-columns: repeat(6, auto); gap: 2px;
    background: rgba(0,0,0,0.22); padding: 4px; border-radius: 6px; min-height: 40px; min-width: 146px;
  }}
  .melds-row {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }}
  .meld-grp {{ display: inline-flex; gap: 1px; background: rgba(0,0,0,0.25); padding: 2px 3px; border-radius: 4px; }}
  .opp-hand {{ display: flex; gap: 2px; flex-wrap: wrap; justify-content: center; }}

  /* NAGA Hand with Probability Bars */
  .ego-hand-wrap {{
    display: flex; align-items: flex-end; gap: 3px; padding: 6px 10px;
    background: rgba(9, 15, 28, 0.55); border-radius: 10px; border: 1px solid rgba(148,163,184,0.2);
    flex-wrap: wrap; justify-content: center;
  }}
  .hand-slot {{ display: flex; flex-direction: column; align-items: center; width: 40px; }}
  .hand-slot.draw-gap {{ margin-left: 12px; }}
  .naga-bar-box {{
    height: 62px; width: 22px; display: flex; flex-direction: column; justify-content: flex-end; align-items: center;
    margin-bottom: 3px; position: relative;
  }}
  .naga-pct {{ font-size: 10px; font-weight: 700; margin-bottom: 2px; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .naga-bar {{ width: 16px; border-radius: 3px 3px 0 0; min-height: 2px; transition: height .18s; }}
  .naga-bar.ai {{ background: linear-gradient(180deg, #38bdf8, #0284c7); }}
  .naga-bar.both {{ background: linear-gradient(180deg, #34d399, #059669); }}
  .naga-bar.human {{ background: linear-gradient(180deg, #fbbf24, #d97706); }}
  .naga-bar.blunder {{ background: linear-gradient(180deg, #f87171, #dc2626); }}
  .naga-bar.other {{ background: #475569; }}
  .tile.hl-both {{ outline: 2.5px solid var(--emerald); outline-offset: 1px; }}
  .tile.hl-ai {{ outline: 2.5px solid var(--cyan); outline-offset: 1px; }}
  .tile.hl-human {{ outline: 2.5px solid var(--amber); outline-offset: 1px; }}
  .tile.hl-blunder {{ outline: 2.5px solid var(--red); outline-offset: 1px; }}
  .slot-tag {{ font-size: 9px; font-weight: 700; margin-top: 3px; padding: 1px 4px; border-radius: 3px; min-height: 14px; }}
  .slot-tag.both {{ background: #065f46; color: #a7f3d0; }}
  .slot-tag.ai {{ background: #075985; color: #bae6fd; }}
  .slot-tag.human {{ background: #78350f; color: #fde68a; }}
  .slot-tag.blunder {{ background: #7f1d1d; color: #fecaca; }}

  /* Call decision banner */
  .call-banner {{
    display: flex; align-items: center; gap: 10px; padding: 6px 12px;
    background: rgba(15, 23, 42, 0.9); border: 1px solid var(--amber); border-radius: 8px;
    font-size: 13px;
  }}

  /* Right sidebar */
  .sidebar {{ display: flex; flex-direction: column; gap: 10px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px;
  }}
  .card h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; display: flex; justify-content: space-between; align-items: center; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
  .stat-box {{ background: var(--panel2); border-radius: 8px; padding: 8px; text-align: center; }}
  .stat-box .val {{ font-size: 18px; font-weight: 800; color: #f8fafc; font-variant-numeric: tabular-nums; }}
  .stat-box .val.cyan {{ color: var(--cyan); }}
  .stat-box .val.red {{ color: var(--red); }}
  .stat-box .lbl {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}

  /* Action prob rows */
  .cand-row {{ display: grid; grid-template-columns: 115px 1fr 54px; align-items: center; gap: 8px; margin: 6px 0; font-size: 13px; }}
  .cand-bar-bg {{ background: #0f172a; height: 14px; border-radius: 7px; overflow: hidden; border: 1px solid #334155; }}
  .cand-bar-fill {{ height: 100%; border-radius: 7px; transition: width .18s; }}
  .cand-label {{ display: flex; align-items: center; gap: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .mini-tag {{ font-size: 10px; padding: 1px 4px; border-radius: 4px; font-weight: 700; }}
  .mini-tag.ai {{ background: #075985; color: #bae6fd; }}
  .mini-tag.hu {{ background: #78350f; color: #fde68a; }}
  .mini-tag.ok {{ background: #065f46; color: #a7f3d0; }}
  .mini-tag.bad {{ background: #7f1d1d; color: #fecaca; }}

  /* Belief tenpai bars */
  .tenpai-row {{ display: grid; grid-template-columns: 60px 1fr 44px; align-items: center; gap: 8px; margin: 5px 0; font-size: 12px; }}
  .tenpai-bg {{ background: #0f172a; height: 10px; border-radius: 5px; overflow: hidden; }}
  .tenpai-fill {{ height: 100%; background: linear-gradient(90deg, #f59e0b, #ef4444); }}

  /* Blunder quick-jump list */
  .blunder-list {{ max-height: 210px; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; }}
  .blunder-item {{
    display: flex; justify-content: space-between; align-items: center; gap: 6px;
    padding: 6px 8px; background: var(--panel2); border-radius: 7px; cursor: pointer;
    border: 1px solid transparent; font-size: 12px;
  }}
  .blunder-item:hover {{ border-color: var(--cyan); }}
  .blunder-item.active {{ border-color: var(--cyan); background: #0c4a6e; }}
  .blunder-item.is-bad {{ border-left: 3px solid var(--red); }}

  /* Bottom navigation & timeline */
  .nav-bar {{
    background: var(--panel); border-top: 1px solid var(--border); padding: 10px 16px;
    display: flex; flex-direction: column; gap: 8px;
  }}
  .nav-controls {{ display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap; }}
  .step-counter {{ font-size: 13px; font-weight: 700; min-width: 120px; text-align: center; font-variant-numeric: tabular-nums; }}
  .timeline {{
    display: flex; gap: 3px; flex-wrap: wrap; justify-content: center; max-height: 44px; overflow-y: auto;
  }}
  .dot {{
    width: 11px; height: 11px; border-radius: 3px; cursor: pointer; opacity: 0.85; transition: .1s;
  }}
  .dot:hover {{ transform: scale(1.3); opacity: 1; }}
  .dot.ok {{ background: #10b981; }}
  .dot.diff {{ background: #f59e0b; }}
  .dot.bad {{ background: #ef4444; }}
  .dot.cur {{ outline: 2px solid #fff; transform: scale(1.35); opacity: 1; }}
  .kbd-hint {{ text-align: center; font-size: 11px; color: var(--muted); }}
</style>
</head>
<body>

<div class="topbar">
  <span class="brand">Sovereign-Mahjong V6</span>
  <span class="badge">规则: {rule_str}</span>
  <span class="badge" title="{pol_src}">牌谱: {title_src}</span>
  <div class="seat-tabs" id="seatTabs"></div>
  <div class="controls-right">
    <select id="roundSelect" title="跳转局次"></select>
    <select id="filterSelect" title="过滤决策步">
      <option value="all">显示: 全部决策</option>
      <option value="diff">仅看: 分歧手</option>
      <option value="blunder">仅看: 恶手</option>
    </select>
    <button class="btn" id="xrayBtn" title="透视四家手牌">👁 透视: 关</button>
  </div>
</div>

<div class="main">
  <!-- Left: 4-Player Table -->
  <div class="table-wrap">
    <div class="seat-top" id="seatTop"></div>

    <div class="table-mid">
      <div class="seat-left" id="seatLeft"></div>
      <div class="compass" id="compass"></div>
      <div class="seat-right" id="seatRight"></div>
    </div>

    <div class="seat-bottom" id="seatBottom"></div>
  </div>

  <!-- Right: KillerDucky / Mortal Analysis Sidebar -->
  <div class="sidebar">
    <div class="card">
      <h3><span id="playerSummaryTitle">玩家统计</span><span id="stepVerdict"></span></h3>
      <div class="stats-grid" id="statsGrid"></div>
    </div>

    <div class="card">
      <h3><span>AI 候选分布 (Top-6)</span><span id="stepMeta" style="color:#38bdf8"></span></h3>
      <div id="candList"></div>
      <div id="riskLine" style="margin-top:8px;font-size:12px;color:#94a3b8;"></div>
    </div>

    <div class="card">
      <h3><span>V6 Belief 对手听牌概率</span><span style="font-size:11px;color:#94a3b8">B_EMA</span></h3>
      <div id="tenpaiList"></div>
    </div>

    <div class="card">
      <h3><span>关键分歧 / 恶手快跳</span><span id="blunderCountTag"></span></h3>
      <div class="blunder-list" id="blunderList"></div>
    </div>
  </div>
</div>

<div class="nav-bar">
  <div class="nav-controls">
    <button class="btn" id="prevRoundBtn">⏮ 上一局</button>
    <button class="btn" id="prevBlunderBtn">⏪ 上一恶手 (↑)</button>
    <button class="btn" id="prevStepBtn">◀ 上一手 (←)</button>
    <span class="step-counter" id="stepCounter">0 / 0</span>
    <button class="btn" id="nextStepBtn">下一手 (→) ▶</button>
    <button class="btn" id="nextBlunderBtn">下一恶手 (↓) ⏩</button>
    <button class="btn" id="nextRoundBtn">下一局 ⏭</button>
  </div>
  <div class="timeline" id="timeline"></div>
  <div class="kbd-hint">快捷键：←/→ 切换步数 · ↑/↓ 跳转恶手 · 1~4 切换座位视角 · X 开关透视</div>
</div>

<script id="reportData" type="application/json">{report_json}</script>
<script>
const REPORT = JSON.parse(document.getElementById("reportData").textContent);
const ALL_FRAMES = REPORT.frames || [];
const SEATS = REPORT.seats || {{}};
const WIND_CN = ["东", "南", "西", "北"];
const HONOR_CN = ["东", "南", "西", "北", "白", "发", "中"];

let state = {{
  seat: parseInt(Object.keys(SEATS)[0] || "0", 10),
  filter: "all",
  xray: false,
  cursor: 0,
}};

function tileHtml(tid, sizeCls = "sm", extraCls = "") {{
  if (tid < 0) return `<span class="tile ${{sizeCls}} back ${{extraCls}}">&nbsp;</span>`;
  let isRed = (tid === 34 || tid === 35 || tid === 36);
  let t34 = tid === 34 ? 4 : (tid === 35 ? 13 : (tid === 36 ? 22 : tid));
  let cls = `tile ${{sizeCls}} ${{isRed ? "red" : ""}} ${{extraCls}}`;
  if (t34 < 27) {{
    const sIdx = Math.floor(t34 / 9);
    const rank = (t34 % 9) + 1;
    const suitChar = ["萬", "筒", "索"][sIdx];
    const suitCls = ["suit-m", "suit-p", "suit-s"][sIdx];
    const numDisp = isRed ? "5" : String(rank);
    return `<span class="${{cls}}"><span class="num ${{suitCls}}">${{numDisp}}</span><span class="sub ${{suitCls}}">${{suitChar}}</span></span>`;
  }} else {{
    const h = t34 - 27;
    const ch = HONOR_CN[h] || "?";
    let hCls = "suit-z";
    if (h === 4) hCls = "suit-z-w";
    else if (h === 5) hCls = "suit-z-g";
    else if (h === 6) hCls = "suit-z-r";
    return `<span class="${{cls}}"><span class="${{hCls}}">${{ch}}</span></span>`;
  }}
}}

function seatWind(absSeat, dealer) {{
  return WIND_CN[(absSeat - dealer + 4) % 4];
}}

function getFilteredFrames() {{
  return ALL_FRAMES.filter(f => {{
    if (f.seat !== state.seat) return false;
    if (state.filter === "diff") return !f.matched_top1;
    if (state.filter === "blunder") return f.is_blunder;
    return true;
  }});
}}

function renderSeatTabs() {{
  const el = document.getElementById("seatTabs");
  el.innerHTML = "";
  Object.keys(SEATS).forEach(k => {{
    const s = SEATS[k];
    const btn = document.createElement("button");
    btn.className = "seat-btn" + (parseInt(k, 10) === state.seat ? " active" : "");
    btn.innerHTML = `<span>P${{k}} ${{s.name}}</span><span class="rt">${{s.rating.toFixed(1)}}</span><span style="color:#94a3b8">${{(s.overall_acc*100).toFixed(0)}}%</span>`;
    btn.onclick = () => {{ state.seat = parseInt(k, 10); state.cursor = 0; renderAll(); }};
    el.appendChild(btn);
  }});
}}

function renderRoundSelect(frames) {{
  const sel = document.getElementById("roundSelect");
  const seen = new Map();
  frames.forEach((f, idx) => {{
    if (!seen.has(f.round_idx)) seen.set(f.round_idx, {{ label: f.round_label, firstIdx: idx }});
  }});
  sel.innerHTML = "";
  seen.forEach((info, rIdx) => {{
    const opt = document.createElement("option");
    opt.value = info.firstIdx;
    opt.textContent = info.label;
    if (frames[state.cursor] && frames[state.cursor].round_idx === rIdx) opt.selected = true;
    sel.appendChild(opt);
  }});
  sel.onchange = () => {{ state.cursor = parseInt(sel.value, 10) || 0; renderAll(); }};
}}

function renderRiver(riverList) {{
  if (!riverList || !riverList.length) return `<div class="river"></div>`;
  return `<div class="river">${{riverList.map(r => {{
    const cls = [r[1] ? "riichi-rot" : "", r[2] ? "called-dim" : "", r[3] ? "tsumo-dim" : ""].filter(Boolean).join(" ");
    return tileHtml(r[0], "xs", cls);
  }}).join("")}}</div>`;
}}

function renderMelds(meldsList) {{
  if (!meldsList || !meldsList.length) return "";
  return `<div class="melds-row">${{meldsList.map(m =>
    `<div class="meld-grp">${{m.tiles.map(t => tileHtml(t, "xs")).join("")}}</div>`
  ).join("")}}</div>`;
}}

function renderOppSeat(containerId, f, absSeat, relIdx) {{
  const el = document.getElementById(containerId);
  const sInfo = SEATS[String(absSeat)] || {{ name: "Seat " + absSeat }};
  const wind = seatWind(absSeat, f.dealer);
  const sc = f.scores[absSeat];
  const isRiichi = f.riichi[absSeat];
  const tp = f.p_tenpai ? f.p_tenpai[relIdx] : 0;
  const hand = f.hands[absSeat] || [];
  const handHtml = `<div class="opp-hand">${{hand.map(t => tileHtml(state.xray ? t : -1, "xs")).join("")}}</div>`;
  const tagHtml = `<div class="player-tag">
    <span class="wind">${{wind}}</span>
    <span>P${{absSeat}} ${{sInfo.name}}</span>
    <span class="sc">${{sc.toLocaleString()}}</span>
    ${{isRiichi ? `<span class="riichi-pill">立直</span>` : ""}}
    ${{tp >= 0.35 ? `<span class="tenpai-pill">听牌 ${{(tp*100).toFixed(0)}}%</span>` : ""}}
  </div>`;
  el.innerHTML = `${{tagHtml}}${{handHtml}}${{renderMelds(f.melds[absSeat])}}${{renderRiver(f.rivers[absSeat])}}`;
}}

function renderEgoHand(f) {{
  const el = document.getElementById("seatBottom");
  const absSeat = f.seat;
  const sInfo = SEATS[String(absSeat)] || {{ name: "Seat " + absSeat }};
  const wind = seatWind(absSeat, f.dealer);
  const sc = f.scores[absSeat];
  const isRiichi = f.riichi[absSeat];

  // Map action id -> probability
  const probMap = {{}};
  (f.candidates || []).forEach(c => {{ probMap[c.a] = c.prob; }});

  // Build hand list separating last_draw if present
  const rawHand = [...(f.hands[absSeat] || [])];
  let drawnTile = -1;
  if (f.last_draw >= 0) {{
    const pos = rawHand.indexOf(f.last_draw);
    if (pos >= 0) {{
      rawHand.splice(pos, 1);
      drawnTile = f.last_draw;
    }}
  }}

  const slots = rawHand.map(t => ({{ tile: t, actionId: t, isDraw: false }}));
  if (drawnTile >= 0) {{
    // For drawn tile, probability could be under TSUMOGIRI (71) or its tile id
    const drawAct = (probMap[71] !== undefined) ? 71 : drawnTile;
    slots.push({{ tile: drawnTile, actionId: drawAct, isDraw: true }});
  }}

  const maxP = Math.max(f.best_prob || 0.01, 0.01);

  const slotsHtml = slots.map(sl => {{
    let p = probMap[sl.actionId];
    if (p === undefined && sl.isDraw) p = probMap[sl.tile];
    if (p === undefined) p = 0;

    const isBest = (f.best_action === sl.actionId) || (sl.isDraw && f.best_action === 71);
    const isHuman = (f.human_action === sl.actionId) || (sl.isDraw && f.human_action === 71);

    let barCls = "other", tileHl = "", tagCls = "", tagTxt = "";
    if (isBest && isHuman) {{ barCls = "both"; tileHl = "hl-both"; tagCls = "both"; tagTxt = "一致✓"; }}
    else if (isBest) {{ barCls = "ai"; tileHl = "hl-ai"; tagCls = "ai"; tagTxt = "AI"; }}
    else if (isHuman) {{
      barCls = f.is_blunder ? "blunder" : "human";
      tileHl = f.is_blunder ? "hl-blunder" : "hl-human";
      tagCls = f.is_blunder ? "blunder" : "human";
      tagTxt = f.is_blunder ? "恶手" : "实战";
    }}

    const hPx = p > 0.004 ? Math.max(3, Math.round((p / maxP) * 44)) : 0;
    const pctTxt = p >= 0.01 ? `${{Math.round(p * 100)}}%` : (p > 0.001 ? "<1%" : "");
    const pctColor = barCls === "both" ? "#34d399" : (barCls === "ai" ? "#38bdf8" : (barCls === "blunder" ? "#f87171" : (barCls === "human" ? "#fbbf24" : "#94a3b8")));

    return `<div class="hand-slot ${{sl.isDraw ? "draw-gap" : ""}}">
      <div class="naga-bar-box">
        <div class="naga-pct" style="color:${{pctColor}}">${{pctTxt}}</div>
        ${{hPx > 0 ? `<div class="naga-bar ${{barCls}}" style="height:${{hPx}}px"></div>` : ""}}
      </div>
      ${{tileHtml(sl.tile, "lg", tileHl)}}
      <div class="slot-tag ${{tagCls}}">${{tagTxt}}</div>
    </div>`;
  }}).join("");

  let callBanner = "";
  if (!f.is_discard || f.best_action >= 72 || f.human_action >= 72) {{
    const tgtHtml = f.target >= 0 ? `<span>对象牌:</span>${{tileHtml(f.target, "sm")}}` : "";
    callBanner = `<div class="call-banner">
      ${{tgtHtml}}
      <span>实战: <b style="color:${{f.matched_top1 ? "#34d399" : (f.is_blunder ? "#f87171" : "#fbbf24")}}">${{f.human_action_str}} (${{(f.human_prob*100).toFixed(1)}}%)</b></span>
      <span>|</span>
      <span>AI推荐: <b style="color:#38bdf8">${{f.best_action_str}} (${{(f.best_prob*100).toFixed(1)}}%)</b></span>
    </div>`;
  }}

  const tagHtml = `<div class="player-tag">
    <span class="wind">${{wind}}</span>
    <span>P${{absSeat}} ${{sInfo.name}} (当前视角)</span>
    <span class="sc">${{sc.toLocaleString()}}</span>
    ${{isRiichi ? `<span class="riichi-pill">立直</span>` : ""}}
  </div>`;

  el.innerHTML = `${{renderRiver(f.rivers[absSeat])}}${{callBanner}}<div class="ego-hand-wrap">${{slotsHtml}}</div>${{renderMelds(f.melds[absSeat])}}${{tagHtml}}`;
}}

function renderCompass(f) {{
  const el = document.getElementById("compass");
  const doras = (f.dora || []).map(t => tileHtml(t, "xs")).join("");
  el.innerHTML = `
    <div class="round-title">${{f.round_label}}</div>
    <div class="wall-meta"><span>余牌 <b>${{f.wall_rem}}</b></span><span>供托 <b>${{f.kyotaku}}</b></span><span>第 <b>${{f.turn}}</b> 巡</span></div>
    <div class="dora-row"><span>宝牌指示:</span>${{doras}}</div>
  `;
}}

function renderSidebar(f, frames) {{
  const s = SEATS[String(state.seat)];
  document.getElementById("playerSummaryTitle").textContent = `P${{state.seat}} ${{s.name}} 总览`;
  document.getElementById("statsGrid").innerHTML = `
    <div class="stat-box"><div class="val cyan">${{s.rating.toFixed(2)}}</div><div class="lbl">Mortal Rating</div></div>
    <div class="stat-box"><div class="val">${{(s.overall_acc*100).toFixed(1)}}%</div><div class="lbl">总一致率</div></div>
    <div class="stat-box"><div class="val">${{(s.discard_acc*100).toFixed(1)}}%</div><div class="lbl">纯弃牌一致</div></div>
    <div class="stat-box"><div class="val">${{(s.call_acc*100).toFixed(1)}}%</div><div class="lbl">鸣牌/立直一致</div></div>
    <div class="stat-box"><div class="val">${{(s.top3_acc*100).toFixed(1)}}%</div><div class="lbl">Top-3 一致</div></div>
    <div class="stat-box"><div class="val red">${{s.blunder_count}} (${{(s.blunder_rate*100).toFixed(1)}}%)</div><div class="lbl">恶手数</div></div>
  `;

  const verd = document.getElementById("stepVerdict");
  if (f.matched_top1) verd.innerHTML = `<span class="mini-tag ok">一致 ✓ (r=100%)</span>`;
  else if (f.is_blunder) verd.innerHTML = `<span class="mini-tag bad">恶手 (r=${{(f.prob_ratio*100).toFixed(1)}}%)</span>`;
  else verd.innerHTML = `<span class="mini-tag hu">分歧 (r=${{(f.prob_ratio*100).toFixed(1)}}%)</span>`;

  document.getElementById("stepMeta").textContent = `${{f.round_label}} · 第${{f.turn}}巡`;

  const cands = (f.candidates || []).slice(0, 6);
  document.getElementById("candList").innerHTML = cands.map(c => {{
    const isAi = c.a === f.best_action;
    const isHu = c.a === f.human_action;
    let color = "#475569", tags = "";
    if (isAi && isHu) {{ color = "#10b981"; tags = `<span class="mini-tag ok">一致</span>`; }}
    else if (isAi) {{ color = "#38bdf8"; tags = `<span class="mini-tag ai">AI</span>`; }}
    else if (isHu) {{ color = f.is_blunder ? "#ef4444" : "#f59e0b"; tags = `<span class="mini-tag ${{f.is_blunder ? "bad" : "hu"}}">实战</span>`; }}
    return `<div class="cand-row">
      <div class="cand-label">${{tags}}<span>${{c.label}}</span></div>
      <div class="cand-bar-bg"><div class="cand-bar-fill" style="width:${{Math.max(1, c.prob*100)}}%;background:${{color}}"></div></div>
      <div style="text-align:right;font-variant-numeric:tabular-nums">${{(c.prob*100).toFixed(1)}}%</div>
    </div>`;
  }}).join("");

  document.getElementById("riskLine").innerHTML =
    `放铳期望 ΔU_ron — 实战: <b>${{f.delta_u_ron_human.toFixed(4)}}</b> | AI推荐: <b>${{f.delta_u_ron_best.toFixed(4)}}</b>`;

  const relNames = ["下家", "对家", "上家"];
  const tp = f.p_tenpai || [0, 0, 0];
  document.getElementById("tenpaiList").innerHTML = [0, 1, 2].map(i => {{
    const p = tp[i] || 0;
    const oppSeat = (f.seat + i + 1) % 4;
    return `<div class="tenpai-row">
      <span>${{relNames[i]}} P${{oppSeat}}</span>
      <div class="tenpai-bg"><div class="tenpai-fill" style="width:${{(p*100).toFixed(1)}}%"></div></div>
      <span style="text-align:right;font-variant-numeric:tabular-nums">${{(p*100).toFixed(0)}}%</span>
    </div>`;
  }}).join("");

  // Blunder / top divergence quick-jump list
  const seatFrames = ALL_FRAMES.filter(x => x.seat === state.seat);
  const ranked = seatFrames
    .map((x, idx) => ({{ frame: x, seatIdx: idx, gap: x.best_prob - x.human_prob }}))
    .filter(x => !x.frame.matched_top1)
    .sort((a, b) => b.gap - a.gap)
    .slice(0, 15);
  document.getElementById("blunderCountTag").textContent = `Top ${{ranked.length}}`;
  document.getElementById("blunderList").innerHTML = ranked.map(item => {{
    const x = item.frame;
    const isCur = (x.idx === f.idx);
    return `<div class="blunder-item ${{x.is_blunder ? "is-bad" : ""}} ${{isCur ? "active" : ""}}" onclick="jumpToFrameIdx(${{x.idx}})">
      <span>${{x.round_label}} 第${{x.turn}}巡</span>
      <span>${{x.human_action_str}} → <b style="color:#38bdf8">${{x.best_action_str}}</b></span>
      <span style="color:${{x.is_blunder ? "#f87171" : "#fbbf24"}}">${{(x.human_prob*100).toFixed(0)}}%</span>
    </div>`;
  }}).join("");
}}

function renderTimeline(frames) {{
  const el = document.getElementById("timeline");
  const cur = frames[state.cursor];
  document.getElementById("stepCounter").textContent = `${{state.cursor + 1}} / ${{frames.length}}`;
  el.innerHTML = frames.map((f, i) => {{
    const cls = f.matched_top1 ? "ok" : (f.is_blunder ? "bad" : "diff");
    return `<div class="dot ${{cls}} ${{i === state.cursor ? "cur" : ""}}" title="${{f.round_label}} 第${{f.turn}}巡: ${{f.human_action_str}} vs ${{f.best_action_str}}" onclick="setCursor(${{i}})"></div>`;
  }}).join("");
}}

window.setCursor = function(i) {{
  const frames = getFilteredFrames();
  state.cursor = Math.max(0, Math.min(i, frames.length - 1));
  renderAll();
}};

window.jumpToFrameIdx = function(globalIdx) {{
  state.filter = "all";
  document.getElementById("filterSelect").value = "all";
  const frames = getFilteredFrames();
  const pos = frames.findIndex(f => f.idx === globalIdx);
  if (pos >= 0) {{ state.cursor = pos; renderAll(); }}
}};

function jumpBlunder(dir) {{
  const frames = getFilteredFrames();
  for (let i = state.cursor + dir; i >= 0 && i < frames.length; i += dir) {{
    if (frames[i].is_blunder || !frames[i].matched_top1) {{
      if (frames[i].is_blunder) {{ state.cursor = i; renderAll(); return; }}
    }}
  }}
  // Fallback to next divergence if no more blunders
  for (let i = state.cursor + dir; i >= 0 && i < frames.length; i += dir) {{
    if (!frames[i].matched_top1) {{ state.cursor = i; renderAll(); return; }}
  }}
}}

function jumpRound(dir) {{
  const frames = getFilteredFrames();
  if (!frames.length) return;
  const curR = frames[state.cursor].round_idx;
  for (let i = state.cursor + dir; i >= 0 && i < frames.length; i += dir) {{
    if (frames[i].round_idx !== curR) {{ state.cursor = i; renderAll(); return; }}
  }}
}}

function renderAll() {{
  renderSeatTabs();
  const frames = getFilteredFrames();
  renderRoundSelect(frames);
  if (!frames.length) {{
    document.getElementById("stepCounter").textContent = "0 / 0";
    document.getElementById("seatBottom").innerHTML = `<div style="padding:30px;color:#94a3b8">当前筛选条件下没有决策步</div>`;
    return;
  }}
  if (state.cursor >= frames.length) state.cursor = frames.length - 1;
  const f = frames[state.cursor];
  renderCompass(f);
  renderOppSeat("seatRight", f, (f.seat + 1) % 4, 0);
  renderOppSeat("seatTop",   f, (f.seat + 2) % 4, 1);
  renderOppSeat("seatLeft",  f, (f.seat + 3) % 4, 2);
  renderEgoHand(f);
  renderSidebar(f, frames);
  renderTimeline(frames);
}}

document.getElementById("filterSelect").onchange = e => {{ state.filter = e.target.value; state.cursor = 0; renderAll(); }};
document.getElementById("xrayBtn").onclick = () => {{
  state.xray = !state.xray;
  const b = document.getElementById("xrayBtn");
  b.textContent = state.xray ? "👁 透视: 开" : "👁 透视: 关";
  b.classList.toggle("active", state.xray);
  renderAll();
}};
document.getElementById("prevStepBtn").onclick = () => setCursor(state.cursor - 1);
document.getElementById("nextStepBtn").onclick = () => setCursor(state.cursor + 1);
document.getElementById("prevBlunderBtn").onclick = () => jumpBlunder(-1);
document.getElementById("nextBlunderBtn").onclick = () => jumpBlunder(1);
document.getElementById("prevRoundBtn").onclick = () => jumpRound(-1);
document.getElementById("nextRoundBtn").onclick = () => jumpRound(1);

document.addEventListener("keydown", e => {{
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") setCursor(state.cursor - 1);
  else if (e.key === "ArrowRight" || e.key === " ") {{ e.preventDefault(); setCursor(state.cursor + 1); }}
  else if (e.key === "ArrowUp") {{ e.preventDefault(); jumpBlunder(-1); }}
  else if (e.key === "ArrowDown") {{ e.preventDefault(); jumpBlunder(1); }}
  else if (e.key === "x" || e.key === "X") document.getElementById("xrayBtn").click();
  else if (["1","2","3","4"].includes(e.key)) {{
    const s = parseInt(e.key, 10) - 1;
    if (SEATS[String(s)]) {{ state.seat = s; state.cursor = 0; renderAll(); }}
  }}
}});

renderAll();
</script>
</body>
</html>"""

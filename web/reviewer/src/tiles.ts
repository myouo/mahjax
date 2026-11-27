import type { MeldInfo } from "./types";

const WIND_CN = ["东", "南", "西", "北"] as const;
const HONOR_CN = ["东", "南", "西", "北", "白", "发", "中"] as const;
const DORA_NEXT = [
  1, 2, 3, 4, 5, 6, 7, 8, 0,
  10, 11, 12, 13, 14, 15, 16, 17, 9,
  19, 20, 21, 22, 23, 24, 25, 26, 18,
  28, 29, 30, 27,
  32, 33, 31,
];

export function escHtml(s: string): string {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function tileSvgFilename(tid: number): string {
  if (tid < 0) return "back.svg";
  if (tid === 34) return "0m.svg";
  if (tid === 35) return "0p.svg";
  if (tid === 36) return "0s.svg";
  if (tid < 9) return `${tid + 1}m.svg`;
  if (tid < 18) return `${tid - 8}p.svg`;
  if (tid < 27) return `${tid - 17}s.svg`;
  if (tid < 34) return `${tid - 26}z.svg`;
  return "back.svg";
}

export function tileLabelCn(tid: number): string {
  if (tid < 0) return "牌背";
  if (tid === 34) return "0m(赤5m)";
  if (tid === 35) return "0p(赤5p)";
  if (tid === 36) return "0s(赤5s)";
  if (tid < 9) return `${tid + 1}m`;
  if (tid < 18) return `${tid - 8}p`;
  if (tid < 27) return `${tid - 17}s`;
  if (tid < 34) return HONOR_CN[tid - 27] ?? "?";
  return "?";
}

export function tileShortLabel(tid: number): string {
  if (tid < 0) return "?";
  if (tid === 34) return "0m";
  if (tid === 35) return "0p";
  if (tid === 36) return "0s";
  if (tid < 9) return `${tid + 1}m`;
  if (tid < 18) return `${tid - 8}p`;
  if (tid < 27) return `${tid - 17}s`;
  if (tid < 34) return HONOR_CN[tid - 27] ?? "?";
  return "?";
}

export function seatWindCn(absSeat: number, dealer: number): string {
  return WIND_CN[(absSeat - dealer + 4) % 4];
}

export function playerName(names: string[] | undefined, seat: number): string {
  const raw = names?.[seat]?.trim();
  return raw ? raw : `玩家${seat + 1}`;
}

export function doraOfIndicator(indTid: number): number {
  const t34 = indTid === 34 ? 4 : indTid === 35 ? 13 : indTid === 36 ? 22 : indTid;
  if (t34 < 0 || t34 >= 34) return -1;
  return DORA_NEXT[t34];
}

export function renderTileImg(
  tid: number,
  size: "lg" | "md" | "sm" | "xs" | "mini" | "micro" = "sm",
  extraCls = "",
  attrs = ""
): string {
  const src = `./tiles/${tileSvgFilename(tid)}`;
  const alt = tileShortLabel(tid);
  const backCls = tid < 0 ? " is-back" : "";
  const cls = `mj-tile mj-${size}${backCls}${extraCls ? " " + extraCls : ""}`;
  return `<span class="${cls}" ${attrs}><img src="${src}" alt="${alt}" draggable="false" /></span>`;
}

export interface MeldDisplayItem {
  tid: number;
  rotated: boolean;
  kakanTop?: number;
}

/** Order meld tiles authentically according to which player (`from`) the tile was claimed from. */
export function formatMeldOrder(meld: MeldInfo, ownerSeat: number): MeldDisplayItem[] {
  const tiles = [...(meld.tiles ?? [])];
  if (tiles.length === 0) return [];

  if (meld.kind === "kan_closed" && tiles.length === 4) {
    return [
      { tid: -1, rotated: false },
      { tid: tiles[1], rotated: false },
      { tid: tiles[2], rotated: false },
      { tid: -1, rotated: false },
    ];
  }

  const from = meld.from;
  const rel = from === null || from === undefined ? 3 : (from - ownerSeat + 4) % 4;
  // rel: 3 = kamicha (left), 2 = toimen (middle), 1 = shimocha (right)

  if (meld.kind === "chi") {
    const cIdx = meld.called ?? 0;
    const calledTile = tiles.splice(Math.max(0, Math.min(cIdx, tiles.length - 1)), 1)[0];
    return [
      { tid: calledTile, rotated: true },
      ...tiles.map((t) => ({ tid: t, rotated: false })),
    ];
  }

  if (meld.kind === "pon") {
    const cIdx = meld.called ?? tiles.length - 1;
    const calledTile = tiles.splice(Math.max(0, Math.min(cIdx, tiles.length - 1)), 1)[0];
    if (rel === 3) {
      return [
        { tid: calledTile, rotated: true },
        { tid: tiles[0], rotated: false },
        { tid: tiles[1], rotated: false },
      ];
    }
    if (rel === 2) {
      return [
        { tid: tiles[0], rotated: false },
        { tid: calledTile, rotated: true },
        { tid: tiles[1], rotated: false },
      ];
    }
    return [
      { tid: tiles[0], rotated: false },
      { tid: tiles[1], rotated: false },
      { tid: calledTile, rotated: true },
    ];
  }

  if (meld.kind === "kan_open") {
    const cIdx = meld.called ?? tiles.length - 1;
    const calledTile = tiles.splice(Math.max(0, Math.min(cIdx, tiles.length - 1)), 1)[0];
    if (rel === 3) {
      return [{ tid: calledTile, rotated: true }, ...tiles.map((t) => ({ tid: t, rotated: false }))];
    }
    if (rel === 2) {
      return [
        { tid: tiles[0], rotated: false },
        { tid: calledTile, rotated: true },
        { tid: tiles[1], rotated: false },
        { tid: tiles[2], rotated: false },
      ];
    }
    return [...tiles.map((t) => ({ tid: t, rotated: false })), { tid: calledTile, rotated: true }];
  }

  if (meld.kind === "kan_added" && tiles.length === 4) {
    const addedTile = tiles[3];
    const base = tiles.slice(0, 3);
    const sideIdx = rel === 3 ? 0 : rel === 2 ? 1 : 2;
    return base.map((t, i) => ({
      tid: t,
      rotated: i === sideIdx,
      kakanTop: i === sideIdx ? addedTile : undefined,
    }));
  }

  return tiles.map((t) => ({ tid: t, rotated: false }));
}

export function renderMeldGroup(meld: MeldInfo, ownerSeat: number, size: "xs" | "sm" = "xs"): string {
  const items = formatMeldOrder(meld, ownerSeat);
  const inner = items
    .map((item) => {
      if (item.kakanTop !== undefined) {
        return `<span class="meld-kakan-stack ms-${size}"><span class="meld-side-box ms-${size}">${renderTileImg(item.tid, size, "meld-side")}</span><span class="meld-side-box ms-${size}">${renderTileImg(item.kakanTop, size, "meld-side")}</span></span>`;
      }
      if (item.rotated) {
        return `<span class="meld-side-box ms-${size}">${renderTileImg(item.tid, size, "meld-side")}</span>`;
      }
      return renderTileImg(item.tid, size);
    })
    .join("");
  return `<div class="meld-grp">${inner}</div>`;
}

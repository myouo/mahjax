const WIND_CN = ["东", "南", "西", "北"] as const;
const HONOR_CN = ["东", "南", "西", "北", "白", "发", "中"] as const;

/** Map MahJax tile id (`0..36`, or `-1` for face-down) to SVG filename in `public/tiles/`. */
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
  if (tid === 34) return "0m";
  if (tid === 35) return "0p";
  if (tid === 36) return "0s";
  if (tid < 9) return `${tid + 1}m`;
  if (tid < 18) return `${tid - 8}p`;
  if (tid < 27) return `${tid - 17}s`;
  if (tid < 34) return HONOR_CN[tid - 27] ?? "?";
  return `?`;
}

export function seatWindCn(absSeat: number, dealer: number): string {
  return WIND_CN[(absSeat - dealer + 4) % 4];
}

export function renderTileImg(
  tid: number,
  size: "lg" | "md" | "sm" | "xs" = "sm",
  extraCls = ""
): string {
  const src = `./tiles/${tileSvgFilename(tid)}`;
  const alt = tileLabelCn(tid);
  const cls = `mj-tile mj-${size}${extraCls ? " " + extraCls : ""}`;
  return `<span class="${cls}" title="${alt}"><img src="${src}" alt="${alt}" draggable="false" /></span>`;
}

/**
 * Fast exact Riichi Mahjong Shanten & Ukeire (有效进张) calculator in pure TypeScript.
 * Works on 34-tile counts (0..8: 1m..9m, 9..17: 1p..9p, 18..26: 1s..9s, 27..33: E/S/W/N/P/F/C).
 */

export function toTile34(tid: number): number {
  if (tid === 34) return 4;
  if (tid === 35) return 13;
  if (tid === 36) return 22;
  return tid;
}

export function tilesToCounts34(tiles: number[]): Int8Array {
  const c = new Int8Array(34);
  for (const t of tiles) {
    const idx = toTile34(t);
    if (idx >= 0 && idx < 34) c[idx]++;
  }
  return c;
}

function calcNormalShanten(c: Int8Array, numMelds: number): number {
  let best = 8 - numMelds * 2;
  const maxGroups = 4 - numMelds;

  function dfs(i: number, melds: number, taatsu: number, pair: number): void {
    const rem = maxGroups - melds;
    const effTaatsu = taatsu > rem ? rem : taatsu;
    const cur = 8 - (melds + numMelds) * 2 - effTaatsu - pair;
    if (cur < best) best = cur;
    if (best <= -1) return;

    while (i < 34 && c[i] === 0) i++;
    if (i >= 34) return;

    if (c[i] >= 3 && melds < maxGroups) {
      c[i] -= 3;
      dfs(i, melds + 1, taatsu, pair);
      c[i] += 3;
    }
    if (i < 27 && i % 9 <= 6 && c[i] >= 1 && c[i + 1] >= 1 && c[i + 2] >= 1 && melds < maxGroups) {
      c[i]--;
      c[i + 1]--;
      c[i + 2]--;
      dfs(i, melds + 1, taatsu, pair);
      c[i]++;
      c[i + 1]++;
      c[i + 2]++;
    }
    if (c[i] >= 2 && pair === 0) {
      c[i] -= 2;
      dfs(i, melds, taatsu, 1);
      c[i] += 2;
    }
    if (c[i] >= 2 && melds + taatsu < maxGroups) {
      c[i] -= 2;
      dfs(i, melds, taatsu + 1, pair);
      c[i] += 2;
    }
    if (i < 27 && i % 9 <= 7 && c[i] >= 1 && c[i + 1] >= 1 && melds + taatsu < maxGroups) {
      c[i]--;
      c[i + 1]--;
      dfs(i, melds, taatsu + 1, pair);
      c[i]++;
      c[i + 1]++;
    }
    if (i < 27 && i % 9 <= 6 && c[i] >= 1 && c[i + 2] >= 1 && melds + taatsu < maxGroups) {
      c[i]--;
      c[i + 2]--;
      dfs(i, melds, taatsu + 1, pair);
      c[i]++;
      c[i + 2]++;
    }
    dfs(i + 1, melds, taatsu, pair);
  }

  dfs(0, 0, 0, 0);
  return best;
}

function calcChiitoiShanten(c: Int8Array): number {
  let pairs = 0;
  let kinds = 0;
  for (let i = 0; i < 34; i++) {
    if (c[i] >= 1) kinds++;
    if (c[i] >= 2) pairs++;
  }
  return 6 - pairs + Math.max(0, 7 - kinds);
}

const TERMINALS = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

function calcKokushiShanten(c: Int8Array): number {
  let kinds = 0;
  let hasPair = 0;
  for (const idx of TERMINALS) {
    if (c[idx] >= 1) kinds++;
    if (c[idx] >= 2) hasPair = 1;
  }
  return 13 - kinds - hasPair;
}

export function calcShanten(tiles: number[], numMelds: number): number {
  const c = tilesToCounts34(tiles);
  let best = calcNormalShanten(c, numMelds);
  if (numMelds === 0) {
    best = Math.min(best, calcChiitoiShanten(c), calcKokushiShanten(c));
  }
  return best;
}

export interface WaitDetail {
  tile34: number;
  rem: number;
}

export interface DiscardUkeire {
  shantenAfter: number;
  ukeireCount: number;
  waitTiles: number[];
  waitDetails: WaitDetail[];
}

export function calcDiscardUkeire(
  handTiles: number[],
  discardTid: number,
  numMelds: number,
  visibleCounts: Int8Array
): DiscardUkeire {
  const c = tilesToCounts34(handTiles);
  const d34 = toTile34(discardTid);
  if (d34 < 0 || d34 >= 34 || c[d34] <= 0) {
    return { shantenAfter: 8, ukeireCount: 0, waitTiles: [], waitDetails: [] };
  }
  c[d34]--;
  let baseShanten = calcNormalShanten(c, numMelds);
  if (numMelds === 0) {
    baseShanten = Math.min(baseShanten, calcChiitoiShanten(c), calcKokushiShanten(c));
  }

  let total = 0;
  const waits: number[] = [];
  const waitDetails: WaitDetail[] = [];
  for (let t = 0; t < 34; t++) {
    const rem = Math.max(0, 4 - visibleCounts[t]);
    if (c[t] >= 4) continue;
    c[t]++;
    let s = calcNormalShanten(c, numMelds);
    if (numMelds === 0) {
      s = Math.min(s, calcChiitoiShanten(c), calcKokushiShanten(c));
    }
    c[t]--;
    if (s < baseShanten) {
      total += rem;
      waits.push(t);
      waitDetails.push({ tile34: t, rem });
    }
  }
  return { shantenAfter: baseShanten, ukeireCount: total, waitTiles: waits, waitDetails };
}

export function calcTenpaiWaits(
  handTiles: number[],
  numMelds: number,
  visibleCounts: Int8Array
): { shanten: number; waitDetails: WaitDetail[]; totalRem: number } {
  const c = tilesToCounts34(handTiles);
  let baseShanten = calcNormalShanten(c, numMelds);
  if (numMelds === 0) {
    baseShanten = Math.min(baseShanten, calcChiitoiShanten(c), calcKokushiShanten(c));
  }
  const waitDetails: WaitDetail[] = [];
  let totalRem = 0;
  for (let t = 0; t < 34; t++) {
    if (c[t] >= 4) continue;
    c[t]++;
    let s = calcNormalShanten(c, numMelds);
    if (numMelds === 0) {
      s = Math.min(s, calcChiitoiShanten(c), calcKokushiShanten(c));
    }
    c[t]--;
    if (s < baseShanten) {
      const rem = Math.max(0, 4 - visibleCounts[t]);
      totalRem += rem;
      waitDetails.push({ tile34: t, rem });
    }
  }
  return { shanten: baseShanten, waitDetails, totalRem };
}

export function formatShantenCn(s: number): string {
  if (s <= -1) return "和牌";
  if (s === 0) return "听牌";
  return `${s}向听`;
}

"""Sovereign-Mahjong V6 Replay Reviewer (雀魂 / 天凤 / MJAI 牌谱一致率与 Rating 评测).

Supports:
- Native MJAI `.jsonl` / `.json` logs (from Mortal, mjai-reviewer, Majsoul-to-MJAI, MahJax)
- Tenhou JSON (`.json`, `tenhou.net/6` & Majsoul-exported Tenhou JSON format)
- Tenhou XML (`.mjlog` / `<mjloggm>`) and direct Tenhou log URLs (`tenhou.net/0/?log=...`)

Computes per-seat and game-wide:
- Overall Top-1 / Top-2 / Top-3 agreement rate (一致率)
- Discard-only agreement rate & Call/Riichi/Pass agreement rate
- Mortal-style Rating (0-100): `100 * mean(p(a_human) / p(a_best))` + Geometric Rating
- Blunder count / rate (`p_human < 0.05` when `p_best >= 0.50`) + Top-K blunder breakdown
  with AI Top-3 suggestions, `Delta_U_ron` deal-in risk, and opponent `p_tenpai`
- Clean terminal summary, JSON report, and standalone HTML visual review page.

CLI usage:
    python -m mahjong_rl.eval.replay_reviewer \\
        --replay path/to/game.jsonl \\
        --ckpt-dir checkpoints/sovereign_v6 \\
        --rule 7d \\
        --seat 0 \\
        --out-html review.html
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np

import mahjax
from mahjax.common.action import Action
from mahjong_rl.algo.legal_ron import compute_analytical_trinity, extract_table_snapshot
from mahjong_rl.algo.sovereign_v6 import NEG_INF, masked_categorical_probs
from mahjong_rl.algo.utility import (
    DAN_UTILITY_MAP,
    W_PHOENIX_7D,
    build_opponent_utility_features,
    make_homogeneous_table,
    rotate_utility_matrix_ego,
    vnm_normalize_utility,
)
from mahjong_rl.config import BeliefConfig, GRPConfig, ModelConfig
from mahjong_rl.data.mjai_parquet_converter import (
    _init_round_state_from_deck,
    parse_mjai_game_to_rounds,
)
from mahjong_rl.features.obs_2d import clean_auto_reset_state
from mahjong_rl.features.obs_v6 import encode_obs_v6
from mahjong_rl.models.belief import BeliefNet
from mahjong_rl.models.equivariant import SovereignPolicyNet
from mahjong_rl.models.grp import GRPNet

# Extended rule presets (Tenhou 4d..10d + M-League + Mahjong Soul Throne)
RAW_MAJSOUL_THRONE: Tuple[float, float, float, float] = (125.0, 60.0, -5.0, -180.0)
REVIEW_RULE_MAP: Dict[str, jnp.ndarray] = {
    **DAN_UTILITY_MAP,
    "phoenix_7d": DAN_UTILITY_MAP["7d"],
    "phoenix_10d": DAN_UTILITY_MAP["10d"],
    "m_league": DAN_UTILITY_MAP["mleague"],
    "majsoul": vnm_normalize_utility(jnp.array(RAW_MAJSOUL_THRONE, dtype=jnp.float32)),
    "throne": vnm_normalize_utility(jnp.array(RAW_MAJSOUL_THRONE, dtype=jnp.float32)),
}

_TILE_STRS_34 = (
    [f"{r}m" for r in range(1, 10)]
    + [f"{r}p" for r in range(1, 10)]
    + [f"{r}s" for r in range(1, 10)]
    + ["东", "南", "西", "北", "白", "发", "中"]
)
_MJAI_HONORS = ("E", "S", "W", "N", "P", "F", "C")
_WIND_CN = ("东", "南", "西", "北")


def format_tile_cn(tile_id: int) -> str:
    """Format MahJax tile id (`0..36`) into concise readable tile string."""
    t = int(tile_id)
    if t == 34:
        return "0m(赤5m)"
    if t == 35:
        return "0p(赤5p)"
    if t == 36:
        return "0s(赤5s)"
    if 0 <= t < 34:
        return _TILE_STRS_34[t]
    return f"tile#{t}"


def format_action_cn(action: int, last_draw: int = -1, target: int = -1) -> str:
    """Format MahJax 87-action id into concise Chinese Mahjong notation."""
    a = int(action)
    if 0 <= a <= 36:
        return f"打 {format_tile_cn(a)}"
    if 37 <= a <= 70:
        return f"暗/加杠 {format_tile_cn(a - 37)}"
    if a == Action.TSUMOGIRI:
        return f"摸切 {format_tile_cn(last_draw)}" if last_draw >= 0 else "摸切"
    if a == Action.RIICHI:
        return "立直"
    if a == Action.TSUMO:
        return "自摸"
    if a == Action.RON:
        return "荣和"
    if a == Action.PON:
        return f"碰 {format_tile_cn(target)}" if target >= 0 else "碰"
    if a == Action.PON_RED:
        return f"碰(赤) {format_tile_cn(target)}" if target >= 0 else "碰(赤)"
    if a == Action.OPEN_KAN:
        return f"大明杠 {format_tile_cn(target)}" if target >= 0 else "大明杠"
    if a == Action.CHI_L:
        return "吃(左)"
    if a == Action.CHI_L_RED:
        return "吃(左·赤)"
    if a == Action.CHI_M:
        return "吃(中)"
    if a == Action.CHI_M_RED:
        return "吃(中·赤)"
    if a == Action.CHI_R:
        return "吃(右)"
    if a == Action.CHI_R_RED:
        return "吃(右·赤)"
    if a == Action.PASS:
        return "PASS(跳过)"
    if a == Action.KYUUSHU:
        return "九种九牌"
    return f"action#{a}"


def is_discard_action(action: int) -> bool:
    a = int(action)
    return (0 <= a <= 36) or (a == Action.TSUMOGIRI)


def format_round_label(round_num: int, honba: int) -> str:
    r = int(round_num)
    wind = _WIND_CN[min(r // 4, 3)]
    kyoku = (r % 4) + 1
    return f"{wind}{kyoku}局 {int(honba)}本场"


# =============================================================================
# 1. Replay Parsers: MJAI JSONL/JSON, Tenhou JSON (Majsoul/Tenhou), Tenhou XML
# =============================================================================


def _tenhou_int_to_mjai(code: int) -> str:
    """Convert Tenhou JSON tile integer (`11..19, 21..29, 31..39, 41..47, 51..53`) to MJAI."""
    c = int(code)
    if c == 51:
        return "5mr"
    if c == 52:
        return "5pr"
    if c == 53:
        return "5sr"
    if 11 <= c <= 19:
        return f"{c - 10}m"
    if 21 <= c <= 29:
        return f"{c - 20}p"
    if 31 <= c <= 39:
        return f"{c - 30}s"
    if 41 <= c <= 47:
        return _MJAI_HONORS[c - 41]
    raise ValueError(f"Unknown Tenhou JSON tile code: {code!r}")


def _tenhou_136_to_mjai(tid: int) -> str:
    """Convert Tenhou XML 136-tile index (`0..135`) to MJAI notation (with red 5s at 16, 52, 88)."""
    t = int(tid)
    if t == 16:
        return "5mr"
    if t == 52:
        return "5pr"
    if t == 88:
        return "5sr"
    t34 = t // 4
    if t34 < 9:
        return f"{t34 + 1}m"
    if t34 < 18:
        return f"{t34 - 8}p"
    if t34 < 27:
        return f"{t34 - 17}s"
    return _MJAI_HONORS[t34 - 27]


def _parse_tenhou_meld_str(meld_str: str) -> Tuple[str, str, List[str], int]:
    """Parse Tenhou JSON meld string (e.g. 'c121314', 'p252525', '15p1515', '111111a11', 'k33333333').

    Returns `(kind, called_mjai, consumed_mjai_list, rel_target_idx)` where `rel_target_idx`:
    `0` = kamicha (-1), `1` = toimen (+2), `2` = shimocha (+1), `-1` = self.
    """
    m = re.search(r"([cpmak])", meld_str)
    if not m:
        raise ValueError(f"Invalid Tenhou meld string: {meld_str!r}")
    kind = m.group(1)
    pos = m.start()
    before = [int(meld_str[i : i + 2]) for i in range(0, pos, 2)]
    after = [int(meld_str[i : i + 2]) for i in range(pos + 1, len(meld_str), 2)]

    if kind == "a":
        # Ankan: e.g. '111111a11' -> 4 tiles
        all_tiles = [_tenhou_int_to_mjai(x) for x in (before + after)]
        return "ankan", all_tiles[0], all_tiles, -1
    if kind == "k":
        # Kakan: tile right after 'k' is the added tile
        added = _tenhou_int_to_mjai(after[0])
        rest = [_tenhou_int_to_mjai(x) for x in (before + after[1:])]
        return "kakan", added, rest, -1

    called = _tenhou_int_to_mjai(after[0])
    consumed = [_tenhou_int_to_mjai(x) for x in (before + after[1:])]
    if kind == "c":
        return "chi", called, consumed, 0
    if kind == "p":
        # Position of 'p': 0 -> kamicha, 1 -> toimen, 2 -> shimocha
        rel = min(len(before), 2)
        return "pon", called, consumed, rel
    if kind == "m":
        # Position of 'm': 0 -> kamicha, 1 -> toimen, 3 -> shimocha
        rel = 0 if len(before) == 0 else (1 if len(before) == 1 else 2)
        return "daiminkan", called, consumed, rel
    raise ValueError(f"Unsupported meld kind: {kind!r}")


def convert_tenhou_json_to_mjai(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert Tenhou JSON (`tenhou.net/6` or Majsoul-exported JSON) into standard MJAI events."""
    names = [str(x) for x in data.get("name", ["Seat 0", "Seat 1", "Seat 2", "Seat 3"])]
    while len(names) < 4:
        names.append(f"Seat {len(names)}")

    events: List[Dict[str, Any]] = [{"type": "start_game"}]
    winds = ("E", "S", "W", "N")

    for kyoku in data.get("log", []):
        round_info = kyoku[0]
        round_num = int(round_info[0])
        honba = int(round_info[1])
        kyotaku = int(round_info[2])
        scores = [int(s) for s in kyoku[1]]
        dora_list = [_tenhou_int_to_mjai(x) for x in kyoku[2]]
        bakaze = winds[min(round_num // 4, 3)]
        kyoku_num = (round_num % 4) + 1
        oya = round_num % 4

        tehais = [[_tenhou_int_to_mjai(x) for x in kyoku[4 + p * 3]] for p in range(4)]
        draws = [list(kyoku[5 + p * 3]) for p in range(4)]
        discards = [list(kyoku[6 + p * 3]) for p in range(4)]

        events.append(
            {
                "type": "start_kyoku",
                "bakaze": bakaze,
                "dora_marker": dora_list[0] if dora_list else "1m",
                "kyoku": kyoku_num,
                "honba": honba,
                "kyotaku": kyotaku,
                "oya": oya,
                "scores": scores,
                "tehais": tehais,
            }
        )

        draw_idx = [0, 0, 0, 0]
        disc_idx = [0, 0, 0, 0]
        dora_emitted = 1
        actor = oya
        last_discard_actor = oya
        last_discard_pai = ""
        pending_reach_actor: Optional[int] = None

        # Step through turns by following draws, calls, and discards
        max_guard = 400
        for _ in range(max_guard):
            # 1. Process current actor's draw or call entry
            if draw_idx[actor] >= len(draws[actor]):
                break
            d_entry = draws[actor][draw_idx[actor]]
            draw_idx[actor] += 1

            last_drawn_pai = ""
            if isinstance(d_entry, int):
                last_drawn_pai = _tenhou_int_to_mjai(d_entry)
                events.append({"type": "tsumo", "actor": actor, "pai": last_drawn_pai})
            elif isinstance(d_entry, str):
                kind, called, consumed, _ = _parse_tenhou_meld_str(d_entry)
                if pending_reach_actor is not None:
                    events.append({"type": "reach_accepted", "actor": pending_reach_actor})
                    pending_reach_actor = None
                events.append(
                    {
                        "type": kind,
                        "actor": actor,
                        "target": last_discard_actor,
                        "pai": called,
                        "consumed": consumed,
                    }
                )
                if kind == "daiminkan":
                    # Replacement draw follows in draws[actor], plus new dora
                    if dora_emitted < len(dora_list):
                        events.append({"type": "dora", "dora_marker": dora_list[dora_emitted]})
                        dora_emitted += 1
                    continue

            # 2. Process current actor's discard / riichi / self-kan
            while True:
                if disc_idx[actor] >= len(discards[actor]):
                    # Tsumo or abortive draw ended the round right after draw
                    actor = -1
                    break
                x_entry = discards[actor][disc_idx[actor]]
                disc_idx[actor] += 1

                if isinstance(x_entry, int):
                    tsumogiri = x_entry == 60
                    pai = last_drawn_pai if tsumogiri else _tenhou_int_to_mjai(x_entry)
                    events.append({"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": tsumogiri})
                    last_discard_actor = actor
                    last_discard_pai = pai
                    break

                if isinstance(x_entry, str):
                    if x_entry.startswith("r"):
                        events.append({"type": "reach", "actor": actor})
                        pending_reach_actor = actor
                        code = int(x_entry[1:])
                        tsumogiri = code == 60
                        pai = last_drawn_pai if tsumogiri else _tenhou_int_to_mjai(code)
                        events.append({"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": tsumogiri})
                        last_discard_actor = actor
                        last_discard_pai = pai
                        break
                    else:
                        kind, called, consumed, _ = _parse_tenhou_meld_str(x_entry)
                        if kind == "ankan":
                            events.append({"type": "ankan", "actor": actor, "consumed": consumed})
                        elif kind == "kakan":
                            events.append({"type": "kakan", "actor": actor, "pai": called, "consumed": consumed})
                        if dora_emitted < len(dora_list):
                            events.append({"type": "dora", "dora_marker": dora_list[dora_emitted]})
                            dora_emitted += 1
                        # Replacement draw for self-kan
                        if draw_idx[actor] >= len(draws[actor]):
                            actor = -1
                            break
                        rep_d = draws[actor][draw_idx[actor]]
                        draw_idx[actor] += 1
                        if isinstance(rep_d, int):
                            last_drawn_pai = _tenhou_int_to_mjai(rep_d)
                            events.append({"type": "tsumo", "actor": actor, "pai": last_drawn_pai})
                        continue

            if actor < 0:
                break

            # 3. Check if any player called the tile just discarded
            caller = None
            for offset in (1, 2, 3):
                p = (last_discard_actor + offset) % 4
                if draw_idx[p] < len(draws[p]) and isinstance(draws[p][draw_idx[p]], str):
                    # Verify this call actually targets last_discard_pai
                    kind, called, _, rel = _parse_tenhou_meld_str(draws[p][draw_idx[p]])
                    expected_target = (p - 1) % 4 if rel == 0 else ((p + 2) % 4 if rel == 1 else (p + 1) % 4)
                    if called == last_discard_pai and expected_target == last_discard_actor:
                        caller = p
                        break

            if caller is not None:
                actor = caller
            else:
                if pending_reach_actor is not None:
                    events.append({"type": "reach_accepted", "actor": pending_reach_actor})
                    pending_reach_actor = None
                actor = (last_discard_actor + 1) % 4

        # 4. Round outcome
        result = kyoku[16] if len(kyoku) > 16 else []
        if result:
            r_type = str(result[0])
            if r_type == "和了":
                # Can have multiple winners (double ron)
                for k in range(1, len(result), 2):
                    deltas = result[k]
                    info = result[k + 1] if k + 1 < len(result) else []
                    winner = int(info[0]) if len(info) > 0 else last_discard_actor
                    loser = int(info[1]) if len(info) > 1 else winner
                    events.append(
                        {
                            "type": "hora",
                            "actor": winner,
                            "target": loser,
                            "deltas": [int(x) for x in deltas],
                        }
                    )
            else:
                deltas = [int(x) for x in result[1]] if len(result) > 1 and isinstance(result[1], list) else [0, 0, 0, 0]
                reason = "kyushukyuhai" if "九種" in r_type else "howanpai"
                events.append({"type": "ryukyoku", "reason": reason, "deltas": deltas})

        events.append({"type": "end_kyoku"})

    events.append({"type": "end_game"})
    return events, names


def _decode_tenhou_xml_meld(m_val: int) -> Tuple[str, str, List[str]]:
    """Decode Tenhou XML `<N m="...">` 16-bit meld integer into `(kind, called_mjai, consumed_mjai)`."""
    m = int(m_val)
    if m & (1 << 2):
        # Chi
        t = (m >> 10) & 0x3F
        r = t % 3
        t //= 3
        suit = t // 7
        rank = t % 7
        base34 = suit * 9 + rank
        offsets = [(m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3]
        tids = [(base34 + i) * 4 + offsets[i] for i in range(3)]
        called = _tenhou_136_to_mjai(tids[r])
        consumed = [_tenhou_136_to_mjai(tids[i]) for i in range(3) if i != r]
        return "chi", called, consumed
    if m & (1 << 3):
        # Pon
        unused = (m >> 5) & 3
        t = (m >> 9) & 0x7F
        r = t % 3
        t34 = t // 3
        tids = [t34 * 4 + i for i in range(4) if i != unused]
        called = _tenhou_136_to_mjai(tids[r])
        consumed = [_tenhou_136_to_mjai(tids[i]) for i in range(3) if i != r]
        return "pon", called, consumed
    if m & (1 << 4):
        # Kakan
        added_off = (m >> 5) & 3
        t = (m >> 9) & 0x7F
        t34 = t // 3
        added_tid = t34 * 4 + added_off
        rest = [_tenhou_136_to_mjai(t34 * 4 + i) for i in range(4) if i != added_off]
        return "kakan", _tenhou_136_to_mjai(added_tid), rest
    # Kan (ankan or daiminkan)
    hai0 = (m >> 8) & 0xFF
    t34 = hai0 // 4
    from_who = m & 3
    if from_who == 0:
        consumed = [_tenhou_136_to_mjai(t34 * 4 + i) for i in range(4)]
        return "ankan", consumed[0], consumed
    called = _tenhou_136_to_mjai(hai0)
    consumed = [_tenhou_136_to_mjai(t34 * 4 + i) for i in range(4) if t34 * 4 + i != hai0]
    return "daiminkan", called, consumed


def convert_tenhou_xml_to_mjai(xml_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert Tenhou `.mjlog` XML string into standard MJAI events."""
    root = ET.fromstring(xml_text)
    names = ["Seat 0", "Seat 1", "Seat 2", "Seat 3"]
    events: List[Dict[str, Any]] = [{"type": "start_game"}]
    winds = ("E", "S", "W", "N")
    last_drawn = ["", "", "", ""]
    last_discard_actor = 0
    in_kyoku = False

    draw_tags = {"T": 0, "U": 1, "V": 2, "W": 3}
    disc_tags = {"D": 0, "E": 1, "F": 2, "G": 3}

    for elem in root:
        tag = elem.tag
        if tag == "UN":
            for p in range(4):
                raw_n = elem.attrib.get(f"n{p}")
                if raw_n:
                    names[p] = urllib.parse.unquote(raw_n)
        elif tag == "INIT":
            if in_kyoku:
                events.append({"type": "end_kyoku"})
            in_kyoku = True
            seed_parts = [int(x) for x in elem.attrib["seed"].split(",")]
            round_num, honba, kyotaku = seed_parts[0], seed_parts[1], seed_parts[2]
            dora_marker = _tenhou_136_to_mjai(seed_parts[5])
            scores = [int(x) * 100 for x in elem.attrib["ten"].split(",")]
            oya = int(elem.attrib["oya"])
            tehais = [
                [_tenhou_136_to_mjai(int(x)) for x in elem.attrib[f"hai{p}"].split(",") if x]
                for p in range(4)
            ]
            events.append(
                {
                    "type": "start_kyoku",
                    "bakaze": winds[min(round_num // 4, 3)],
                    "dora_marker": dora_marker,
                    "kyoku": (round_num % 4) + 1,
                    "honba": honba,
                    "kyotaku": kyotaku,
                    "oya": oya,
                    "scores": scores,
                    "tehais": tehais,
                }
            )
            last_drawn = ["", "", "", ""]
        elif len(tag) >= 2 and tag[0] in draw_tags and tag[1:].isdigit():
            p = draw_tags[tag[0]]
            pai = _tenhou_136_to_mjai(int(tag[1:]))
            last_drawn[p] = pai
            events.append({"type": "tsumo", "actor": p, "pai": pai})
        elif len(tag) >= 2 and tag[0] in disc_tags and tag[1:].isdigit():
            p = disc_tags[tag[0]]
            pai = _tenhou_136_to_mjai(int(tag[1:]))
            tsumogiri = pai == last_drawn[p]
            events.append({"type": "dahai", "actor": p, "pai": pai, "tsumogiri": tsumogiri})
            last_discard_actor = p
        elif tag == "N":
            p = int(elem.attrib["who"])
            kind, called, consumed = _decode_tenhou_xml_meld(int(elem.attrib["m"]))
            if kind == "ankan":
                events.append({"type": "ankan", "actor": p, "consumed": consumed})
            elif kind == "kakan":
                events.append({"type": "kakan", "actor": p, "pai": called, "consumed": consumed})
            else:
                events.append(
                    {
                        "type": kind,
                        "actor": p,
                        "target": last_discard_actor,
                        "pai": called,
                        "consumed": consumed,
                    }
                )
        elif tag == "REACH":
            p = int(elem.attrib["who"])
            step = int(elem.attrib["step"])
            events.append({"type": "reach" if step == 1 else "reach_accepted", "actor": p})
        elif tag == "DORA":
            events.append({"type": "dora", "dora_marker": _tenhou_136_to_mjai(int(elem.attrib["hai"]))})
        elif tag == "AGARI":
            who = int(elem.attrib["who"])
            from_who = int(elem.attrib["fromWho"])
            sc = [int(x) for x in elem.attrib["sc"].split(",")]
            deltas = [sc[2 * i + 1] * 100 for i in range(4)]
            events.append({"type": "hora", "actor": who, "target": from_who, "deltas": deltas})
        elif tag == "RYUUKYOKU":
            sc = [int(x) for x in elem.attrib.get("sc", "0,0,0,0,0,0,0,0").split(",")]
            deltas = [sc[2 * i + 1] * 100 for i in range(4)] if len(sc) >= 8 else [0, 0, 0, 0]
            reason = "kyushukyuhai" if elem.attrib.get("type") == "yao9" else "howanpai"
            events.append({"type": "ryukyoku", "reason": reason, "deltas": deltas})

    if in_kyoku:
        events.append({"type": "end_kyoku"})
    events.append({"type": "end_game"})
    return events, names


def load_replay_to_mjai(source: str | Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load a replay from file path or Tenhou URL and return `(mjai_events, player_names)`."""
    src_str = str(source).strip()
    if src_str.startswith(("http://", "https://")):
        if "maj-soul.com" in src_str or "mahjongsoul" in src_str:
            raise ValueError(
                "Mahjong Soul URLs require authenticated WebSocket decryption. "
                "Please export the log as Tenhou JSON or MJAI JSONL (e.g. via Majsoul Plus / "
                "雀魂牌谱屋 / mjai-reviewer) and pass the exported file path."
            )
        # Support Tenhou URL like https://tenhou.net/0/?log=2026010100gm-00a9-0000-...
        parsed = urllib.parse.urlparse(src_str)
        qs = urllib.parse.parse_qs(parsed.query)
        log_id = qs.get("log", [""])[0]
        if not log_id:
            raise ValueError(f"Could not extract Tenhou `log=` id from URL: {src_str}")
        fetch_url = f"https://tenhou.net/0/log/?{log_id}"
        req = urllib.request.Request(fetch_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8")
        return convert_tenhou_xml_to_mjai(text)

    path = Path(src_str)
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty replay file: {path}")

    # 1. Tenhou XML (.mjlog)
    if raw.startswith("<"):
        return convert_tenhou_xml_to_mjai(raw)

    # 2. JSON object (Tenhou JSON / Majsoul-exported JSON) or JSON array (MJAI list)
    if raw.startswith("{") or raw.startswith("["):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "log" in obj:
                return convert_tenhou_json_to_mjai(obj)
            if isinstance(obj, list):
                names = ["Seat 0", "Seat 1", "Seat 2", "Seat 3"]
                for ev in obj:
                    if isinstance(ev, dict) and ev.get("type") == "start_game" and "names" in ev:
                        names = [str(x) for x in ev["names"][:4]]
                        break
                return obj, names
        except json.JSONDecodeError:
            pass  # Fall through to JSONL

    # 3. MJAI JSONL (one JSON event per line)
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    names = ["Seat 0", "Seat 1", "Seat 2", "Seat 3"]
    for ev in events:
        if ev.get("type") == "start_game" and "names" in ev:
            names = [str(x) for x in ev["names"][:4]]
            break
    return events, names


# =============================================================================
# 2. Checkpoint Loader (Local or Hugging Face Hub)
# =============================================================================


def load_v6_reviewer_models(
    ckpt_dir: Optional[Path | str] = None,
    *,
    hf_repo: Optional[str] = None,
    policy_ckpt: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Load GRP_base, BeliefNet, and SovereignPolicyNet for replay review."""
    if ckpt_dir is None:
        ckpt_dir = Path("checkpoints/sovereign_v6")
    cdir = Path(ckpt_dir)

    if hf_repo and not (cdir / "grp_phase0b.pkl").exists():
        from mahjong_rl.data.hf_sync import download_checkpoints_from_hf
        download_checkpoints_from_hf(hf_repo, cdir)

    with open(cdir / "grp_phase0b.pkl", "rb") as f:
        d_grp = pickle.load(f)
    grp_cfg = GRPConfig(**d_grp["config"])
    grp_model = GRPNet(config=grp_cfg)
    grp_params = jax.device_put(d_grp["params"])

    with open(cdir / "belief_phase0a.pkl", "rb") as f:
        d_bel = pickle.load(f)
    bel_cfg = BeliefConfig(**d_bel["config"])
    bel_model = BeliefNet(config=bel_cfg)
    bel_params = jax.device_put(d_bel["params"])
    belief_t_cal = float(d_bel.get("t_cal", 1.0))

    # Load model_cfg from bc_phase0c.pkl or deploy_sdtap.pkl
    model_cfg = ModelConfig()
    if (cdir / "bc_phase0c.pkl").exists():
        with open(cdir / "bc_phase0c.pkl", "rb") as f:
            d_bc = pickle.load(f)
        if "model_config" in d_bc:
            model_cfg = ModelConfig(**d_bc["model_config"])

    # Pick policy checkpoint: explicit > deploy_sdtap > rl_phase2 > rl_resume_state > bc_phase0c
    if policy_ckpt is not None:
        pol_path = Path(policy_ckpt)
    else:
        candidates = [
            cdir / "deploy_sdtap.pkl",
            cdir / "rl_phase2.pkl",
            cdir / "rl_resume_state.pkl",
            cdir / "bc_phase0c.pkl",
        ]
        pol_path = next((p for p in candidates if p.exists()), None)
        if pol_path is None:
            raise FileNotFoundError(f"No policy checkpoint found in {cdir}")

    with open(pol_path, "rb") as f:
        d_pol = pickle.load(f)
    if "model_config" in d_pol:
        model_cfg = ModelConfig(**d_pol["model_config"])
    pol_model = SovereignPolicyNet(config=model_cfg)

    pol_params = jax.device_put(
        d_pol.get("policy_params", d_pol.get("params_pi", d_pol.get("params")))
    )
    if "belief_ema_params" in d_pol:
        bel_params = jax.device_put(d_pol["belief_ema_params"])
    elif "params_b_ema" in d_pol:
        bel_params = jax.device_put(d_pol["params_b_ema"])
    if "t_cal" in d_pol:
        belief_t_cal = float(d_pol["t_cal"])

    return {
        "policy_model": pol_model,
        "policy_params": pol_params,
        "belief_model": bel_model,
        "belief_params": bel_params,
        "belief_t_cal": belief_t_cal,
        "grp_model": grp_model,
        "grp_params": grp_params,
        "policy_source": str(pol_path),
    }


# =============================================================================
# 3. Core Review Engine (Agreement Rate + Mortal-Style Rating + Blunders)
# =============================================================================


@dataclass
class DecisionReview:
    round_idx: int
    round_label: str
    turn: int
    seat: int
    is_discard: bool
    human_action: int
    human_action_str: str
    human_prob: float
    best_action: int
    best_action_str: str
    best_prob: float
    prob_ratio: float
    nll: float
    matched_top1: bool
    matched_top2: bool
    matched_top3: bool
    is_blunder: bool
    top3: List[Tuple[str, float]]
    delta_u_ron_human: float
    delta_u_ron_best: float
    max_opp_tenpai: float


def _build_eval_step_fn(
    policy_model: SovereignPolicyNet,
    policy_params: Any,
    belief_model: BeliefNet,
    belief_params: Any,
    grp_model: GRPNet,
    grp_params: Any,
    w_rule: jnp.ndarray,
    belief_t_cal: float,
):
    w_tab = make_homogeneous_table(w_rule)
    is_known = jnp.zeros((3,), dtype=jnp.float32)
    grp_fn = lambda gf: grp_model.apply(grp_params, gf)

    @jax.jit
    def eval_state(state):
        clean_st = clean_auto_reset_state(state)
        cp = clean_st.current_player.astype(jnp.int32)
        legal_mask = clean_st.legal_action_mask.astype(jnp.bool_)
        obs = encode_obs_v6(clean_st)
        tbl = extract_table_snapshot(clean_st)
        w_ego_4x4 = rotate_utility_matrix_ego(w_tab, cp)
        w_me = w_ego_4x4[0]
        w_opp_5d = build_opponent_utility_features(
            w_ego_4x4[1:],
            is_known,
            rng_key=None,
            dropout_prob=0.0,
            w_table_default=w_rule,
        )
        b_out = belief_model.apply(
            belief_params,
            planes=obs["planes"],
            scalars=obs["scalars"],
            river_tokens=obs["river_tokens"],
            river_tile_ids=obs["river_tile_ids"],
            river_valid_mask=obs["river_valid_mask"],
            w_opp_5d=w_opp_5d,
            temperature_cal=belief_t_cal,
        )
        du_ron_3opp, du_ron_87, du_agari_3, du_ryu, call_threat = compute_analytical_trinity(
            s_table=tbl,
            p_deal_3opp=b_out["p_deal"],
            p_score_bins=b_out["p_score_bin"],
            p_call_threat_3=b_out["p_call_threat"],
            p_tenpai_3=b_out["p_tenpai"],
            w_me=w_me,
            last_draw_type=obs["last_draw_type"],
            is_last_draw_red=obs["is_last_draw_red"],
            grp_fn=grp_fn,
        )
        logits = policy_model.apply(
            policy_params,
            planes=obs["planes"],
            delta_u_ron_3opp=du_ron_3opp,
            call_threat_plane=call_threat,
            scalars=obs["scalars"],
            river_tokens=obs["river_tokens"],
            river_tile_ids=obs["river_tile_ids"],
            river_valid_mask=obs["river_valid_mask"],
            last_draw_type=obs["last_draw_type"],
            is_last_draw_red=obs["is_last_draw_red"],
            target_type=obs["target_type"],
            target_is_red=obs["target_is_red"],
            w_ego=w_me,
            delta_u_agari_3=du_agari_3,
            delta_u_ryuukyoku=du_ryu,
            delta_u_ron_87=du_ron_87,
            legal_action_mask=legal_mask,
        )
        probs = masked_categorical_probs(logits, legal_mask)
        return probs, du_ron_87, b_out["p_tenpai"]

    return eval_state


def _summarize_decisions(decisions: List[DecisionReview], top_k_blunders: int = 10) -> Dict[str, Any]:
    n = len(decisions)
    if n == 0:
        return {
            "total_decisions": 0,
            "rating": 0.0,
            "geo_rating": 0.0,
            "overall_acc": 0.0,
            "top2_acc": 0.0,
            "top3_acc": 0.0,
            "discard_decisions": 0,
            "discard_acc": 0.0,
            "discard_rating": 0.0,
            "call_decisions": 0,
            "call_acc": 0.0,
            "call_rating": 0.0,
            "blunder_count": 0,
            "blunder_rate": 0.0,
            "avg_nll": 0.0,
            "perplexity": 0.0,
            "top_blunders": [],
        }

    ratios = np.array([d.prob_ratio for d in decisions], dtype=np.float64)
    nlls = np.array([d.nll for d in decisions], dtype=np.float64)
    disc = [d for d in decisions if d.is_discard]
    call = [d for d in decisions if not d.is_discard]
    blunders = [d for d in decisions if d.is_blunder]

    # Sort worst deviations by (best_prob - human_prob) descending
    ranked_blunders = sorted(
        [d for d in decisions if not d.matched_top1],
        key=lambda d: (d.best_prob - d.human_prob),
        reverse=True,
    )[:top_k_blunders]

    avg_nll = float(np.mean(nlls))
    return {
        "total_decisions": n,
        "rating": float(100.0 * np.mean(ratios)),
        "geo_rating": float(100.0 * np.exp(np.mean(np.log(np.clip(ratios, 1e-4, 1.0))))),
        "overall_acc": float(np.mean([d.matched_top1 for d in decisions])),
        "top2_acc": float(np.mean([d.matched_top2 for d in decisions])),
        "top3_acc": float(np.mean([d.matched_top3 for d in decisions])),
        "discard_decisions": len(disc),
        "discard_acc": float(np.mean([d.matched_top1 for d in disc])) if disc else 0.0,
        "discard_rating": float(100.0 * np.mean([d.prob_ratio for d in disc])) if disc else 0.0,
        "call_decisions": len(call),
        "call_acc": float(np.mean([d.matched_top1 for d in call])) if call else 0.0,
        "call_rating": float(100.0 * np.mean([d.prob_ratio for d in call])) if call else 0.0,
        "blunder_count": len(blunders),
        "blunder_rate": float(len(blunders) / n),
        "avg_nll": avg_nll,
        "perplexity": float(math.exp(min(avg_nll, 20.0))),
        "top_blunders": [asdict(d) for d in ranked_blunders],
    }


def review_replay(
    replay_source: str | Path,
    models: Dict[str, Any],
    *,
    rule: str = "7d",
    custom_rule_points: Optional[Sequence[float]] = None,
    target_seat: Optional[int] = None,
    top_k_blunders: int = 10,
) -> Dict[str, Any]:
    """Review a full Tenhou / Mahjong Soul / MJAI replay with Sovereign-Mahjong V6."""
    if custom_rule_points is not None:
        w_rule = vnm_normalize_utility(jnp.array(list(custom_rule_points), dtype=jnp.float32))
        rule_name = f"custom({','.join(str(x) for x in custom_rule_points)})"
    else:
        key = rule.lower().strip()
        if key not in REVIEW_RULE_MAP:
            raise ValueError(f"Unknown rule {rule!r}. Choose from {sorted(REVIEW_RULE_MAP)}")
        w_rule = REVIEW_RULE_MAP[key]
        rule_name = key

    events, player_names = load_replay_to_mjai(replay_source)
    events_jsonl = "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events)
    parsed_rounds = parse_mjai_game_to_rounds(events_jsonl, max_events_per_round=160)
    if not parsed_rounds:
        raise ValueError("No valid kyokus could be parsed from replay.")

    env = mahjax.make("red_4p", round_mode="half", observe_type="dict", round_end="confirm")
    step_fn = jax.jit(env.step)
    eval_fn = _build_eval_step_fn(
        policy_model=models["policy_model"],
        policy_params=models["policy_params"],
        belief_model=models["belief_model"],
        belief_params=models["belief_params"],
        grp_model=models["grp_model"],
        grp_params=models["grp_params"],
        w_rule=w_rule,
        belief_t_cal=models["belief_t_cal"],
    )
    dummy_key = jax.random.PRNGKey(0)

    decisions_by_seat: Dict[int, List[DecisionReview]] = {0: [], 1: [], 2: [], 3: []}
    desynced_rounds = 0

    for r_i, pr in enumerate(parsed_rounds):
        st = _init_round_state_from_deck(
            jnp.asarray(pr.deck),
            jnp.asarray(pr.round_num),
            jnp.asarray(pr.honba),
            jnp.asarray(pr.kyotaku),
            jnp.asarray(pr.dealer),
            jnp.asarray(pr.scores),
            round_limit=7,
        )
        r_label = format_round_label(int(pr.round_num), int(pr.honba))
        cursor = 0
        n_ev = int(pr.n_events)
        turn_counter = [0, 0, 0, 0]

        for _ in range(220):
            round_over = bool(st.round_state.terminated_round) or bool(st.terminated) or bool(st.truncated)
            if round_over:
                break
            cp = int(st.current_player)
            mask_np = np.asarray(st.legal_action_mask, dtype=bool)
            num_legal = int(mask_np.sum())

            # Match replay event to env action (exact same protocol as mjai_parquet_converter)
            matched_ev = False
            chosen_action = int(Action.DUMMY)
            if cursor < n_ev:
                tgt_actor = int(pr.ev_actors[cursor])
                tgt_action = int(pr.ev_actions[cursor])
                if tgt_actor == cp and 0 <= tgt_action < Action.NUM_ACTION and mask_np[tgt_action]:
                    chosen_action = tgt_action
                    matched_ev = True
                elif (
                    tgt_actor == cp
                    and 0 <= tgt_action < 37
                    and not mask_np[tgt_action]
                    and int(st.round_state.last_draw) == tgt_action
                    and mask_np[Action.TSUMOGIRI]
                ):
                    chosen_action = int(Action.TSUMOGIRI)
                    matched_ev = True
                elif tgt_actor < 0 and tgt_action == Action.KYUUSHU and mask_np[Action.KYUUSHU]:
                    chosen_action = int(Action.KYUUSHU)
                    matched_ev = True

            if matched_ev:
                cursor += 1
            elif mask_np[Action.PASS]:
                chosen_action = int(Action.PASS)
            else:
                desynced_rounds += 1
                break

            is_disc = is_discard_action(chosen_action)
            if is_disc:
                turn_counter[cp] += 1

            # Evaluate non-forced decisions for requested seat(s)
            if num_legal >= 2 and chosen_action != Action.DUMMY and (target_seat is None or cp == target_seat):
                probs_j, du_ron_j, p_tenpai_j = eval_fn(st)
                probs = np.asarray(jax.device_get(probs_j), dtype=np.float64)
                du_ron = np.asarray(jax.device_get(du_ron_j), dtype=np.float64)
                p_tenpai = np.asarray(jax.device_get(p_tenpai_j), dtype=np.float64)

                legal_idx = np.flatnonzero(mask_np)
                order = legal_idx[np.argsort(-probs[legal_idx])]
                best_act = int(order[0])
                best_p = float(probs[best_act])
                human_p = float(probs[chosen_action])
                ratio = float(np.clip(human_p / max(best_p, 1e-8), 0.0, 1.0))
                nll = float(-math.log(max(human_p, 1e-7)))

                ld = int(st.round_state.last_draw)
                tgt = int(st.round_state.target)
                top3 = [
                    (format_action_cn(int(a), last_draw=ld, target=tgt), float(probs[int(a)]))
                    for a in order[:3]
                ]
                decisions_by_seat[cp].append(
                    DecisionReview(
                        round_idx=r_i,
                        round_label=r_label,
                        turn=max(turn_counter[cp], 1),
                        seat=cp,
                        is_discard=is_disc,
                        human_action=chosen_action,
                        human_action_str=format_action_cn(chosen_action, last_draw=ld, target=tgt),
                        human_prob=human_p,
                        best_action=best_act,
                        best_action_str=format_action_cn(best_act, last_draw=ld, target=tgt),
                        best_prob=best_p,
                        prob_ratio=ratio,
                        nll=nll,
                        matched_top1=(chosen_action == best_act),
                        matched_top2=(chosen_action in order[:2]),
                        matched_top3=(chosen_action in order[:3]),
                        is_blunder=(human_p < 0.05 and best_p >= 0.50),
                        top3=top3,
                        delta_u_ron_human=float(du_ron[chosen_action]),
                        delta_u_ron_best=float(du_ron[best_act]),
                        max_opp_tenpai=float(np.max(p_tenpai)),
                    )
                )

            st = step_fn(st, jnp.int32(chosen_action), dummy_key)

    seats_report: Dict[str, Any] = {}
    all_decisions: List[DecisionReview] = []
    for p in range(4):
        if target_seat is not None and p != target_seat:
            continue
        all_decisions.extend(decisions_by_seat[p])
        seats_report[str(p)] = {
            "seat": p,
            "name": player_names[p],
            **_summarize_decisions(decisions_by_seat[p], top_k_blunders=top_k_blunders),
        }

    return {
        "replay_source": str(replay_source),
        "policy_source": models.get("policy_source", ""),
        "rule": rule_name,
        "rounds_total": len(parsed_rounds),
        "rounds_desynced": desynced_rounds,
        "overall": _summarize_decisions(all_decisions, top_k_blunders=top_k_blunders),
        "seats": seats_report,
    }


# =============================================================================
# 4. Terminal & Standalone HTML Report Renderers
# =============================================================================


def format_terminal_report(report: Dict[str, Any]) -> str:
    lines = [
        "=" * 92,
        f"Sovereign-Mahjong V6 牌谱评测报告 | 规则: {report['rule']} | 局数: {report['rounds_total']} (失步: {report['rounds_desynced']})",
        f"模型权重: {report['policy_source']}",
        "=" * 92,
        f"{'座位':<4} {'玩家':<14} {'Rating':>7} {'总一致率':>8} {'弃牌一致':>8} {'鸣牌一致':>8} {'Top-3':>7} {'恶手率':>7} {'步数':>5}",
        "-" * 92,
    ]
    for s_key, s in report["seats"].items():
        name = str(s["name"])[:12]
        lines.append(
            f"P{s_key:<3} {name:<14} {s['rating']:>7.2f} "
            f"{s['overall_acc']*100:>7.1f}% {s['discard_acc']*100:>7.1f}% "
            f"{s['call_acc']*100:>7.1f}% {s['top3_acc']*100:>6.1f}% "
            f"{s['blunder_rate']*100:>6.1f}% {s['total_decisions']:>5d}"
        )
    lines.append("=" * 92)

    for s_key, s in report["seats"].items():
        blunders = s.get("top_blunders", [])
        if not blunders:
            continue
        lines.append(f"\n[P{s_key} {s['name']}] 关键分歧手 Top {len(blunders)} (Rating={s['rating']:.2f}, 恶手={s['blunder_count']}):")
        for idx, b in enumerate(blunders, 1):
            top3_str = " | ".join(f"{act}({p*100:.1f}%)" for act, p in b["top3"])
            tag = " [恶手]" if b["is_blunder"] else ""
            lines.append(
                f"  {idx:>2}. {b['round_label']} 第{b['turn']:>2}巡 | "
                f"实战: {b['human_action_str']} ({b['human_prob']*100:.1f}%) -> "
                f"AI推荐: {top3_str} | 对手听牌率max={b['max_opp_tenpai']*100:.0f}%{tag}"
            )
    return "\n".join(lines)


def render_html_report(report: Dict[str, Any]) -> str:
    """Render a self-contained HTML review report inspired by Mortal / NAGA."""
    seat_rows = []
    blunder_sections = []
    for s_key, s in report["seats"].items():
        name = html.escape(str(s["name"]))
        seat_rows.append(
            f"<tr><td>P{s_key}</td><td><strong>{name}</strong></td>"
            f"<td class='rating'>{s['rating']:.2f}</td>"
            f"<td>{s['overall_acc']*100:.1f}%</td>"
            f"<td>{s['discard_acc']*100:.1f}% ({s['discard_decisions']})</td>"
            f"<td>{s['call_acc']*100:.1f}% ({s['call_decisions']})</td>"
            f"<td>{s['top3_acc']*100:.1f}%</td>"
            f"<td>{s['blunder_count']} ({s['blunder_rate']*100:.1f}%)</td>"
            f"<td>{s['avg_nll']:.3f}</td></tr>"
        )
        b_rows = []
        for idx, b in enumerate(s.get("top_blunders", []), 1):
            top3_html = " &nbsp; ".join(
                f"<span class='pill'>{html.escape(act)} <b>{p*100:.1f}%</b></span>" for act, p in b["top3"]
            )
            cls = "blunder" if b["is_blunder"] else ""
            b_rows.append(
                f"<tr class='{cls}'><td>{idx}</td><td>{html.escape(b['round_label'])} 第{b['turn']}巡</td>"
                f"<td><b>{html.escape(b['human_action_str'])}</b> ({b['human_prob']*100:.1f}%)</td>"
                f"<td>{top3_html}</td>"
                f"<td>{b['max_opp_tenpai']*100:.0f}%</td></tr>"
            )
        if b_rows:
            blunder_sections.append(
                f"<h2>P{s_key} {name} — 关键分歧手 / 恶手复盘 (Rating {s['rating']:.2f})</h2>"
                f"<table><thead><tr><th>#</th><th>局次/巡目</th><th>实战选择</th><th>AI Top-3 推荐</th><th>对手最高听牌率</th></tr></thead>"
                f"<tbody>{''.join(b_rows)}</tbody></table>"
            )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Sovereign-Mahjong V6 牌谱评测报告</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 24px auto; max-width: 1080px; padding: 0 16px; }}
  h1, h2 {{ color: #f8fafc; }}
  .meta {{ color: #94a3b8; margin-bottom: 18px; font-size: 14px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0 28px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid #334155; text-align: left; font-size: 14px; }}
  th {{ background: #0b1220; color: #94a3b8; font-weight: 600; }}
  .rating {{ color: #38bdf8; font-weight: 700; font-size: 16px; }}
  .pill {{ display: inline-block; background: #334155; padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
  tr.blunder {{ background: rgba(239, 68, 68, 0.14); }}
</style>
</head>
<body>
  <h1>Sovereign-Mahjong V6 牌谱评测报告</h1>
  <div class="meta">
    牌谱: {html.escape(str(report['replay_source']))} &nbsp;|&nbsp;
    规则: <b>{html.escape(str(report['rule']))}</b> &nbsp;|&nbsp;
    局数: {report['rounds_total']} &nbsp;|&nbsp;
    模型: {html.escape(str(report['policy_source']))}
  </div>
  <h2>四家一致率与 Rating 总览</h2>
  <table>
    <thead>
      <tr><th>座位</th><th>玩家</th><th>Rating</th><th>总一致率</th><th>纯弃牌一致率</th><th>鸣牌/立直一致率</th><th>Top-3</th><th>恶手</th><th>Avg NLL</th></tr>
    </thead>
    <tbody>{''.join(seat_rows)}</tbody>
  </table>
  {''.join(blunder_sections)}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign-Mahjong V6 Replay Reviewer (雀魂/天凤/MJAI 牌谱评测)")
    parser.add_argument("--replay", required=True, help="Path to MJAI .jsonl/.json, Tenhou JSON, .mjlog XML, or Tenhou URL")
    parser.add_argument("--ckpt-dir", default="checkpoints/sovereign_v6", help="Directory containing V6 checkpoints")
    parser.add_argument("--hf-repo", default=None, help="Optional HF model repo (e.g. myouo/sovereign-mahjong-v6)")
    parser.add_argument("--policy-ckpt", default=None, help="Optional explicit policy .pkl path")
    parser.add_argument("--rule", default="7d", help="Rule preset: 4d..10d, mleague, majsoul/throne")
    parser.add_argument("--rule-points", default=None, help="Optional custom 4-rank points, e.g. 125,60,-5,-180")
    parser.add_argument("--seat", type=int, default=None, choices=[0, 1, 2, 3], help="Evaluate only a specific seat (0-3)")
    parser.add_argument("--top-blunders", type=int, default=10, help="Number of top diverging moves to show per seat")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save detailed JSON report")
    parser.add_argument("--out-html", type=Path, default=None, help="Optional path to save visual HTML report")
    args = parser.parse_args()

    custom_pts = [float(x) for x in args.rule_points.split(",")] if args.rule_points else None
    models = load_v6_reviewer_models(args.ckpt_dir, hf_repo=args.hf_repo, policy_ckpt=args.policy_ckpt)
    report = review_replay(
        args.replay,
        models,
        rule=args.rule,
        custom_rule_points=custom_pts,
        target_seat=args.seat,
        top_k_blunders=args.top_blunders,
    )
    print(format_terminal_report(report))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON report -> {args.out_json}")
    if args.out_html is not None:
        args.out_html.parent.mkdir(parents=True, exist_ok=True)
        args.out_html.write_text(render_html_report(report), encoding="utf-8")
        print(f"Saved HTML report -> {args.out_html}")


if __name__ == "__main__":
    main()

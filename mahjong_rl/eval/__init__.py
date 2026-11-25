"""Evaluation & deployment utilities: Fixed-wall Duplicate 1v3 & V6 UI/mjai agent bridge."""

from mahjong_rl.eval.duplicate_eval import evaluate_duplicate_1v3
from mahjong_rl.eval.mjai_bridge import make_v6_policy_actor, register_ui_agent

__all__ = [
    "evaluate_duplicate_1v3",
    "make_v6_policy_actor",
    "register_ui_agent",
]

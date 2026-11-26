"""Evaluation & deployment utilities: Fixed-wall Duplicate 1v3, Replay Reviewer & V6 UI/mjai bridge."""

from mahjong_rl.eval.duplicate_eval import evaluate_duplicate_1v3
from mahjong_rl.eval.mjai_bridge import make_v6_policy_actor, register_ui_agent
from mahjong_rl.eval.replay_reviewer import (
    format_terminal_report,
    load_replay_to_mjai,
    load_v6_reviewer_models,
    render_html_report,
    review_replay,
)

__all__ = [
    "evaluate_duplicate_1v3",
    "format_terminal_report",
    "load_replay_to_mjai",
    "load_v6_reviewer_models",
    "make_v6_policy_actor",
    "register_ui_agent",
    "render_html_report",
    "review_replay",
]

"""Sovereign-Mahjong V6 Training Entry Points (Phases 0-A, 0-B, 0-C, 1, 2, 3)."""

from mahjong_rl.trainers.distill_sdtap import distill_sdtap
from mahjong_rl.trainers.train_bc import init_policy_params, train_bc
from mahjong_rl.trainers.train_belief import train_belief
from mahjong_rl.trainers.train_grp import train_grp
from mahjong_rl.trainers.train_rl import V6RLResult, train_rl_v6

__all__ = [
    "V6RLResult",
    "distill_sdtap",
    "init_policy_params",
    "train_bc",
    "train_belief",
    "train_grp",
    "train_rl_v6",
]

"""Unified configuration and hardware presets for Sovereign-Mahjong V6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

# Tenhou 4-player Phoenix 7-dan placement points (default eval/core benchmark)
TENHOU_RANK_POINTS: Tuple[int, int, int, int] = (90, 45, 0, -135)
DEFAULT_UMA_10_30: Tuple[int, int, int, int] = (300, 100, -100, -300)


@dataclass(frozen=True)
class ModelConfig:
    """36x Algebraically Equivariant Sovereign-Mahjong V6 Backbone & Head config."""

    num_blocks: int = 4               # SuitBlocked + Cross-Suit Sanshoku residual blocks
    channels: int = 128               # Backbone spatial width (~5.2M at 6x160 or lean 4x128)
    river_dim: int = 32               # Per-player CausalRiverScatterEncoder width (4 * 32 = 128ch)
    attn_heads: int = 4               # Top 34-token equivariant Multi-Head Self-Attention
    se_reduction: int = 4
    global_hidden_dim: int = 128
    num_actions: int = 87
    # Bounded strictly-positive physics residual gates: beta = beta_max * sigmoid(raw)
    beta_risk_max: float = 3.0
    beta_offense_max: float = 3.0
    # Logit centering & bidirectional bound: max odds ratio = exp(16) ~ 8.88e6
    logit_clip_bound: float = 8.0


@dataclass(frozen=True)
class BeliefConfig:
    """P0 Quasi-Static Belief Network (B_fast -> B_EMA) config."""

    num_blocks: int = 3
    channels: int = 96
    river_dim: int = 24
    attn_heads: int = 4
    global_hidden_dim: int = 96
    # Unweighted standard BCE with prior log-odds bias init log(0.015 / 0.985)
    ron_prior_bias_init: float = -4.184137
    score_bin_loss_coef: float = 0.30
    call_threat_loss_coef: float = 0.15
    tenpai_loss_coef: float = 0.05
    ema_tau: float = 1e-4             # Quasi-static EMA smoothing for B_EMA
    utility_dropout: float = 0.20     # Bayesian fallback to w_table + is_known=0
    lr: float = 2e-4
    weight_decay: float = 5e-4
    grad_clip_norm: float = 0.3
    batch_size: int = 512
    num_epochs: int = 5
    seed: int = 0


@dataclass(frozen=True)
class GRPConfig:
    """Birkhoff Doubly-Stochastic (4, 4) Global Reward Predictor (GRP_base + GRP_EMA)."""

    hidden_dim: int = 128
    num_layers: int = 3
    sinkhorn_iters: int = 50
    tau_prior: float = 1.5            # Temperature scaling when injected as Oracle residual
    ema_tau: float = 1e-4             # Slow GRP_EMA for TD anchoring only (input stays frozen!)
    lr: float = 3e-4
    batch_size: int = 1024
    num_epochs: int = 10


@dataclass(frozen=True)
class OracleConfig:
    """P1 Birkhoff Doubly-Stochastic History-State Dual-Horizon Oracle Critic."""

    num_blocks: int = 4
    channels: int = 128
    attn_heads: int = 4
    global_hidden_dim: int = 128
    sinkhorn_iters: int = 50
    # Dual-horizon training-only auxiliary heads (zero inference cost)
    round_win_loss_coef: float = 0.20
    round_score_loss_coef: float = 0.05
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class RNaDConfig:
    """P3 Clean Unbiased Asymmetric PPO + R-NaD + Triple-Armor Safety Shield + PFSP."""

    # Dual-rate Simplex TD(lambda): round-horizon MC (lambda_intra=1.0) avoids ego-perspective critic bootstrap bias
    lambda_intra: float = 1.0         # 1.0 = exact round-end return (no intra-round Oracle bootstrapping)
    lambda_inter: float = 0.70        # Convex blend with GRP_EMA at round boundaries

    # Running-RMS advantage scaling (NO mean subtraction, active_mask only)
    running_rms_beta: float = 0.99
    running_rms_eps: float = 1e-5

    # PPO-Clip + R-NaD discrete hard-reset magnet (tuned for gentle fine-tuning over 79% BC)
    clip_eps: float = 0.20
    alpha_rnad: float = 0.08
    outer_reset_steps: int = 400      # 400 updates per outer epoch (~1.6M steps/epoch at B=32,T=128)
    reset_warmup_steps: int = 30      # Short LR warmup after AdamW m_t surgical reset
    # Dual-anchor entropy floor on rho_{k+1} = (1 - mix_bc - mix_u)*pi_k + mix_bc*pi_bc + mix_u*U_legal
    rho_mix_bc: float = 0.10
    rho_mix_uniform: float = 0.003

    # Triple-Armor Softened Support-Set Safety Shield
    bc_soft_temp: float = 1.8
    shield_tau_safe: float = 1e-3
    shield_c_max: float = 5.0
    shield_lambda_start: float = 0.05
    shield_lambda_end: float = 0.005
    shield_decay_scale: float = 0.30

    # Policy optimizer (isolated from Oracle & Belief)
    policy_lr: float = 4e-5
    policy_weight_decay: float = 1e-4
    policy_grad_clip: float = 0.5

    # Freeze Phase 0 calibrated BeliefNet during RL by default to prevent Ron-truncation poisoning
    freeze_belief: bool = True
    # Sparse static-capacity legal-ron compaction when freeze_belief=False (auto-floored to 2 * n_per_dev)
    ron_k_max: int = 8192
    # Per-rollout single-snapshot PFSP elite pool size
    pfsp_pool_size: int = 8
    utility_dropout: float = 0.20

    # Phase 1 frozen-policy warmup & Oracle readiness gates before unlocking policy updates
    # (`warmup_ece_gate` & `warmup_auc_gate` are Phase 0 GRP/Belief validation gates kept for config compatibility)
    warmup_updates: int = 100
    warmup_ece_gate: float = 0.04
    warmup_auc_gate: float = 0.85

    # Self-Healing Circuit Breaker thresholds
    audit_interval: int = 50
    max_approx_kl: float = 0.045
    min_policy_entropy: float = 0.35
    min_distinct_deck_ratio: float = 0.95
    max_birkhoff_err: float = 1e-3
    max_p_riichi_legal: float = 0.78
    riichi_probe_window: int = 5
    riichi_probe_min_samples: int = 160
    max_outer_snapshots: int = 5
    max_consecutive_heals: int = 5


@dataclass(frozen=True)
class BCConfig:
    """Phase 0-C Isomorphic Behavior Cloning config (includes frozen Delta_U + beta_risk/offense)."""

    env_name: Literal["red_4p", "no_red_4p"] = "red_4p"
    model: ModelConfig = field(default_factory=ModelConfig)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    num_epochs: int = 5
    grad_clip_norm: float = 1.0
    suit_augment: bool = True
    utility_dropout: float = 0.30
    seed: int = 0


@dataclass(frozen=True)
class RLConfig:
    """Full-hanchan PureJaxRL Sovereign-Mahjong V6 training configuration."""

    env_name: Literal["red_4p", "no_red_4p"] = "red_4p"
    round_mode: Literal["single", "east", "half"] = "half"

    model: ModelConfig = field(default_factory=ModelConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    grp: GRPConfig = field(default_factory=GRPConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    rnad: RNaDConfig = field(default_factory=RNaDConfig)

    # Single-GPU scaled rollout & optimization (B=32, T=128 -> 4096 transitions/update)
    num_envs: int = 32
    num_steps: int = 128
    num_updates: int = 12000          # ~49.2M env steps = 10 complete R-NaD outer epochs
    update_epochs: int = 2
    minibatch_size: int = 1024        # Micro-batch sized for <0.5GB dynamic activations
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    """Fixed-wall Duplicate 1v3 evaluation configuration."""

    env_name: Literal["red_4p", "no_red_4p"] = "red_4p"
    round_mode: Literal["single", "east", "half"] = "half"
    num_walls: int = 250              # Total games = 4 * num_walls
    max_steps: int = 2400
    rank_points: Tuple[float, float, float, float] = (90.0, 45.0, 0.0, -135.0)
    seed: int = 42


def make_dev_6gb_config() -> RLConfig:
    """Fast memory-friendly preset for local RTX 3060 6GB dev & smoke runs."""
    return RLConfig(
        round_mode="half",
        model=ModelConfig(num_blocks=2, channels=64, river_dim=16, attn_heads=4, global_hidden_dim=64),
        belief=BeliefConfig(num_blocks=2, channels=48, river_dim=16, attn_heads=4, global_hidden_dim=48),
        grp=GRPConfig(hidden_dim=64, num_layers=2),
        oracle=OracleConfig(num_blocks=2, channels=64, attn_heads=4, global_hidden_dim=64),
        rnad=RNaDConfig(
            outer_reset_steps=20,
            reset_warmup_steps=5,
            ron_k_max=256,
            warmup_updates=2,
            audit_interval=5,
        ),
        num_envs=16,
        num_steps=64,
        num_updates=10,
        update_epochs=2,
        minibatch_size=256,
        seed=0,
    )

"""
Central configuration for Hokm training and evaluation.

Everything that used to be a magic constant scattered through
`enhanced_player.py`, `train_backend.py`, `train_hokm.py`, etc. lives here.

Design goals:
  - A single `HokmConfig` dataclass that can be constructed with defaults
    and overridden from code, YAML, or a CLI.
  - No silent global state: callers pass the config explicitly.
  - Every knob is named, documented, and has a typed default.

The config is intentionally small — add to it when (and only when) something
needs to be tunable from outside the module where it lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from game_constants import STATE_DIM, ACTION_DIM


# -------------------------
# Reward modes
# -------------------------

REWARD_HEURISTIC = "heuristic"           # original dense shaping (pre-overhaul default)
REWARD_OUTCOME = "outcome"               # terminal-only: (tricks_diff, win bonus, +kot)
REWARD_MIXED = "mixed"                   # heuristic * shaping_weight + terminal outcome

VALID_REWARD_MODES = (REWARD_HEURISTIC, REWARD_OUTCOME, REWARD_MIXED)


@dataclass
class NetworkConfig:
    """Neural network sizes and optimizer settings.

    `state_dim` and `action_dim` are sourced from `game_constants` so that
    changes to the observation/action space (e.g. STATE_DIM 114 → 194) can
    never silently desynchronise the training pipeline from what
    `EnhancedPlayer.get_state()` actually produces. Overriding these from
    a config file is still supported but discouraged.
    """
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    q_hidden: tuple = (256, 128, 64)
    pi_hidden: tuple = (256, 256)
    q_lr: float = 1e-3
    pi_lr: float = 1e-3
    grad_clip: float = 1.0


@dataclass
class NFSPConfig:
    """Neural Fictitious Self-Play training knobs."""
    gamma: float = 0.99
    eta: float = 0.25                     # P(sample from average policy)
    epsilon_start: float = 0.15
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9995         # per-learn-step
    replay_size: int = 100_000
    sl_reservoir_size: int = 150_000
    batch_size: int = 32
    target_update_frequency: int = 200
    learn_every: int = 4                  # gradient step every N plays
    reward_mode: str = REWARD_OUTCOME
    shaping_weight: float = 0.1           # scale for heuristic signal in "mixed"
    win_bonus: float = 5.0                # terminal reward when agent's team reaches 7
    kot_bonus: float = 3.0                # additional reward for 7-0 sweep (extension)


@dataclass
class OpponentMix:
    """
    Probability mass over opponent pools for each training game (per seat,
    except the designated learner seats). Weights are normalized.

    - self_play: use the shared learner's current policy for this seat.
    - random: uniform legal-action policy.
    - heuristic: hand-written strong-ish baseline.
    - frozen_pool: sample a random frozen checkpoint from `frozen_pool_dir`.

    Note: `trainable_seats` controls which seats still learn. All seats still
    generate transitions; non-trainable seats are used for distributional
    diversity (i.e. they don't push RL transitions to the shared buffer).

    Defaults rationale: pure self-play (`trainable_seats=[0,1,2,3]`,
    `self_play=1.0`) is symmetry-poisoned — Team 1 win rate is *forced* to
    0.5 and trick differential to 0 by symmetry, so the metrics dashboard
    becomes uninformative. We default to training only Team 1 (seats 0, 2)
    while seats 1, 3 sample a mix of self-play / heuristic / random. This
    breaks the symmetry, makes the dashboard track real progress, and also
    counters the off-distribution brittleness that hits pure-self-play
    agents when they meet humans for the first time.
    """
    self_play: float = 0.5
    random: float = 0.1
    heuristic: float = 0.4
    frozen_pool: float = 0.0
    frozen_pool_dir: Optional[str] = None
    trainable_seats: List[int] = field(default_factory=lambda: [0, 2])


@dataclass
class EvalConfig:
    """Evaluation-time behavior (must be deterministic by default)."""
    num_games: int = 1000
    epsilon: float = 0.0                  # no random exploration
    eta: float = 0.0                      # no NFSP stochastic branch (pure greedy Q)
    seed: Optional[int] = 42


@dataclass
class HokmConfig:
    """Top-level config bundle."""
    seed: Optional[int] = None
    num_games: int = 1000
    model_save_interval: int = 100
    minimal_logging: bool = True
    network: NetworkConfig = field(default_factory=NetworkConfig)
    nfsp: NFSPConfig = field(default_factory=NFSPConfig)
    opponents: OpponentMix = field(default_factory=OpponentMix)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # -------------------------
    # Helpers
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HokmConfig":
        """Shallow dataclass reconstruction from a plain dict."""
        def _sub(kls, d):
            return kls(**{k: v for k, v in (d or {}).items() if k in kls.__dataclass_fields__})

        return cls(
            seed=data.get("seed"),
            num_games=data.get("num_games", cls.__dataclass_fields__["num_games"].default),
            model_save_interval=data.get(
                "model_save_interval",
                cls.__dataclass_fields__["model_save_interval"].default,
            ),
            minimal_logging=data.get("minimal_logging", True),
            network=_sub(NetworkConfig, data.get("network", {})),
            nfsp=_sub(NFSPConfig, data.get("nfsp", {})),
            opponents=_sub(OpponentMix, data.get("opponents", {})),
            eval=_sub(EvalConfig, data.get("eval", {})),
        )


DEFAULT_CONFIG = HokmConfig()

"""
Greedy, deterministic evaluation with no weight updates.

Key differences from training:
  - `learning_enabled = False` on every player  -> no `store_experience`, no `optimize_model` effects.
  - `epsilon = 0`                               -> Q-greedy branch never random-explores.
  - `eta = 0`                                   -> NFSP stochastic branch is disabled, so `select_action`
                                                    is pure argmax over legal Q-values.
  - Deterministic RNG if `seed` is provided    -> deck shuffles and tie-breaks are reproducible.

`run_evaluation` is preserved (backwards compatible with the dev console) but
now also accepts an `eta` kwarg and a `seed`. Default `eta=0.0` fixes the
previous train/eval mismatch where the average policy still sampled 25% of
the time during "evaluation".
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from enhanced_player import EnhancedPlayer
from game_constants import ACTION_DIM, STATE_DIM
from hokm import Hokm
from seed_utils import seed_all


def _make_player(name: str, checkpoint_path: Optional[str], epsilon: float, eta: float) -> EnhancedPlayer:
    p = EnhancedPlayer(name, STATE_DIM, ACTION_DIM, epsilon=epsilon, eta=eta)
    p.learning_enabled = False
    if checkpoint_path:
        p.load_policy_state(checkpoint_path)
    return p


def run_evaluation(
    num_games: int,
    checkpoint_paths: List[Optional[str]],
    *,
    epsilon: float = 0.0,
    eta: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    checkpoint_paths: length-4 list; None entries keep random init for that seat.
    Returns aggregate stats + per-game outcomes. Deterministic when `seed` is set.
    """
    if len(checkpoint_paths) != 4:
        raise ValueError("checkpoint_paths must have length 4 (one per seat)")

    # Seed everything (Python/NumPy/Torch) so that network initialization,
    # replay sampling (not used here), and shuffling are all reproducible.
    seed_all(seed)
    rng = random.Random(seed) if seed is not None else None

    players: List[EnhancedPlayer] = [
        _make_player(f"Player {i + 1}", checkpoint_paths[i], epsilon=epsilon, eta=eta)
        for i in range(4)
    ]

    game = Hokm(players, minimal_logging=True, rng=rng)
    games_out: List[Dict[str, Any]] = []
    team1_wins = 0
    team2_wins = 0

    for g in range(num_games):
        game.play_game(save_excel_log=False)
        s1, s2 = game.scores[1], game.scores[2]
        if s1 > s2:
            w = 1
            team1_wins += 1
        elif s2 > s1:
            w = 2
            team2_wins += 1
        else:
            w = 0
        games_out.append(
            {
                "index": g + 1,
                "team1_tricks": s1,
                "team2_tricks": s2,
                "winner_team": w,
            }
        )

    return {
        "num_games": num_games,
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "tie_or_incomplete": num_games - team1_wins - team2_wins,
        "games": games_out,
        "epsilon": epsilon,
        "eta": eta,
        "seed": seed,
    }

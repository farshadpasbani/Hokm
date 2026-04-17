"""Run greedy self-play evaluation without weight updates (learning disabled)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from enhanced_player import EnhancedPlayer
from game_constants import ACTION_DIM, STATE_DIM
from hokm import Hokm


def run_evaluation(
    num_games: int,
    checkpoint_paths: List[Optional[str]],
    *,
    epsilon: float = 0.0,
) -> Dict[str, Any]:
    """
    checkpoint_paths: length-4 list; None entries keep random init for that seat.
    Returns aggregate stats plus per-game outcomes.
    """
    if len(checkpoint_paths) != 4:
        raise ValueError("checkpoint_paths must have length 4 (one per seat)")

    players: List[EnhancedPlayer] = []
    for i in range(4):
        p = EnhancedPlayer(
            f"Player {i + 1}", STATE_DIM, ACTION_DIM, epsilon=epsilon
        )
        p.learning_enabled = False
        path = checkpoint_paths[i]
        if path:
            p.load_policy_state(path)
        players.append(p)

    game = Hokm(players)
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
    }

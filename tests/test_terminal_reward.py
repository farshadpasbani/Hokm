"""
Regression tests for the terminal-reward / tricks_won pipeline.

Background: `Hokm.reset_players` used to do `self.tricks_won = {...}`, which
creates a *new* dict object and silently desynchronised it from the players'
`tricks_won` attribute (set once in `__init__` / `_maybe_swap_opponents`).
The result was that `compute_terminal_reward` always saw an empty dict,
returned -win_bonus for every player every game, and training quietly
plateaued because the value function was being trained against a constant
"I lost" signal.

These tests pin the invariants that keep that bug from coming back.
"""

from __future__ import annotations

import random

from baselines import RandomAgent
from config import REWARD_OUTCOME
from enhanced_player import EnhancedPlayer
from hokm import Hokm


def _make_game(seed: int = 0) -> Hokm:
    players = [RandomAgent(f"P{i+1}", rng=random.Random(seed + i)) for i in range(4)]
    return Hokm(players, minimal_logging=True, rng=random.Random(seed))


# ------------------------------------------------------------------
# Aliasing invariants
# ------------------------------------------------------------------

def test_game_and_players_share_tricks_won_after_init():
    """At construction, every player must alias the game's tricks_won dict."""
    game = _make_game(seed=1)
    for p in game.players:
        assert p.tricks_won is game.tricks_won, (
            f"player {p.name}.tricks_won is a different object from game.tricks_won "
            "— this would break terminal reward computation"
        )


def test_alias_is_preserved_across_reset_players():
    """`reset_players` must not break the aliasing — otherwise terminal rewards
    silently collapse to -win_bonus for every player every game."""
    game = _make_game(seed=2)
    original_dict_id = id(game.tricks_won)
    for _ in range(5):
        game.reset_players()
        assert id(game.tricks_won) == original_dict_id, (
            "reset_players() replaced the dict object instead of mutating it; "
            "players still reference the old (stale) dict."
        )
        for p in game.players:
            assert p.tricks_won is game.tricks_won


def test_alias_is_preserved_across_full_games():
    """Multi-hand smoke: across several full play_game() rounds, the aliasing
    must hold so each player's compute_terminal_reward reads the right counts.
    """
    game = _make_game(seed=3)
    for _ in range(3):
        game.play_game(save_excel_log=False)
        for p in game.players:
            assert p.tricks_won is game.tricks_won
        # End-of-game: total tricks across players must equal sum of team scores.
        total = sum(game.tricks_won.values())
        assert total == game.scores[1] + game.scores[2]


# ------------------------------------------------------------------
# Terminal-reward correctness
# ------------------------------------------------------------------

def _winning_and_losing_player(game: Hokm):
    """Return (winner, loser) — one player from the winning team, one from
    the losing team — after a hand has been resolved."""
    winning_team = game.team1 if game.scores[1] >= 7 else game.team2
    losing_team = game.team2 if winning_team is game.team1 else game.team1
    return winning_team[0], losing_team[0]


def test_terminal_reward_sign_matches_actual_winner():
    """compute_terminal_reward must be POSITIVE for a player on the winning
    team and NEGATIVE for a player on the losing team. This is the invariant
    the pre-fix codebase silently violated."""
    # Use EnhancedPlayer for compute_terminal_reward; seed for determinism.
    players = [
        EnhancedPlayer(f"P{i+1}", reward_mode=REWARD_OUTCOME, win_bonus=5.0)
        for i in range(4)
    ]
    game = Hokm(players, minimal_logging=True, rng=random.Random(7))
    # A few hands so we cover the reset path.
    for _ in range(3):
        game.play_game(save_excel_log=False)
        winner, loser = _winning_and_losing_player(game)
        rw = winner.compute_terminal_reward()
        rl = loser.compute_terminal_reward()
        assert rw > 0, (
            f"winning player {winner.name} got non-positive terminal reward "
            f"{rw} (game scores: {game.scores})"
        )
        assert rl < 0, (
            f"losing player {loser.name} got non-negative terminal reward "
            f"{rl} (game scores: {game.scores})"
        )
        # Symmetry in pure self-play: rewards should be equal-and-opposite up
        # to the small ±0.1 * trick-diff term, which is bounded by ±1.3.
        assert abs(rw + rl) <= 2.6, (
            f"winner+loser terminal rewards ({rw}, {rl}) are not approximately "
            "anti-symmetric, which would point to a bookkeeping bug"
        )


def test_terminal_rewards_sum_to_zero_per_team_pair():
    """Across both members of each team, terminal rewards must be identical
    (since `compute_terminal_reward` is a function of team-aggregate tricks).
    """
    players = [
        EnhancedPlayer(f"P{i+1}", reward_mode=REWARD_OUTCOME, win_bonus=5.0)
        for i in range(4)
    ]
    game = Hokm(players, minimal_logging=True, rng=random.Random(11))
    game.play_game(save_excel_log=False)
    r0, r2 = game.team1[0].compute_terminal_reward(), game.team1[1].compute_terminal_reward()
    r1, r3 = game.team2[0].compute_terminal_reward(), game.team2[1].compute_terminal_reward()
    assert r0 == r2, "teammates on team1 must see identical terminal reward"
    assert r1 == r3, "teammates on team2 must see identical terminal reward"
    # And the two teams must be opposites (modulo the ±0.1 * diff).
    assert (r0 > 0) != (r1 > 0), "the two teams cannot both win"

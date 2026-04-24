"""Tests for baselines and agent-level invariants (no torch optimization)."""

from __future__ import annotations

import random

import pytest

from baselines import HeuristicAgent, RandomAgent
from enhanced_player import (
    REWARD_HEURISTIC,
    REWARD_OUTCOME,
    REWARD_MIXED,
    EnhancedPlayer,
)
from hokm import Hokm


def _game_with(players, seed: int = 0) -> Hokm:
    return Hokm(players, minimal_logging=True, rng=random.Random(seed))


def test_random_and_heuristic_never_revoke():
    players = [
        HeuristicAgent("H1", rng=random.Random(0)),
        RandomAgent("R1", rng=random.Random(1)),
        HeuristicAgent("H2", rng=random.Random(2)),
        RandomAgent("R2", rng=random.Random(3)),
    ]
    g = _game_with(players, seed=99)
    g.play_game(save_excel_log=False)
    # If any legality bug existed, play_game would raise in apply_play /
    # play_round. Surviving to here means legal moves throughout.
    assert g.scores[1] + g.scores[2] > 0


def test_reward_mode_outcome_zeros_per_play_reward():
    p = EnhancedPlayer("P", reward_mode=REWARD_OUTCOME)
    from game_constants import Card
    card = Card("Hearts", "Ace")
    assert p.evaluate_play(card, lead_suit="Hearts", round_num=1) == 0.0


def test_reward_mode_mixed_scales_heuristic():
    import math
    from game_constants import Card

    base = EnhancedPlayer("P", reward_mode=REWARD_HEURISTIC)
    mixed = EnhancedPlayer("M", reward_mode=REWARD_MIXED, shaping_weight=0.1)
    # Align trump for a predictable reward
    base.update_trump_suit("Hearts")
    mixed.update_trump_suit("Hearts")

    card = Card("Hearts", "Ace")
    r_base = base.evaluate_play(card, lead_suit=None, round_num=10)
    r_mix = mixed.evaluate_play(card, lead_suit=None, round_num=10)
    assert math.isclose(r_mix, 0.1 * r_base, rel_tol=1e-9)


def test_terminal_reward_sign_correct():
    """If the agent's team 'won' (7+ tricks), terminal reward should be positive."""
    p = EnhancedPlayer("P", reward_mode=REWARD_OUTCOME, win_bonus=5.0)
    other = EnhancedPlayer("Q")
    p.team = [p, EnhancedPlayer("Pm")]  # dummy teammate
    p.tricks_won = {p: 4, p.team[1]: 3, other: 3, EnhancedPlayer("X"): 3}
    # my team tricks = 7 > opp 6 → won
    r = p.compute_terminal_reward()
    assert r > 0


def test_terminal_reward_zero_in_heuristic_mode():
    p = EnhancedPlayer("P", reward_mode=REWARD_HEURISTIC)
    p.team = [p, EnhancedPlayer("Pm")]
    p.tricks_won = {p: 7, p.team[1]: 0}
    assert p.compute_terminal_reward() == 0.0

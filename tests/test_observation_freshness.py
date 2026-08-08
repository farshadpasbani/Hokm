"""
Regression test: every agent's observation at its decision point must
reflect the LIVE current trick, not a stale reference.

Guards against the bug where play_round() rebound `self.current_trick`
each trick without re-syncing player references, so seats that had not
yet played observed the previous trick's four cards (corrupting the
current-trick, lead-suit, trick-position, and current-winner blocks).
"""

import random

from baselines import RandomAgent
from game_constants import STATE_LAYOUT, card_to_index
from hokm import Hokm


def test_trick_observation_is_live_at_every_decision():
    players = [RandomAgent(f"P{i}", rng=random.Random(i)) for i in range(4)]
    game = Hokm(players, minimal_logging=True, rng=random.Random(42))
    game.start_game()
    game.choose_trump_suit()
    game.round_count = 0

    mismatches = []
    originals = {p: p.play_card for p in players}

    def instrument(p):
        def wrapped(lead_suit, selected_card=None):
            state = p.get_state().tolist()
            lo, hi = STATE_LAYOUT["current_trick"]
            seen = {i for i, v in enumerate(state[lo:hi]) if v}
            actual = {card_to_index(c) for _, c in game.current_trick}
            if seen != actual:
                mismatches.append(
                    (game.round_count, p.name, sorted(actual), sorted(seen))
                )
            return originals[p](lead_suit, selected_card)

        return wrapped

    for p in players:
        p.play_card = instrument(p)

    for _ in range(13):
        if game.scores[1] >= 7 or game.scores[2] >= 7:
            break
        game.play_round()

    assert not mismatches, (
        f"{len(mismatches)} decision points observed a stale trick; "
        f"first: {mismatches[0]}"
    )

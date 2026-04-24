"""
Rule-correctness tests for the Hokm engine.

These tests are expected to be cheap and fully deterministic — they do NOT
instantiate any PyTorch networks. They use `baselines.RandomAgent`, which
is cheap to construct (one `EnhancedPlayer.__init__` call builds networks
we simply never train), and drives decisions purely from a seeded RNG.

Run:
    pytest tests/ -q
"""

from __future__ import annotations

import random
from typing import List

import pytest

from baselines import RandomAgent
from game_constants import Card, suits
from hokm import Deck, Hokm


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_players(seed: int = 0) -> List[RandomAgent]:
    return [RandomAgent(f"P{i+1}", rng=random.Random(seed + i)) for i in range(4)]


def make_game(seed: int = 0) -> Hokm:
    players = make_players(seed)
    rng = random.Random(seed)
    return Hokm(players, minimal_logging=True, rng=rng)


# ------------------------------------------------------------------
# Deck
# ------------------------------------------------------------------

def test_deck_has_52_unique_cards():
    d = Deck()
    assert len(d.cards) == 52
    assert len(set((c.suit, c.rank) for c in d.cards)) == 52


# ------------------------------------------------------------------
# Dealing
# ------------------------------------------------------------------

def test_after_set_trump_every_player_has_13_cards():
    g = make_game(seed=1)
    g.start_game()
    # Hakem sees 5 first; nobody else has any yet
    assert len(g.hakem.hand) == 5
    for p in g.players:
        if p is not g.hakem:
            assert len(p.hand) == 0
    g.set_trump_suit("Hearts")
    for p in g.players:
        assert len(p.hand) == 13, f"{p.name} has {len(p.hand)} cards, expected 13"
    # 4 * 13 = 52 dealt, deck empty
    assert len(g.deck.cards) == 0


def test_set_trump_rejects_invalid_suit():
    g = make_game(seed=2)
    g.start_game()
    with pytest.raises(ValueError):
        g.set_trump_suit("Stars")


# ------------------------------------------------------------------
# Legal-move enforcement
# ------------------------------------------------------------------

def test_follow_suit_required_when_able():
    """
    Seat the first player with a known hand, lead a spade, and ensure
    `apply_play` refuses a non-spade when spades are available.
    """
    g = make_game(seed=3)
    g.start_game()
    g.set_trump_suit("Hearts")

    # Force a lead suit by picking first card in hakem's hand
    first = g.hakem
    lead_card = next(c for c in first.hand if c.suit != "Hearts")  # not trump
    # Ensure the NEXT seat has at least one card of lead_card.suit AND one off-suit.
    seat_order = [g.players[(g.players.index(first) + i) % 4] for i in range(1, 4)]
    for nxt in seat_order:
        same_suit = [c for c in nxt.hand if c.suit == lead_card.suit]
        off_suit = [c for c in nxt.hand if c.suit != lead_card.suit]
        if same_suit and off_suit:
            second = nxt
            break
    else:
        pytest.skip("Random deal did not produce a seat with mixed suits on try 3")

    # Simulate the lead being played
    first.hand.remove(lead_card)
    g.current_trick.append((first, lead_card))
    g.lead_suit = lead_card.suit
    g._sync_player_trick_context()

    # Illegal move: off-suit while we hold the lead suit
    illegal = next(c for c in second.hand if c.suit != lead_card.suit)
    g.trick_starter_index = g.players.index(first)
    err = g.apply_play(second, illegal)
    assert err is not None, "engine should reject off-suit play when lead suit held"


# ------------------------------------------------------------------
# Trick winner resolution
# ------------------------------------------------------------------

def _trick_from_cards(players, cards):
    """Build a current_trick list from a list of (seat_idx, Card) pairs."""
    return [(players[i], c) for i, c in cards]


def test_trump_beats_nontrump():
    g = make_game(seed=4)
    g.trump_suit = "Hearts"
    g.lead_suit = "Spades"
    g.current_trick = _trick_from_cards(
        g.players,
        [
            (0, Card("Spades", "Ace")),      # lead, highest non-trump
            (1, Card("Hearts", "2")),        # small trump
            (2, Card("Spades", "King")),     # non-trump
            (3, Card("Clubs", "Queen")),     # off-suit, non-trump
        ],
    )
    winner = g.determine_trick_winner()
    assert winner is g.players[1], "small trump should beat ace of non-trump lead"


def test_higher_trump_wins():
    g = make_game(seed=5)
    g.trump_suit = "Hearts"
    g.lead_suit = "Spades"
    g.current_trick = _trick_from_cards(
        g.players,
        [
            (0, Card("Spades", "10")),
            (1, Card("Hearts", "5")),
            (2, Card("Hearts", "Jack")),
            (3, Card("Hearts", "3")),
        ],
    )
    assert g.determine_trick_winner() is g.players[2]


def test_no_trump_highest_lead_wins():
    g = make_game(seed=6)
    g.trump_suit = "Hearts"
    g.lead_suit = "Spades"
    g.current_trick = _trick_from_cards(
        g.players,
        [
            (0, Card("Spades", "9")),
            (1, Card("Diamonds", "Ace")),    # off-suit non-trump — cannot win
            (2, Card("Spades", "King")),
            (3, Card("Clubs", "Queen")),
        ],
    )
    assert g.determine_trick_winner() is g.players[2]


# ------------------------------------------------------------------
# Full-game invariants (stochastic but seeded)
# ------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 17, 42])
def test_full_game_terminates_with_consistent_scores(seed):
    g = make_game(seed=seed)
    g.play_game(save_excel_log=False)
    s1, s2 = g.scores[1], g.scores[2]
    # One team reaches 7; the other is in [0..6].
    assert s1 >= 7 or s2 >= 7
    assert not (s1 >= 7 and s2 >= 7)
    assert 0 <= s1 + s2 <= 13

    # Per-player trick wins sum to team scores.
    t1 = sum(g.tricks_won[p] for p in g.team1)
    t2 = sum(g.tricks_won[p] for p in g.team2)
    assert t1 == s1 and t2 == s2


@pytest.mark.parametrize("seed", [7, 19])
def test_hakem_changes_sensibly_after_hand(seed):
    g = make_game(seed=seed)
    g.play_game(save_excel_log=False)
    # Hakem after rotation is on the winning team (standard behavior here).
    winning_team = g.team1 if g.scores[1] >= 7 else g.team2
    assert g.hakem in winning_team


# ------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------

def test_same_seed_gives_same_scores():
    g1 = make_game(seed=123)
    g1.play_game(save_excel_log=False)
    g2 = make_game(seed=123)
    g2.play_game(save_excel_log=False)
    assert (g1.scores[1], g1.scores[2]) == (g2.scores[1], g2.scores[2])

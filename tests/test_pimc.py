"""
Tests for the PIMC search player.

Covers: determinization validity (hand sizes, void constraints, exact card
partition), fast-simulator trick-winner equivalence against the real
engine on random tricks, and a full engine-driven game with PIMC seats.
"""

import random

import pytest

from baselines import RandomAgent
from game_constants import Card, ranks, suits
from hokm import Hokm
from pimc import (
    PIMCPlayer,
    _to_fast,
    _trick_winner_idx,
    sample_determinization,
)


def _deck():
    return [Card(s, r) for s in suits for r in ranks]


class TestDeterminization:
    def test_partition_and_sizes(self):
        rng = random.Random(1)
        deck = _deck()
        rng.shuffle(deck)
        my_hand, unseen = deck[:13], deck[13:]
        sizes = {0: 13, 1: 13, 2: 13, 3: 13}
        deal = sample_determinization(0, my_hand, sizes, unseen, {}, rng)
        assert deal is not None
        dealt = [c for cards in deal.values() for c in cards]
        assert len(dealt) == 39
        assert {(c.suit, c.rank) for c in dealt} == {
            (c.suit, c.rank) for c in unseen
        }
        assert all(len(deal[s]) == 13 for s in (1, 2, 3))

    def test_respects_voids(self):
        rng = random.Random(2)
        deck = _deck()
        rng.shuffle(deck)
        my_hand, unseen = deck[:13], deck[13:]
        sizes = {0: 13, 1: 13, 2: 13, 3: 13}
        voids = {1: {"Hearts"}, 3: {"Spades", "Clubs"}}
        for trial in range(20):
            deal = sample_determinization(0, my_hand, sizes, unseen, voids, rng)
            assert deal is not None
            assert all(c.suit != "Hearts" for c in deal[1])
            assert all(c.suit not in ("Spades", "Clubs") for c in deal[3])

    def test_unsatisfiable_returns_none(self):
        rng = random.Random(3)
        unseen = [Card("Hearts", r) for r in ranks]  # 13 hearts only
        sizes = {0: 0, 1: 5, 2: 4, 3: 4}
        voids = {1: {"Hearts"}, 2: {"Hearts"}, 3: {"Hearts"}}
        deal = sample_determinization(0, [], sizes, unseen, voids, rng)
        assert deal is None

    def test_mid_hand_sizes(self):
        rng = random.Random(4)
        deck = _deck()
        rng.shuffle(deck)
        # Mid-hand: I hold 7, others hold 7, 6, 6 → unseen = 19
        my_hand = deck[:7]
        unseen = deck[7:26]
        sizes = {0: 7, 1: 7, 2: 6, 3: 6}
        deal = sample_determinization(0, my_hand, sizes, unseen, {}, rng)
        assert deal is not None
        assert [len(deal[s]) for s in (1, 2, 3)] == [7, 6, 6]


class TestTrickWinnerEquivalence:
    def test_matches_engine_on_random_tricks(self):
        """The fast simulator's winner must equal Hokm.determine_trick_winner
        for the same four cards, across many random tricks and trumps."""
        rng = random.Random(5)
        players = [RandomAgent(f"P{i}", rng=random.Random(i)) for i in range(4)]
        g = Hokm(players, minimal_logging=True, rng=random.Random(5))
        for trial in range(300):
            deck = _deck()
            rng.shuffle(deck)
            cards = deck[:4]
            trump_name = rng.choice(suits)
            g.trump_suit = trump_name
            g.lead_suit = cards[0].suit
            g.current_trick = list(zip(players, cards))
            g.tricks_won = {p: 0 for p in players}
            engine_winner = g.determine_trick_winner()
            fast_trick = [(i, _to_fast(c)) for i, c in enumerate(cards)]
            fast_winner = _trick_winner_idx(fast_trick, suits.index(trump_name))
            assert players.index(engine_winner) == fast_winner, (
                f"trial {trial}: trick={[str(c) for c in cards]} "
                f"trump={trump_name}"
            )


class TestPIMCPlays:
    def test_full_game_with_pimc_seats(self):
        pimc = PIMCPlayer("PIMC", determinizations=6, rng=random.Random(9))
        others = [RandomAgent(f"R{i}", rng=random.Random(i)) for i in range(3)]
        g = Hokm([pimc] + others, minimal_logging=True, rng=random.Random(9))
        g.start_game()
        g.choose_trump_suit()
        g.round_count = 0
        for _ in range(400):
            if g.scores[1] >= 7 or g.scores[2] >= 7:
                break
            if all(len(p.hand) == 0 for p in g.players):
                break
            if len(g.current_trick) == 4:
                g.resolve_trick_if_complete()
                continue
            nxt = g.get_next_to_play()
            card, _ = nxt.play_card(g.lead_suit)
            err = g.apply_play(nxt, card)
            assert err is None, f"{nxt.name} illegal play: {err}"
        assert g.scores[1] + g.scores[2] > 0

    def test_pimc_follows_suit(self):
        """PIMC must never propose an off-suit card while holding lead suit."""
        pimc = PIMCPlayer("PIMC", determinizations=4, rng=random.Random(11))
        others = [RandomAgent(f"R{i}", rng=random.Random(i)) for i in range(3)]
        g = Hokm(others[:1] + [pimc] + others[1:], minimal_logging=True,
                 rng=random.Random(11))
        g.start_game()
        g.choose_trump_suit()
        checked = 0
        for _ in range(200):
            if g.scores[1] >= 7 or g.scores[2] >= 7:
                break
            if all(len(p.hand) == 0 for p in g.players):
                break
            if len(g.current_trick) == 4:
                g.resolve_trick_if_complete()
                continue
            nxt = g.get_next_to_play()
            card, _ = nxt.play_card(g.lead_suit)
            if nxt is pimc and g.lead_suit:
                has_lead = any(c.suit == g.lead_suit for c in pimc.hand)
                if has_lead:
                    assert card.suit == g.lead_suit
                    checked += 1
            err = g.apply_play(nxt, card)
            assert err is None
        assert checked > 0

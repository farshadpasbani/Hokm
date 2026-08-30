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
import pimc as pimc_mod
from pimc import (
    HAKEM_TRUMP_BIAS,
    PIMCPlayer,
    _prune_last_seat_candidates,
    _rollout,
    _rollout_policy,
    _to_fast,
    _trick_winner,
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


class TestTeamOrientation:
    """`_search` must score every rollout for the MOVER's own team.

    Seat parity picks which half of `_rollout`'s (t0, t1) counts as
    "mine": seats 0 and 2 are team 0 (the engine's Team 1), seats 1 and
    3 are team 1. Reading that backwards makes the seat play to lose —
    invisible to legality or shape tests, because every card it then
    picks is still a legal card.
    """

    TRUMP = "Spades"

    def _endgame(self, my_seat):
        """Fourth to a trick an opponent is winning, with the mover's own
        team one trick short of taking the hand.

        Two candidates, both legal follows: the Ace takes the trick and
        the hand; the deuce ducks, and the Ace then dies to a ruff on the
        last trick because every remaining opponent card is a trump. So
        the right play is unambiguous — and it inverts exactly when the
        team lens does.
        """
        me = PIMCPlayer(
            "PIMC",
            determinizations=24,
            rng=random.Random(4242),
            prune_last_seat=False,
        )
        seats = [
            me if i == my_seat else RandomAgent(f"R{i}", rng=random.Random(i))
            for i in range(4)
        ]
        game = Hokm(seats, minimal_logging=True)
        for p in seats:
            p._sync_seats(game)

        my_hand = [Card("Hearts", "Ace"), Card("Hearts", "2")]
        me.hand = my_hand
        me.trump_suit = self.TRUMP
        unseen = [Card(self.TRUMP, r) for r in ("3", "4", "5")]
        others = [s for s in range(4) if s != my_seat]
        for card, seat in zip(unseen, others):
            seats[seat].hand = [card]

        lead, partner, rho = (
            (my_seat + 1) % 4, (my_seat + 2) % 4, (my_seat + 3) % 4
        )
        on_table = {
            lead: Card("Hearts", "3"),
            partner: Card("Hearts", "4"),
            rho: Card("Hearts", "Queen"),  # the opponent before me is winning
        }
        me.current_trick = [
            (seats[s], on_table[s]) for s in (lead, partner, rho)
        ]

        live = my_hand + unseen + list(on_table.values())
        game.cards_played_this_hand = [
            Card(s, r)
            for s in suits
            for r in ranks
            if all((s, r) != (c.suit, c.rank) for c in live)
        ]
        game.trump_suit = self.TRUMP
        game.hakem = me  # I declared trump: no hakem-trump bias to model
        game.void_map = {p: set() for p in seats}
        my_team = 1 if my_seat % 2 == 0 else 2
        game.scores = {my_team: 6, 3 - my_team: 0}
        return me, game

    @pytest.mark.parametrize("my_seat", [0, 1, 2, 3])
    def test_mover_plays_to_win_its_own_teams_hand(self, my_seat):
        me, game = self._endgame(my_seat)
        assert len(game.cards_played_this_hand) == 44  # only 8 cards live
        card, _ = me.play_card("Hearts")
        assert str(card) == "Ace of Hearts", (
            f"seat {my_seat} ducked the trick that wins its own hand"
        )


# ---------------------------------------------------------------------------
# Behaviour pins.
#
# These freeze the *decisions* of the rollout machinery so that performance
# work on `_legal` / `_rollout_policy` / `_rollout` can be proven
# decision-neutral: optimizing the hot path must not move a single card.
# The expected values were captured from the pre-optimization implementation.
# ---------------------------------------------------------------------------

H, D, C, S = 0, 1, 2, 3

# (hand, trick, trump, seat) — ten hand-built decision states covering every
# branch of the greedy policy: lead, forced, duck-under-partner, min-beat,
# ruff, discard, trump lead, and the fourth-seat cases.
POLICY_CASES = [
    ([(H, 14), (H, 5), (D, 9), (D, 8), (D, 3), (S, 12)], [], S, 0),
    ([(H, 7), (D, 2), (S, 4)], [(0, (H, 11))], S, 1),
    ([(H, 13), (H, 4), (S, 6)], [(0, (H, 14)), (1, (H, 3))], S, 2),
    ([(H, 6), (H, 12), (H, 13), (S, 2)], [(1, (H, 10))], S, 2),
    ([(D, 5), (D, 9), (S, 3), (S, 14)], [(0, (C, 14)), (1, (C, 2))], S, 2),
    ([(D, 5), (D, 9), (C, 2), (S, 3)], [(1, (H, 14)), (2, (S, 10))], S, 3),
    ([(S, 4), (S, 9), (S, 13)], [], S, 0),
    ([(S, 4), (S, 9), (H, 13)], [(3, (S, 11))], S, 0),
    ([(H, 2), (H, 14), (C, 7)], [(1, (D, 9)), (2, (S, 5)), (3, (D, 12))], S, 0),
    ([(S, 2), (S, 7), (S, 14)], [(2, (D, 6)), (3, (D, 13))], S, 0),
]

POLICY_EXPECTED = [
    (1, 9),
    (0, 7),
    (0, 4),
    (0, 12),
    (1, 5),
    (2, 2),
    (3, 13),
    (3, 4),
    (0, 2),
    (3, 2),
]

# (hands, trump, next_seat, tricks_so_far) full-playout pins.
ROLLOUT_CASES = [
    ([[(0, 6), (1, 4), (1, 12), (2, 4), (3, 4), (3, 9)],
      [(0, 8), (0, 14), (1, 2), (1, 5), (1, 8), (3, 6)],
      [(0, 2), (0, 4), (2, 3), (2, 5), (3, 5), (3, 7)],
      [(1, 3), (1, 7), (1, 10), (1, 14), (2, 9), (3, 8)]], 0, 0, [0, 1]),
    ([[(0, 6), (1, 12), (3, 6), (3, 7), (3, 13)],
      [(0, 10), (1, 14), (2, 2), (2, 10), (3, 5)],
      [(0, 2), (1, 4), (1, 9), (1, 11), (2, 11)],
      [(0, 3), (0, 4), (0, 11), (0, 14), (1, 10)]], 1, 1, [1, 2]),
    ([[(0, 10), (0, 14), (1, 10), (1, 14)],
      [(0, 4), (2, 2), (2, 13), (3, 7)],
      [(1, 6), (2, 9), (2, 12), (3, 9)],
      [(0, 5), (1, 2), (1, 9), (3, 13)]], 2, 2, [2, 0]),
    ([[(1, 7), (2, 5), (3, 2)],
      [(0, 4), (1, 9), (3, 8)],
      [(1, 10), (3, 4), (3, 7)],
      [(1, 3), (2, 8), (3, 3)]], 3, 3, [0, 1]),
    ([[(0, 3), (1, 13), (2, 2), (2, 12), (3, 4), (3, 13)],
      [(0, 11), (0, 14), (1, 5), (1, 12), (2, 14), (3, 8)],
      [(1, 10), (2, 6), (2, 10), (3, 7), (3, 10), (3, 14)],
      [(0, 2), (0, 12), (2, 13), (3, 3), (3, 9), (3, 12)]], 0, 0, [1, 2]),
]

ROLLOUT_EXPECTED = [(3, 4), (5, 3), (3, 3), (2, 2), (4, 5)]

# Full `_search` pins: seat unseated (13/13/13/13), leading, seeded RNG.
SEARCH_CASES = [(1, 101, "Spades"), (2, 202, "Hearts"), (3, 303, "Diamonds")]
SEARCH_EXPECTED = ["3 of Clubs", "7 of Hearts", "Jack of Spades"]


class TestDecisionPins:
    @pytest.mark.parametrize("i", range(len(POLICY_CASES)))
    def test_rollout_policy_pinned(self, i):
        hand, trick, trump, seat = POLICY_CASES[i]
        counts = [len(hand)] * 4
        got = _rollout_policy(list(hand), list(trick), trump, seat, counts)
        assert got == POLICY_EXPECTED[i], f"case {i}: {got} != {POLICY_EXPECTED[i]}"

    @pytest.mark.parametrize("i", range(len(ROLLOUT_CASES)))
    def test_rollout_outcome_pinned(self, i):
        hands, trump, seat, tricks = ROLLOUT_CASES[i]
        got = _rollout([list(h) for h in hands], [], trump, seat, list(tricks))
        assert got == tuple(ROLLOUT_EXPECTED[i])

    @pytest.mark.parametrize("i", range(len(SEARCH_CASES)))
    def test_search_choice_pinned(self, i):
        seed, hand_seed, trump_name = SEARCH_CASES[i]
        deck = _deck()
        random.Random(hand_seed).shuffle(deck)
        p = PIMCPlayer("P", determinizations=8, rng=random.Random(seed))
        p.hand = deck[:13]
        p.trump_suit = trump_name
        p.current_trick = []
        card, _ = p.play_card(None)
        assert str(card) == SEARCH_EXPECTED[i]

    def test_rollout_never_plays_illegal_card(self):
        """Sanity net around the pins: every card the policy picks must be a
        legal follow when the seat holds the led suit."""
        rng = random.Random(77)
        for _ in range(200):
            deck = [(s, v) for s in range(4) for v in range(2, 15)]
            rng.shuffle(deck)
            hand = deck[:6]
            lead = deck[6]
            trump = rng.randrange(4)
            trick = [(3, lead)]
            got = _rollout_policy(list(hand), trick, trump, 0, [6] * 4)
            assert got in hand
            if any(c[0] == lead[0] for c in hand):
                assert got[0] == lead[0]


def _reference_policy(hand, trick, trump, seat):
    """The pre-optimization greedy policy, transcribed verbatim.

    `_policy_index` is a hand-optimized rewrite of exactly this; keeping the
    slow, obvious version here as an oracle is what lets the fast one be
    refactored freely. Randomized differential testing against it is a much
    stronger guarantee than the fixed pins above.
    """
    lead_suit = trick[0][1][0] if trick else None
    if lead_suit is None:
        legal = list(hand)
    else:
        legal = [c for c in hand if c[0] == lead_suit] or list(hand)
    if len(legal) == 1:
        return legal[0]

    if not trick:
        non_trump = [c for c in legal if c[0] != trump]
        pool = non_trump or legal
        suit_len = [0, 0, 0, 0]
        for c in hand:
            suit_len[c[0]] += 1
        best_suit = max({c[0] for c in pool}, key=lambda s: suit_len[s])
        pool_best = [c for c in pool if c[0] == best_suit]
        return max(pool_best, key=lambda c: c[1])

    win_seat, win_card = _trick_winner(trick, trump)
    if win_seat == (seat + 2) % 4:
        non_trump = [c for c in legal if c[0] != trump]
        return min(non_trump or legal, key=lambda c: c[1])

    candidates = []
    for c in legal:
        if c[0] == trump:
            if win_card[0] != trump or c[1] > win_card[1]:
                candidates.append(c)
        elif c[0] == lead_suit and win_card[0] == lead_suit and c[1] > win_card[1]:
            candidates.append(c)
    if candidates:
        return min(candidates, key=lambda c: c[1])
    non_trump = [c for c in legal if c[0] != trump]
    return min(non_trump or legal, key=lambda c: c[1])


class TestRolloutMatchesReference:
    """The fast rollout core must be decision-identical to the reference."""

    def test_policy_matches_reference_on_random_states(self):
        rng = random.Random(12345)
        for _ in range(4000):
            deck = [(s, v) for s in range(4) for v in range(2, 15)]
            rng.shuffle(deck)
            trump = rng.randrange(4)
            n = rng.randint(1, 8)
            hand = deck[:n]
            k = rng.randrange(4)
            trick = [(i, deck[n + i]) for i in range(k)]
            seat = k if k else rng.randrange(4)
            assert _rollout_policy(list(hand), list(trick), trump, seat) == (
                _reference_policy(list(hand), list(trick), trump, seat)
            ), (hand, trick, trump, seat)

    def test_full_playouts_match_reference(self):
        """Replay whole hands, stepping the reference policy by hand, and
        require the same trick counts as `_rollout`."""
        rng = random.Random(999)
        for _ in range(300):
            deck = [(s, v) for s in range(4) for v in range(2, 15)]
            rng.shuffle(deck)
            trump = rng.randrange(4)
            k = rng.randint(1, 13)
            hands = [deck[i * k:(i + 1) * k] for i in range(4)]
            seat = rng.randrange(4)
            start = [rng.randrange(7), rng.randrange(7)]

            # Reference playout.
            ref_hands = [list(h) for h in hands]
            tricks = list(start)
            trick = []
            s = seat
            while True:
                if tricks[0] >= 7 or tricks[1] >= 7:
                    break
                if all(not h for h in ref_hands) and not trick:
                    break
                card = _reference_policy(ref_hands[s], trick, trump, s)
                ref_hands[s].remove(card)
                trick.append((s, card))
                if len(trick) == 4:
                    w = _trick_winner_idx(trick, trump)
                    tricks[w % 2] += 1
                    trick = []
                    s = w
                else:
                    s = (s + 1) % 4

            got = _rollout([list(h) for h in hands], [], trump, seat, list(start))
            assert got == (tricks[0], tricks[1])


class TestHakemTrumpBias:
    """The trump declaration leaks that the hakem's hand is trump-dense;
    `bias` must move sampled deals toward that posterior without ever
    violating hand sizes or voids."""

    @staticmethod
    def _trump_share(bias, n=300, seed=17, trump="Spades", voids=None):
        rng = random.Random(seed)
        deck = _deck()
        rng.shuffle(deck)
        my_hand, unseen = deck[:13], deck[13:]
        sizes = {0: 13, 1: 13, 2: 13, 3: 13}
        unseen_trumps = sum(1 for c in unseen if c.suit == trump)
        assert unseen_trumps >= 4
        total = 0
        for _ in range(n):
            deal = sample_determinization(
                0, my_hand, sizes, unseen, voids or {}, rng, bias=bias
            )
            assert deal is not None
            assert [len(deal[s]) for s in (1, 2, 3)] == [13, 13, 13]
            total += sum(1 for c in deal[1] if c.suit == trump)
        return total / n / unseen_trumps

    def test_biased_seat_gets_more_trump(self):
        uniform = self._trump_share(None)
        biased = self._trump_share({1: {"Spades": 2.0}})
        # Uniform sampling splits the unseen trumps three ways.
        assert 0.28 < uniform < 0.39, uniform
        # A 2x weight must be clearly visible, not a coin-flip difference.
        assert biased > uniform + 0.07, (uniform, biased)
        assert biased < 0.60, biased

    def test_bias_is_monotone_in_weight(self):
        shares = [
            self._trump_share(None if w == 1.0 else {1: {"Spades": w}})
            for w in (1.0, 2.0, 3.0)
        ]
        assert shares[0] < shares[1] < shares[2], shares

    def test_bias_never_breaks_voids(self):
        """Biasing is a preference, not an override: a void seat still gets
        no trump even when it is the biased seat."""
        voids = {1: {"Spades"}}
        rng = random.Random(23)
        deck = _deck()
        rng.shuffle(deck)
        my_hand, unseen = deck[:13], deck[13:]
        sizes = {0: 13, 1: 13, 2: 13, 3: 13}
        for _ in range(40):
            deal = sample_determinization(
                0, my_hand, sizes, unseen, voids, rng,
                bias={1: {"Spades": 5.0}},
            )
            assert deal is not None
            assert all(c.suit != "Spades" for c in deal[1])
            assert [len(deal[s]) for s in (1, 2, 3)] == [13, 13, 13]

    def test_other_suits_give_way(self):
        """Weighting trump toward one seat must not deal it extra of
        everything — the hand size is fixed, so non-trump must shrink."""
        rng = random.Random(31)
        deck = _deck()
        rng.shuffle(deck)
        my_hand, unseen = deck[:13], deck[13:]
        sizes = {0: 13, 1: 13, 2: 13, 3: 13}
        non_trump = 0
        for _ in range(200):
            deal = sample_determinization(
                0, my_hand, sizes, unseen, {}, rng,
                bias={1: {"Spades": 2.0}},
            )
            non_trump += sum(1 for c in deal[1] if c.suit != "Spades")
        assert non_trump / 200 < 13 * 0.90

    def test_player_builds_bias_for_hakem_seat(self):
        pimc = PIMCPlayer("PIMC", determinizations=2, rng=random.Random(3))
        others = [RandomAgent(f"R{i}", rng=random.Random(i)) for i in range(3)]
        g = Hokm([pimc] + others, minimal_logging=True, rng=random.Random(3))
        g.start_game()
        g.choose_trump_suit()
        bias = pimc._hakem_bias(g, 0, g.trump_suit)
        if g.hakem is pimc:
            assert bias is None          # nothing to infer about myself
        else:
            seat = g.players.index(g.hakem)
            assert bias == {seat: {g.trump_suit: pytest.approx(HAKEM_TRUMP_BIAS)}}

    def test_no_bias_when_unseated(self):
        pimc = PIMCPlayer("PIMC", determinizations=2, rng=random.Random(3))
        assert pimc._hakem_bias(None, 0, "Spades") is None


class TestLastSeatPruning:
    """Fourth to a trick, two legal cards of the same suit with the same
    trick result are interchangeable — the cheaper one dominates. Pruning
    must drop exactly those and nothing else."""

    @staticmethod
    def _cards(*specs):
        return [Card(s, r) for s, r in specs]

    @staticmethod
    def _beats(card, trick, trump):
        win = _trick_winner(trick, trump)[1]
        lead = trick[0][1][0]
        s, v = _to_fast(card)
        if s == trump:
            return win[0] != trump or v > win[1]
        return s == lead and win[0] == lead and v > win[1]

    def test_cannot_beat_keeps_only_dumps(self):
        trump = suits.index("Spades")
        # Winner is the Ace of Hearts (lead suit); I hold only losing hearts.
        trick = [(0, (H, 14)), (1, (H, 3)), (2, (H, 4))]
        valid = self._cards(("Hearts", "5"), ("Hearts", "9"), ("Hearts", "King"))
        out = _prune_last_seat_candidates(valid, trick, trump)
        assert [str(c) for c in out] == ["5 of Hearts"]

    def test_cannot_beat_offers_trump_and_non_trump_dump(self):
        trump = suits.index("Spades")
        # An opponent already ruffed high; I am void in hearts, so every
        # card is legal but none of them wins.
        trick = [(0, (H, 14)), (1, (S, 13)), (2, (H, 4))]
        valid = self._cards(
            ("Diamonds", "7"), ("Diamonds", "Queen"),
            ("Spades", "2"), ("Spades", "10"),
        )
        out = _prune_last_seat_candidates(valid, trick, trump)
        assert {str(c) for c in out} == {"7 of Diamonds", "2 of Spades"}

    def test_winners_are_minimal_plus_one_dump(self):
        trump = suits.index("Spades")
        # Winner: 10 of Hearts. I can overtake in hearts or ruff.
        trick = [(0, (H, 10)), (1, (H, 3)), (2, (H, 4))]
        valid = self._cards(
            ("Hearts", "2"), ("Hearts", "Jack"), ("Hearts", "Ace"),
            ("Spades", "3"), ("Spades", "King"),
        )
        out = _prune_last_seat_candidates(valid, trick, trump)
        # Cheapest heart winner, cheapest ruff, cheapest discard; the Ace
        # and the King of trump are dominated.
        assert {str(c) for c in out} == {
            "Jack of Hearts", "3 of Spades", "2 of Hearts",
        }

    def test_pruned_set_is_always_a_legal_subset(self):
        """Random fourth-seat positions: output must be a non-empty subset
        of the legal cards and must never drop every winning option."""
        rng = random.Random(101)
        checked = 0
        for _ in range(400):
            deck = _deck()
            rng.shuffle(deck)
            trump_name = rng.choice(suits)
            trump = suits.index(trump_name)
            table = deck[:3]
            trick = [(i, _to_fast(c)) for i, c in enumerate(table)]
            lead = table[0].suit
            hand = deck[3:3 + rng.randint(2, 8)]
            valid = [c for c in hand if c.suit == lead] or hand
            out = _prune_last_seat_candidates(valid, trick, trump)
            assert out, "pruning must never empty the candidate set"
            assert all(any(c is v for v in valid) for c in out)
            # Exactly one survivor per (suit, wins?) class, and it is the
            # cheapest of its class — i.e. only dominated cards are dropped.
            classes = {}
            for c in valid:
                classes.setdefault(
                    (c.suit, self._beats(c, trick, trump)), []
                ).append(c)
            assert len(out) == len(classes)
            for c in out:
                peers = classes[(c.suit, self._beats(c, trick, trump))]
                assert c.value == min(p.value for p in peers), (
                    f"kept dominated card {c}"
                )
            if any(self._beats(c, trick, trump) for c in valid):
                checked += 1
                assert any(self._beats(c, trick, trump) for c in out), (
                    "a winning option must always survive"
                )
        assert checked > 50

    def test_search_only_rolls_out_pruned_candidates(self, monkeypatch):
        """End-to-end: in a real game the fourth-seat search feeds the
        rollouts only winners + one dump, and still plays a legal card."""
        seen = []
        real = pimc_mod._prune_last_seat_candidates

        def spy(valid, fast_trick, trump):
            out = real(valid, fast_trick, trump)
            seen.append((list(valid), list(fast_trick), trump, list(out)))
            return out

        monkeypatch.setattr(pimc_mod, "_prune_last_seat_candidates", spy)

        player = PIMCPlayer("PIMC", determinizations=4, rng=random.Random(5))
        others = [RandomAgent(f"R{i}", rng=random.Random(i)) for i in range(3)]
        g = Hokm(others + [player], minimal_logging=True, rng=random.Random(5))
        g.start_game()
        g.choose_trump_suit()
        for _ in range(300):
            if g.scores[1] >= 7 or g.scores[2] >= 7:
                break
            if all(len(p.hand) == 0 for p in g.players):
                break
            if len(g.current_trick) == 4:
                g.resolve_trick_if_complete()
                continue
            nxt = g.get_next_to_play()
            card, _ = nxt.play_card(g.lead_suit)
            if nxt is player and g.lead_suit:
                if any(c.suit == g.lead_suit for c in player.hand):
                    assert card.suit == g.lead_suit
            assert g.apply_play(nxt, card) is None

        assert seen, "no fourth-seat decision was pruned"
        pruned_any = False
        for valid, trick, trump, out in seen:
            assert out and len(out) <= len(valid)
            assert all(any(c is v for v in valid) for c in out)
            pruned_any |= len(out) < len(valid)
            for c in out:
                peers = [
                    o for o in valid
                    if o.suit == c.suit
                    and self._beats(o, trick, trump)
                    == self._beats(c, trick, trump)
                ]
                assert c.value == min(p.value for p in peers)
            if any(self._beats(c, trick, trump) for c in valid):
                assert any(self._beats(c, trick, trump) for c in out)
        assert pruned_any, "pruning never actually removed a candidate"

    def test_flag_off_searches_every_candidate(self, monkeypatch):
        """Negative control for the test above: same seating and seed, but
        with the flag off nothing is ever pruned."""
        calls = []
        monkeypatch.setattr(
            pimc_mod, "_prune_last_seat_candidates",
            lambda *a: calls.append(a) or list(a[0]),
        )
        player = PIMCPlayer(
            "PIMC", determinizations=4, rng=random.Random(5),
            prune_last_seat=False,
        )
        others = [RandomAgent(f"R{i}", rng=random.Random(i)) for i in range(3)]
        g = Hokm(others + [player], minimal_logging=True, rng=random.Random(5))
        g.start_game()
        g.choose_trump_suit()
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
            assert g.apply_play(nxt, card) is None
        assert calls == []

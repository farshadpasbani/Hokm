"""
Tests for the 194-dim observation space.

These tests don't train anything — they verify that *after* a few card plays,
the feature slices that `get_state()` produces actually contain the
information they're supposed to. If any of these regress, the corresponding
lemma is no longer learnable even in principle.

See `game_constants.STATE_LAYOUT` for the canonical slice map.
"""

from __future__ import annotations

import random
from typing import List

import pytest

from baselines import RandomAgent
from game_constants import (
    ACTION_DIM,
    Card,
    STATE_DIM,
    STATE_LAYOUT,
    card_to_index,
    suits,
)
from hokm import Hokm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_game(seed: int = 0) -> Hokm:
    players = [RandomAgent(f"P{i+1}", rng=random.Random(seed + i)) for i in range(4)]
    return Hokm(players, minimal_logging=True, rng=random.Random(seed))


def _slice(vec: List[float], key: str) -> List[float]:
    lo, hi = STATE_LAYOUT[key]
    return list(vec[lo:hi])


def _state_of(player) -> List[float]:
    return player.get_state().tolist()


# ------------------------------------------------------------------
# Shape / layout sanity
# ------------------------------------------------------------------

def test_state_dim_constant_matches_vector():
    g = _make_game(seed=0)
    g.start_game()
    g.set_trump_suit("Hearts")
    vec = _state_of(g.players[0])
    assert len(vec) == STATE_DIM == 194


def test_state_layout_slices_are_contiguous_and_cover_full_vector():
    covered = sorted(STATE_LAYOUT.values())
    # must start at 0, end at STATE_DIM, and be gap-free
    assert covered[0][0] == 0
    assert covered[-1][1] == STATE_DIM
    for (a, b), (c, d) in zip(covered, covered[1:]):
        assert b == c, f"gap between {(a,b)} and {(c,d)}"


# ------------------------------------------------------------------
# Block 1: hand one-hot
# ------------------------------------------------------------------

def test_hand_block_one_hot_matches_player_hand():
    g = _make_game(seed=1)
    g.start_game()
    g.set_trump_suit("Hearts")
    p = g.players[0]
    vec = _slice(_state_of(p), "hand")
    assert sum(vec) == len(p.hand)
    for c in p.hand:
        assert vec[card_to_index(c)] == 1


# ------------------------------------------------------------------
# Block 2: cards_played — lemma #1 (high-card promotion)
# ------------------------------------------------------------------

def test_cards_played_block_reflects_played_cards_after_one_trick():
    g = _make_game(seed=2)
    g.start_game()
    g.set_trump_suit("Hearts")
    played_before = sum(_slice(_state_of(g.players[0]), "cards_played"))
    assert played_before == 0, "no cards played yet"

    g.play_round()

    # Exactly the cards from the now-finished trick should be flagged.
    for p in g.players:
        vec = _slice(_state_of(p), "cards_played")
        assert sum(vec) == 4, f"{p.name} sees {sum(vec)} played cards, expected 4"
        for card in g.cards_played_this_hand:
            assert vec[card_to_index(card)] == 1, (
                f"{p.name} missed card {card} in cards_played slice"
            )


# ------------------------------------------------------------------
# Block 3: void flags — lemma #2
# ------------------------------------------------------------------

def test_void_flag_set_when_opponent_fails_to_follow_suit():
    """
    Force a deterministic situation: seat 0 leads a suit seat 1 doesn't hold,
    so seat 1 must play off-suit → seat 1 becomes void in the lead suit.
    All other seats' views of seat 1's void should flip on.
    """
    g = _make_game(seed=11)
    g.start_game()
    g.set_trump_suit("Hearts")

    # Stack the deck: give seat 0 a spade, and remove all spades from seat 1.
    s0, s1 = g.players[0], g.players[1]

    # Find a spade seat 0 holds; if none, swap one in from seat 2's hand.
    if not any(c.suit == "Spades" for c in s0.hand):
        donor = next(p for p in g.players[2:] if any(c.suit == "Spades" for c in p.hand))
        spade = next(c for c in donor.hand if c.suit == "Spades")
        donor.hand.remove(spade)
        giveaway = s0.hand.pop()
        s0.hand.append(spade)
        donor.hand.append(giveaway)

    # Remove every spade from seat 1 (swap them to seat 3).
    s3 = g.players[3]
    s1_spades = [c for c in s1.hand if c.suit == "Spades"]
    for sp in s1_spades:
        s1.hand.remove(sp)
        swap = next((c for c in s3.hand if c.suit != "Spades"), None)
        if swap is None:
            pytest.skip("Could not construct a no-spade hand for seat 1 deterministically")
        s3.hand.remove(swap)
        s3.hand.append(sp)
        s1.hand.append(swap)

    # Lead a spade from seat 0.
    lead_spade = next(c for c in s0.hand if c.suit == "Spades")
    g.trick_starter_index = 0
    err = g.apply_play(s0, lead_spade)
    assert err is None, err

    # Seat 1 must now play a non-spade.
    off = next(c for c in s1.hand if c.suit != "Spades")
    err = g.apply_play(s1, off)
    assert err is None, err

    # Every player's void_map view of seat 1 now includes Spades.
    # Inspect via one representative (seat 2, an opponent of seat 1).
    p2 = g.players[2]
    voids = _slice(_state_of(p2), "voids")
    # Layout: 3 other players × 4 suits, order [RHO, partner, LHO] relative.
    # From seat 2's POV: RHO=seat 1, partner=seat 0, LHO=seat 3.
    rho_block = voids[0:4]
    spades_idx = suits.index("Spades")
    assert rho_block[spades_idx] == 1, (
        f"seat 2 should see seat 1 (RHO) void in spades; voids={voids}"
    )


# ------------------------------------------------------------------
# Block 6/7/8: trick position + winner seat + winner value — lemmas #3, #4
# ------------------------------------------------------------------

def test_trick_position_is_one_hot_and_correct_for_leader():
    g = _make_game(seed=3)
    g.start_game()
    g.set_trump_suit("Hearts")
    # Leader sees position = 1st (index 0).
    pos = _slice(_state_of(g.hakem), "trick_position")
    assert pos == [1, 0, 0, 0]


def test_winner_seat_marks_partner_when_partner_leads_winning_card():
    """
    After partner plays the opening lead, I'm 3rd to play and the current
    winner should be tagged as 'partner' in my state.
    """
    g = _make_game(seed=4)
    g.start_game()
    g.set_trump_suit("Hearts")

    # Seats 0 and 2 are partners; seats 1 and 3 are partners.
    p0, p1, p2, p3 = g.players
    lead_card = p0.hand[0]
    g.trick_starter_index = 0
    assert g.apply_play(p0, lead_card) is None

    # p1 follows legally
    legal_p1 = g.legal_cards_for_player(p1)
    assert g.apply_play(p1, legal_p1[0]) is None

    # From p2's POV seating is: p0 = partner (across), p1 = RHO (plays just
    # before me), p3 = LHO (plays just after me). So the current winner is
    # either p0 (partner, slot 2) or p1 (RHO, slot 4), never LHO (slot 3)
    # because p3 hasn't played yet.
    vec = _state_of(p2)
    winner_block = _slice(vec, "winner_seat")
    # one-hot invariant
    assert sum(winner_block) == 1, f"winner_seat not one-hot: {winner_block}"
    # layout: [empty, me, partner, LHO, RHO]
    assert winner_block[0] == 0, "empty flag must be 0 once trick has cards"
    assert winner_block[1] == 0, "'me' flag must be 0 since p2 hasn't played"
    assert winner_block[3] == 0, "LHO (p3) cannot be winning; it hasn't played"
    assert winner_block[2] + winner_block[4] == 1, (
        f"winner must be partner (slot 2) or RHO (slot 4); got {winner_block}"
    )
    val = _slice(vec, "winner_value")[0]
    assert 0.0 < val <= 1.0


# ------------------------------------------------------------------
# Block 9: Hakem flags — lemma #7
# ------------------------------------------------------------------

def test_hakem_is_me_flag_set_for_hakem_and_partner_flag_for_teammate():
    g = _make_game(seed=5)
    g.start_game()
    g.set_trump_suit("Hearts")

    hakem_vec = _state_of(g.hakem)
    assert _slice(hakem_vec, "hakem_is_me") == [1]
    assert _slice(hakem_vec, "hakem_is_partner") == [0]

    # Hakem's partner sits at (seat + 2) % 4
    seat = g.players.index(g.hakem)
    partner = g.players[(seat + 2) % 4]
    partner_vec = _state_of(partner)
    assert _slice(partner_vec, "hakem_is_me") == [0]
    assert _slice(partner_vec, "hakem_is_partner") == [1]


# ------------------------------------------------------------------
# Block 11/12: trump one-hot and hand-suit-counts — lemma #6
# ------------------------------------------------------------------

def test_trump_slice_matches_trump_suit():
    g = _make_game(seed=6)
    g.start_game()
    g.set_trump_suit("Diamonds")
    vec = _slice(_state_of(g.players[0]), "trump")
    assert vec[suits.index("Diamonds")] == 1
    assert sum(vec) == 1


def test_hand_suit_counts_match_hand():
    g = _make_game(seed=7)
    g.start_game()
    g.set_trump_suit("Hearts")
    p = g.players[0]
    vec = _slice(_state_of(p), "hand_suit_counts")
    # denormalise back to integers
    counts = [round(x * 13) for x in vec]
    for s_idx, s in enumerate(suits):
        assert counts[s_idx] == sum(1 for c in p.hand if c.suit == s)
    assert sum(counts) == len(p.hand) == 13

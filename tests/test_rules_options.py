"""
Rule-authenticity options and engine edge cases.

Covers:
  * `Hokm(hakem_stays_on_win=...)` — traditional vs. legacy Hakem rotation
    (RULES.md §8).
  * Kot detection (`Hokm.is_kot()` / `Hokm.last_hand_kot`, RULES.md §7).
  * `update_last_winning_team()` on aborted / tied hands.
  * `EnhancedPlayer._can_win_trick` under real Hokm trick resolution
    (RULES.md §6).

All tests construct states directly — no full games are simulated — so the
suite stays cheap and fully deterministic.

Run:
    pytest tests/ -q
"""

from __future__ import annotations

import random
from typing import List

import pytest

from baselines import RandomAgent
from game_constants import Card
from hokm import Hokm


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_players(seed: int = 0) -> List[RandomAgent]:
    return [RandomAgent(f"P{i+1}", rng=random.Random(seed + i)) for i in range(4)]


def make_game(seed: int = 0, *, hakem_stays_on_win: bool = False) -> Hokm:
    """A Hokm instance with no cards dealt — we drive its state directly."""
    return Hokm(
        make_players(seed),
        minimal_logging=True,
        rng=random.Random(seed),
        hakem_stays_on_win=hakem_stays_on_win,
    )


def set_tricks(g: Hokm, team1_tricks: int, team2_tricks: int) -> None:
    """Drive per-player trick counts (and the mirrored team scores) directly."""
    g.tricks_won[g.team1[0]] = team1_tricks
    g.tricks_won[g.team1[1]] = 0
    g.tricks_won[g.team2[0]] = team2_tricks
    g.tricks_won[g.team2[1]] = 0
    g.scores = {1: team1_tricks, 2: team2_tricks}


# ==================================================================
# 1. Hakem rotation — traditional mode (hakem_stays_on_win=True)
# ==================================================================

@pytest.mark.parametrize("hakem_seat", [0, 2])
def test_traditional_winning_hakem_keeps_hakemship_team1(hakem_seat):
    g = make_game(hakem_stays_on_win=True)
    g.hakem = g.players[hakem_seat]
    g.last_winning_team = g.team1  # Hakem's own team won
    g.rotate_hakem()
    assert g.hakem is g.players[hakem_seat]


@pytest.mark.parametrize("hakem_seat", [1, 3])
def test_traditional_winning_hakem_keeps_hakemship_team2(hakem_seat):
    g = make_game(hakem_stays_on_win=True)
    g.hakem = g.players[hakem_seat]
    g.last_winning_team = g.team2
    g.rotate_hakem()
    assert g.hakem is g.players[hakem_seat]


@pytest.mark.parametrize(
    "hakem_seat,expected_seat",
    [
        # Losing Hakem on team 1 (seats 0/2); team 2 (seats 1/3) won.
        # New Hakem = first winning-team seat clockwise after the old Hakem.
        (0, 1),
        (2, 3),
    ],
)
def test_traditional_losing_hakem_passes_to_next_winning_seat_team2_wins(
    hakem_seat, expected_seat
):
    g = make_game(hakem_stays_on_win=True)
    g.hakem = g.players[hakem_seat]
    g.last_winning_team = g.team2
    g.rotate_hakem()
    assert g.hakem is g.players[expected_seat]
    # Explicitly NOT team[0] — that would be seat 1 in both cases.
    if hakem_seat == 2:
        assert g.hakem is not g.team2[0]


@pytest.mark.parametrize(
    "hakem_seat,expected_seat",
    [
        # Losing Hakem on team 2 (seats 1/3); team 1 (seats 0/2) won.
        (1, 2),
        (3, 0),
    ],
)
def test_traditional_losing_hakem_passes_to_next_winning_seat_team1_wins(
    hakem_seat, expected_seat
):
    g = make_game(hakem_stays_on_win=True)
    g.hakem = g.players[hakem_seat]
    g.last_winning_team = g.team1
    g.rotate_hakem()
    assert g.hakem is g.players[expected_seat]
    if hakem_seat == 1:
        # team1[0] is seat 0; the correct successor here is seat 2.
        assert g.hakem is not g.team1[0]


def test_traditional_mode_default_is_off():
    g = make_game()
    assert g.hakem_stays_on_win is False


# ==================================================================
# 2. Hakem rotation — legacy mode (default) must be unchanged
# ==================================================================

@pytest.mark.parametrize(
    "hakem_seat,winner,expected_seat",
    [
        # Hakem's team won -> toggle within the team.
        (0, 1, 2),
        (2, 1, 0),
        (1, 2, 3),
        (3, 2, 1),
        # Hakem's team lost -> winning_team[0] (seat 0 for team 1, seat 1 for team 2).
        (0, 2, 1),
        (2, 2, 1),
        (1, 1, 0),
        (3, 1, 0),
    ],
)
def test_legacy_rotation_matrix(hakem_seat, winner, expected_seat):
    g = make_game()
    g.hakem = g.players[hakem_seat]
    g.last_winning_team = g.team1 if winner == 1 else g.team2
    g.rotate_hakem()
    assert g.hakem is g.players[expected_seat]


def test_legacy_and_traditional_differ_only_on_a_hakem_team_win():
    """The two modes agree on losses and disagree on wins."""
    for hakem_seat in range(4):
        for winner in (1, 2):
            legacy = make_game()
            trad = make_game(hakem_stays_on_win=True)
            for g in (legacy, trad):
                g.hakem = g.players[hakem_seat]
                g.last_winning_team = g.team1 if winner == 1 else g.team2
                g.rotate_hakem()
            legacy_seat = legacy.players.index(legacy.hakem)
            trad_seat = trad.players.index(trad.hakem)
            hakem_team_won = (hakem_seat % 2 == 0) == (winner == 1)
            if hakem_team_won:
                assert trad_seat == hakem_seat
                assert legacy_seat != hakem_seat
            else:
                # Loss: legacy takes winning_team[0], traditional takes the
                # next winning-team seat clockwise. They coincide only when
                # those happen to be the same seat.
                assert trad_seat == (hakem_seat + 1) % 4


# ==================================================================
# 3. Kot detection
# ==================================================================

@pytest.mark.parametrize(
    "t1,t2,expected",
    [
        (7, 0, True),    # team 1 kot
        (0, 7, True),    # team 2 kot
        (13, 0, True),   # full sweep (still a kot)
        (7, 1, False),
        (7, 6, False),
        (1, 7, False),
        (0, 0, False),   # nothing played yet
        (5, 0, False),   # unfinished hand, not a kot yet
    ],
)
def test_is_kot_live_view(t1, t2, expected):
    g = make_game()
    set_tricks(g, t1, t2)
    assert g.is_kot() is expected


def test_kot_flag_does_not_change_scores_or_winner():
    g = make_game()
    set_tricks(g, 7, 0)
    assert g.is_kot() is True
    g.update_last_winning_team()
    assert g.last_winning_team is g.team1
    assert g.scores == {1: 7, 2: 0}


def _resolve_one_trick(g: Hokm, winning_seat: int) -> None:
    """Drive a full 4-card trick through the step API, won by `winning_seat`."""
    g.trump_suit = "Hearts"
    g.lead_suit = "Spades"
    values = ["2", "3", "4", "5"]
    values[winning_seat] = "Ace"
    g.current_trick = [(g.players[i], Card("Spades", values[i])) for i in range(4)]
    winner, _ = g.resolve_trick_if_complete()
    assert winner is g.players[winning_seat]


# ---- latch semantics ---------------------------------------------

def test_last_hand_kot_is_false_before_any_hand_completes():
    g = make_game()
    assert g.last_hand_kot is False
    # A mid-hand 7-0-looking score is *not* latched until a hand completes.
    set_tricks(g, 3, 0)
    assert g.last_hand_kot is False


def test_kot_latched_on_step_api_path():
    """
    The incremental (web-app) path never calls play_game(); the latch is set
    by resolve_trick_if_complete() when the trick it just resolved ends the
    hand.
    """
    g = make_game()
    set_tricks(g, 6, 0)
    assert g.last_hand_kot is False  # hand not over yet
    _resolve_one_trick(g, winning_seat=0)  # 7-0
    assert g.scores == {1: 7, 2: 0}
    assert g.is_kot() is True
    assert g.last_hand_kot is True


def test_step_api_non_kot_hand_latches_false():
    g = make_game()
    set_tricks(g, 6, 2)
    _resolve_one_trick(g, winning_seat=0)  # 7-2
    assert g.is_kot() is False
    assert g.last_hand_kot is False


def test_latch_survives_start_game():
    """The whole point: the value must cross the hand boundary."""
    g = make_game()
    set_tricks(g, 6, 0)
    _resolve_one_trick(g, winning_seat=0)  # 7-0 kot
    assert g.last_hand_kot is True

    g.start_game()  # resets scores/hands for the next hand
    assert g.scores == {1: 0, 2: 0}
    assert g.is_kot() is False, "live view follows the new (empty) hand"
    assert g.last_hand_kot is True, "latched value must survive the reset"


def test_second_hand_overwrites_latched_value():
    g = make_game()
    set_tricks(g, 6, 0)
    _resolve_one_trick(g, winning_seat=0)  # hand 1: 7-0 kot
    assert g.last_hand_kot is True

    g.start_game()
    set_tricks(g, 3, 6)
    _resolve_one_trick(g, winning_seat=1)  # hand 2: 3-7, not a kot
    assert g.scores == {1: 3, 2: 7}
    assert g.last_hand_kot is False, "hand 2 must overwrite hand 1's latch"

    g.start_game()
    set_tricks(g, 0, 6)
    _resolve_one_trick(g, winning_seat=1)  # hand 3: 0-7 kot for team 2
    assert g.last_hand_kot is True


def test_latch_set_by_play_game_path():
    g = make_game(seed=3)
    g.play_game(save_excel_log=False)
    assert g.last_hand_kot is g.is_kot()
    g.start_game()
    assert g.is_kot() is False
    # Whatever play_game latched is still readable after the next start_game.
    assert isinstance(g.last_hand_kot, bool)


def test_aborted_hand_does_not_clobber_latch():
    """
    An aborted hand never completed, so it must leave the previous hand's Kot
    value alone.
    """
    g = make_game()
    set_tricks(g, 6, 0)
    _resolve_one_trick(g, winning_seat=0)  # real 7-0 kot
    assert g.last_hand_kot is True

    g.start_game()
    # Simulate an abort partway through: scores are partial and nobody hit 7,
    # and the players still hold cards.
    g.scores = {1: 5, 2: 4}
    assert g._hand_is_complete() is False
    g._latch_kot_if_hand_complete()
    assert g.last_hand_kot is True, "incomplete hand must not overwrite the latch"


def test_last_hand_kot_is_read_only():
    g = make_game()
    with pytest.raises(AttributeError):
        g.last_hand_kot = True


# ==================================================================
# 4. update_last_winning_team
# ==================================================================

def test_update_last_winning_team_normal_hand_team1():
    g = make_game()
    set_tricks(g, 7, 3)
    g.update_last_winning_team()
    assert g.last_winning_team is g.team1


def test_update_last_winning_team_normal_hand_team2():
    g = make_game()
    set_tricks(g, 3, 7)
    g.update_last_winning_team()
    assert g.last_winning_team is g.team2


def test_update_last_winning_team_aborted_hand_credits_trick_leader():
    """Regression: an aborted 5-4 hand used to silently credit team 2."""
    g = make_game()
    g.last_winning_team = g.team2  # make sure we're not just seeing the default
    set_tricks(g, 5, 4)
    g.update_last_winning_team()
    assert g.last_winning_team is g.team1


def test_update_last_winning_team_aborted_hand_team2_ahead():
    g = make_game()
    g.last_winning_team = g.team1
    set_tricks(g, 4, 5)
    g.update_last_winning_team()
    assert g.last_winning_team is g.team2


@pytest.mark.parametrize("previous", ["team1", "team2"])
def test_update_last_winning_team_tie_keeps_previous(previous):
    g = make_game()
    g.last_winning_team = g.team1 if previous == "team1" else g.team2
    expected = g.last_winning_team
    set_tricks(g, 4, 4)
    g.update_last_winning_team()
    assert g.last_winning_team is expected


def test_update_last_winning_team_defaults_to_team1_when_never_set():
    g = make_game()
    set_tricks(g, 0, 0)
    g.update_last_winning_team()
    assert g.last_winning_team is g.team1


# ==================================================================
# 5. _can_win_trick
# ==================================================================

def _make_actor(g: Hokm, seat: int, trump: str, trick):
    """Return the player at `seat` with trump + in-progress trick installed."""
    p = g.players[seat]
    p.update_trump_suit(trump)
    p.current_trick = [(g.players[i], c) for i, c in trick]
    return p


def test_can_win_trick_true_when_leading():
    g = make_game()
    p = _make_actor(g, 0, "Hearts", [])
    assert p._can_win_trick(Card("Clubs", "2"), None) is True


def test_offsuit_ace_cannot_beat_a_trump():
    """A trump is on the table: no off-suit card, however high, can win."""
    g = make_game()
    p = _make_actor(
        g,
        3,
        "Hearts",
        [(0, Card("Spades", "King")), (1, Card("Hearts", "2"))],
    )
    # The old max-by-value implementation compared against the King of Spades
    # and wrongly reported True here.
    assert p._can_win_trick(Card("Spades", "Ace"), "Spades") is False
    assert p._can_win_trick(Card("Diamonds", "Ace"), "Spades") is False


def test_low_trump_beats_offsuit_ace():
    g = make_game()
    p = _make_actor(
        g,
        1,
        "Hearts",
        [(0, Card("Spades", "Ace"))],
    )
    # Old implementation: highest_card = Spades Ace (14) > Hearts 2 -> False.
    assert p._can_win_trick(Card("Hearts", "2"), "Spades") is True


def test_higher_trump_needed_to_beat_a_trump():
    g = make_game()
    p = _make_actor(
        g,
        2,
        "Hearts",
        [(0, Card("Spades", "King")), (1, Card("Hearts", "9"))],
    )
    assert p._can_win_trick(Card("Hearts", "10"), "Spades") is True
    assert p._can_win_trick(Card("Hearts", "8"), "Spades") is False
    assert p._can_win_trick(Card("Hearts", "9"), "Spades") is False  # can't tie


def test_no_trump_must_follow_lead_to_win():
    g = make_game()
    p = _make_actor(
        g,
        3,
        "Hearts",
        [(0, Card("Spades", "9")), (1, Card("Clubs", "Ace")), (2, Card("Spades", "10"))],
    )
    # Spades led, no hearts played: only a spade above the 10 wins.
    assert p._can_win_trick(Card("Spades", "Jack"), "Spades") is True
    assert p._can_win_trick(Card("Spades", "8"), "Spades") is False
    # Off-suit non-trump can never win — even the ace of clubs already on the
    # table is losing.
    assert p._can_win_trick(Card("Diamonds", "Ace"), "Spades") is False
    assert p._can_win_trick(Card("Clubs", "King"), "Spades") is False


def test_no_trump_offsuit_discard_is_not_the_reference_card():
    """
    The highest card on the table is an off-suit discard that cannot win;
    the reference card must be the highest *lead-suit* card.
    """
    g = make_game()
    p = _make_actor(
        g,
        2,
        "Hearts",
        [(0, Card("Spades", "5")), (1, Card("Clubs", "Ace"))],
    )
    # Old implementation compared against Clubs Ace and returned False.
    assert p._can_win_trick(Card("Spades", "6"), "Spades") is True


def test_can_win_trick_uses_trick_lead_not_stale_hint():
    g = make_game()
    p = _make_actor(g, 1, "Hearts", [(0, Card("Spades", "5"))])
    # Even with a bogus/None hint, the lead suit comes from the trick itself.
    assert p._can_win_trick(Card("Spades", "6"), None) is True
    assert p._can_win_trick(Card("Clubs", "Ace"), None) is False

"""
Perfect Information Monte Carlo (PIMC) player for Hokm.

The classic trick-taking-game search technique (Bridge/Skat lineage):
at every decision,

  1. *Determinize*: sample complete deals of the unseen cards to the three
     hidden hands, consistent with everything observed so far — exact hand
     sizes, and proven voids (a player who failed to follow a suit can
     never be dealt that suit).
  2. *Rollout*: for each legal candidate card, play the rest of the hand
     to completion in a fast rules-identical simulator, with every seat
     following a cheap greedy policy.
  3. *Vote*: pick the card with the best mean outcome (team tricks won,
     with hand wins weighted on top) across determinizations.

The rollout policy is a dependency-free re-implementation of
`baselines.HeuristicAgent`'s decision rules operating on plain tuples, so
thousands of rollouts per decision stay affordable without touching torch.

`PIMCPlayer` subclasses `EnhancedPlayer` only to satisfy the seat
interface (`hand`, `play_card`, bookkeeping attrs); it never uses the
neural networks, never learns, and reads public state from the live
`Hokm` instance (`_game`, populated by `Hokm.start_game → _sync_seats`).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from enhanced_player import EnhancedPlayer
from game_constants import Card, card_to_index, suits

# A card in the fast simulator is just (suit_index, value). Conversions
# happen once at the PIMC boundary, not inside rollouts.
FastCard = Tuple[int, int]


def _to_fast(card: Card) -> FastCard:
    return (suits.index(card.suit), card.value)


# ---------------------------------------------------------------------------
# Fast rollout simulator (rules-identical to Hokm's trick logic; verified by
# tests/test_pimc.py against Hokm.determine_trick_winner on random tricks).
# ---------------------------------------------------------------------------

def _legal(hand: Sequence[FastCard], lead_suit: Optional[int]) -> List[FastCard]:
    if lead_suit is None:
        return list(hand)
    following = [c for c in hand if c[0] == lead_suit]
    return following or list(hand)


def _trick_winner_idx(
    trick: Sequence[Tuple[int, FastCard]], trump: int
) -> int:
    """Seat of the winning (seat, card) entry, per Hokm rules."""
    lead_suit = trick[0][1][0]
    best_seat, best_card = trick[0]
    has_trump = any(c[0] == trump for _, c in trick)
    for seat, card in trick[1:]:
        if has_trump:
            if card[0] == trump and (
                best_card[0] != trump or card[1] > best_card[1]
            ):
                best_seat, best_card = seat, card
        else:
            if card[0] == lead_suit and (
                best_card[0] != lead_suit or card[1] > best_card[1]
            ):
                best_seat, best_card = seat, card
    return best_seat


def _current_winner(
    trick: Sequence[Tuple[int, FastCard]], trump: int
) -> Tuple[int, FastCard]:
    i = _trick_winner_idx(trick, trump)
    for seat, card in trick:
        if seat == i:
            return seat, card
    return trick[0]


def _rollout_policy(
    hand: List[FastCard],
    trick: List[Tuple[int, FastCard]],
    trump: int,
    seat: int,
    hand_counts: Sequence[int],
) -> FastCard:
    """Greedy rules mirroring HeuristicAgent: lead long suit high, beat
    cheaply, duck under a winning partner, dump low otherwise."""
    lead_suit = trick[0][1][0] if trick else None
    legal = _legal(hand, lead_suit)
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

    win_seat, win_card = _current_winner(trick, trump)
    partner = (seat + 2) % 4
    if win_seat == partner:
        non_trump = [c for c in legal if c[0] != trump]
        dump = non_trump or legal
        return min(dump, key=lambda c: c[1])

    # Minimum card that beats the current winner.
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
    dump = non_trump or legal
    return min(dump, key=lambda c: c[1])


def _rollout(
    hands: List[List[FastCard]],
    trick: List[Tuple[int, FastCard]],
    trump: int,
    next_seat: int,
    tricks_team: List[int],
) -> Tuple[int, int]:
    """Play the hand to completion (or 7 tricks). Returns (team0, team1)
    trick counts, where team0 = seats {0, 2}."""
    trick = list(trick)
    tricks = list(tricks_team)
    seat = next_seat
    while True:
        if tricks[0] >= 7 or tricks[1] >= 7:
            break
        if all(not h for h in hands) and not trick:
            break
        hand_counts = [len(h) for h in hands]
        card = _rollout_policy(hands[seat], trick, trump, seat, hand_counts)
        hands[seat].remove(card)
        trick.append((seat, card))
        if len(trick) == 4:
            winner = _trick_winner_idx(trick, trump)
            tricks[winner % 2] += 1
            trick = []
            seat = winner
        else:
            seat = (seat + 1) % 4
    return tricks[0], tricks[1]


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------

def sample_determinization(
    my_seat: int,
    my_hand: Sequence[Card],
    hand_sizes: Dict[int, int],
    unseen: Sequence[Card],
    voids: Dict[int, set],
    rng: random.Random,
    max_tries: int = 200,
) -> Optional[Dict[int, List[Card]]]:
    """
    Deal `unseen` to the other three seats respecting `hand_sizes` (exact)
    and `voids` (seat -> set of suit names that seat can NOT hold).

    Rejection sampling with a constrained greedy fill: cards that fewer
    seats can legally hold are placed first, which makes dead-ends rare.
    Returns {seat: [Card...]} for the three hidden seats, or None if no
    valid assignment was found (caller falls back to unconstrained).
    """
    other_seats = [s for s in range(4) if s != my_seat]
    for _ in range(max_tries):
        remaining = {s: hand_sizes[s] for s in other_seats}
        hands: Dict[int, List[Card]] = {s: [] for s in other_seats}
        # Most-constrained cards first, random tie-break.
        cards = list(unseen)
        rng.shuffle(cards)
        cards.sort(key=lambda c: sum(
            1 for s in other_seats
            if c.suit not in voids.get(s, ()) and remaining[s] > 0
        ))
        ok = True
        for card in cards:
            options = [
                s for s in other_seats
                if remaining[s] > 0 and card.suit not in voids.get(s, ())
            ]
            if not options:
                ok = False
                break
            weights = [remaining[s] for s in options]
            pick = rng.choices(options, weights=weights, k=1)[0]
            hands[pick].append(card)
            remaining[pick] -= 1
        if ok and all(v == 0 for v in remaining.values()):
            return hands
    return None


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------

class PIMCPlayer(EnhancedPlayer):
    """
    Drop-in Hokm seat that picks cards by determinized Monte-Carlo search.

    Parameters
    ----------
    determinizations : deals sampled per decision (default 24).
    win_weight : bonus added to a rollout's score when our team wins the
        hand — biases choices toward hand wins over raw trick count.
    rng : seeded random.Random for reproducible play.
    """

    def __init__(
        self,
        name: str,
        *,
        determinizations: int = 24,
        win_weight: float = 4.0,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(name)
        self.learning_enabled = False
        self.epsilon = 0.0
        self.eta = 0.0
        self.determinizations = determinizations
        self.win_weight = win_weight
        self._rng = rng or random.Random()

    # Baselines contract: no learning, no buffers.
    def store_experience(self, *_a, **_k) -> None:
        return None

    def optimize_model(self, *_a, **_k) -> None:
        return None

    # ------------------------------------------------------------------

    def _observed(self):
        """Collect public info from the live game. Falls back to
        heuristic-compatible defaults when unseated (unit tests)."""
        game = self._game
        my_seat = self._seat if self._seat is not None else 0
        trick = list(self.current_trick or [])
        played = list(getattr(game, "cards_played_this_hand", []) or []) if game else []
        voids_by_seat: Dict[int, set] = {}
        if game is not None:
            for p, sset in (getattr(game, "void_map", None) or {}).items():
                try:
                    voids_by_seat[game.players.index(p)] = set(sset)
                except ValueError:
                    continue
            hand_sizes = {i: len(p.hand) for i, p in enumerate(game.players)}
        else:
            hand_sizes = {i: len(self.hand) for i in range(4)}
        return my_seat, trick, played, voids_by_seat, hand_sizes

    def play_card(self, lead_suit, selected_card=None):
        self.lead_suit = lead_suit
        valid = (
            self.hand
            if lead_suit is None
            else [c for c in self.hand if c.suit == lead_suit] or self.hand
        )
        if not valid:
            raise ValueError(f"No valid cards to play for {self.name}")
        if len(valid) == 1:
            card = valid[0]
            return card, card_to_index(card)

        card = self._search(valid)
        return card, card_to_index(card)

    def _search(self, valid: List[Card]) -> Card:
        my_seat, trick, played, voids, hand_sizes = self._observed()
        trump_name = self.trump_suit or suits[0]
        trump = suits.index(trump_name)

        # Unseen = full deck minus my hand minus everything on the table or
        # already played this hand.
        seen = {(c.suit, c.rank) for c in self.hand}
        seen.update((c.suit, c.rank) for c in played)
        for _, c in trick:
            seen.add((c.suit, c.rank))
        from game_constants import ranks as all_ranks
        unseen = [
            Card(s, r) for s in suits for r in all_ranks if (s, r) not in seen
        ]

        # Trick in fast form, with seats resolved via the live game.
        game = self._game
        fast_trick: List[Tuple[int, FastCard]] = []
        for p, c in trick:
            try:
                seat = game.players.index(p) if game else 0
            except ValueError:
                seat = 0
            fast_trick.append((seat, _to_fast(c)))

        scores: Dict[Card, float] = {c: 0.0 for c in valid}
        samples = 0

        # Rollout counts teams by seat parity with team0 = seats {0, 2},
        # which is exactly the engine's "Team 1"; no re-mapping needed.
        base_tricks = [0, 0]
        if game is not None:
            base_tricks = [game.scores[1], game.scores[2]]

        for _ in range(self.determinizations):
            deal = sample_determinization(
                my_seat, self.hand, hand_sizes, unseen, voids, self._rng
            )
            if deal is None:
                # Fall back: unconstrained deal (voids unsatisfiable due to
                # inconsistent info shouldn't happen, but never crash a game).
                deal = sample_determinization(
                    my_seat, self.hand, hand_sizes, unseen, {}, self._rng
                )
                if deal is None:
                    continue
            samples += 1
            for cand in valid:
                hands: List[List[FastCard]] = [[] for _ in range(4)]
                hands[my_seat] = [_to_fast(c) for c in self.hand if c is not cand]
                for seat, cards in deal.items():
                    hands[seat] = [_to_fast(c) for c in cards]
                trick_now = fast_trick + [(my_seat, _to_fast(cand))]
                if len(trick_now) == 4:
                    winner = _trick_winner_idx(trick_now, trump)
                    tricks = [base_tricks[0], base_tricks[1]]
                    tricks[winner % 2] += 1
                    t0, t1 = _rollout(hands, [], trump, winner, tricks)
                else:
                    t0, t1 = _rollout(
                        hands,
                        trick_now,
                        trump,
                        (my_seat + 1) % 4,
                        [base_tricks[0], base_tricks[1]],
                    )
                mine, theirs = (t0, t1) if my_seat % 2 == 0 else (t1, t0)
                score = mine - theirs + (self.win_weight if mine >= 7 else 0.0)
                scores[cand] += score

        if samples == 0:
            return max(valid, key=lambda c: c.value)
        return max(valid, key=lambda c: scores[c])

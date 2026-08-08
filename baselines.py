"""
Baseline opponents for training curricula and evaluation suites.

All baselines satisfy the same interface as `EnhancedPlayer` (they inherit
from it), so they drop into `Hokm(players=[...])` unchanged. They override
`select_action` only, disable learning, and carry no RL transitions into
the shared learner — all callers that evaluate them should still set
`learning_enabled = False` for the NFSP seats too when benchmarking.

Two opponents are defined:

  - `RandomAgent`     : uniform random over legal cards.
  - `HeuristicAgent`  : deterministic (seedable) hand-coded strategy using
                         simple, human-sensible rules for Hokm.

Both are cheap, have no neural networks to load, and are deterministic
given a seeded `random.Random`.

Design notes for the heuristic:
  * Lead: play highest card of the longest non-trump suit, unless trump
    is this agent's best shot at winning the game (then push high trump).
  * Second to play: if partner not played yet, try to win cheaply.
  * Third/fourth to play: if partner is currently winning, duck; otherwise
    play minimum winning card (trump if necessary and allowed).
  * Always follow suit when legal (engine-enforced anyway).

This heuristic is intentionally *not* state-of-the-art — it exists as a
non-trivial baseline so we can measure whether a trained policy has
learned anything beyond random, and to feed diverse training opponents.
"""

from __future__ import annotations

import random
from typing import List, Optional

from enhanced_player import EnhancedPlayer
from game_constants import Card, card_to_index, suits


class _BaselinePlayer(EnhancedPlayer):
    """Shared scaffolding: no learning, no shared buffers, picks via override."""

    def __init__(self, name: str, *, rng: Optional[random.Random] = None) -> None:
        super().__init__(name)  # builds networks but we never train them
        self.learning_enabled = False
        self.eta = 0.0           # never enter the NFSP avg-policy branch
        self.epsilon = 0.0       # never explore
        self._rng = rng or random.Random()

    # Non-learning baselines store nothing and never optimize.
    def store_experience(self, *_args, **_kwargs) -> None:
        return None

    def optimize_model(self, *_args, **_kwargs) -> None:
        return None


class RandomAgent(_BaselinePlayer):
    """Uniform random over legal cards."""

    def select_action(self, valid_cards: List[Card]) -> int:
        if not valid_cards:
            return 0
        return card_to_index(self._rng.choice(valid_cards))


class HeuristicAgent(_BaselinePlayer):
    """
    Deterministic rule-based opponent.

    Not optimal, but follows the kind of advice a casual-competent human
    player internalizes: follow suit, win cheaply, conserve trump, set up
    partner when possible.
    """

    # ---------- helpers ----------

    def _highest(self, cards: List[Card]) -> Card:
        return max(cards, key=lambda c: c.value)

    def _lowest(self, cards: List[Card]) -> Card:
        return min(cards, key=lambda c: c.value)

    def _is_trump(self, card: Card) -> bool:
        return self.trump_suit is not None and card.suit == self.trump_suit

    def _current_winner(self) -> Optional[Card]:
        """Highest card currently winning the trick given lead + trump."""
        if not self.current_trick:
            return None
        trick = self.current_trick
        trumps = [c for _, c in trick if self._is_trump(c)]
        if trumps:
            return max(trumps, key=lambda c: c.value)
        lead = trick[0][1].suit
        in_suit = [c for _, c in trick if c.suit == lead]
        return max(in_suit, key=lambda c: c.value) if in_suit else trick[0][1]

    def _partner_is_winning(self) -> bool:
        """True if our teammate played the current-best card in the trick."""
        if not self.current_trick:
            return False
        winner_card = self._current_winner()
        if winner_card is None:
            return False
        teammate = self._get_teammate()
        if teammate is None:
            return False
        for p, c in self.current_trick:
            if c is winner_card and p == teammate:
                return True
        return False

    def _min_card_that_beats(self, cards: List[Card], to_beat: Card, lead: Optional[str]) -> Optional[Card]:
        """Lowest card in `cards` that would beat `to_beat` given lead & trump."""
        trump = self.trump_suit
        candidates: List[Card] = []
        trump_beats_nontrump = self._is_trump(to_beat) is False
        for c in cards:
            if c.suit == trump:
                if to_beat.suit != trump:
                    # any trump beats any non-trump
                    candidates.append(c)
                elif c.value > to_beat.value:
                    candidates.append(c)
            elif lead is not None and c.suit == lead:
                # same-suit-as-lead only wins against non-trump higher lead
                if to_beat.suit == lead and c.value > to_beat.value:
                    candidates.append(c)
                # lead card is non-trump here; if to_beat is trump, can't beat it
                _ = trump_beats_nontrump  # keep name for readability; no-op
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.value)

    # ---------- action selection ----------

    def select_action(self, valid_cards: List[Card]) -> int:
        if not valid_cards:
            return 0
        self.last_rl_eligible = False

        # Leading (no lead suit yet): play a mid/high non-trump if we have it;
        # otherwise play lowest trump we have (don't spike the highest trump
        # unless we're late in the hand).
        if not self.current_trick:
            non_trump = [c for c in valid_cards if not self._is_trump(c)]
            pool = non_trump or valid_cards
            # Prefer the highest card of the longest non-trump suit in hand.
            # Iterate suits in canonical order so length ties break to the
            # lowest suit index deterministically — `max` over a *set* of
            # suit names breaks ties by string-hash order, which made seeded
            # evaluations non-reproducible across processes.
            suits_in_pool = sorted({c.suit for c in pool}, key=suits.index)
            best_suit = max(
                suits_in_pool,
                key=lambda s: sum(1 for c in self.hand if c.suit == s),
            )
            best_suit_cards = [c for c in pool if c.suit == best_suit]
            choice = self._highest(best_suit_cards) if best_suit_cards else self._highest(pool)
            return card_to_index(choice)

        lead = self.current_trick[0][1].suit
        winner = self._current_winner()
        if winner is None:
            return card_to_index(self._rng.choice(valid_cards))

        # If our partner is already winning, duck with our lowest legal card
        # (we only "duck" when legal cards are plentiful; if only one card
        # is playable, it's forced anyway).
        if self._partner_is_winning():
            return card_to_index(self._lowest(valid_cards))

        # Otherwise try to beat `winner` with the minimum card that can.
        best = self._min_card_that_beats(valid_cards, winner, lead)
        if best is not None:
            return card_to_index(best)

        # Can't win; dump lowest legal (preserve high cards & trumps).
        non_trump_legal = [c for c in valid_cards if not self._is_trump(c)]
        dump_pool = non_trump_legal or valid_cards
        return card_to_index(self._lowest(dump_pool))

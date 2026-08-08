"""
Perfect Information Monte Carlo (PIMC) player for Hokm.

The classic trick-taking-game search technique (Bridge/Skat lineage):
at every decision,

  1. *Determinize*: sample complete deals of the unseen cards to the three
     hidden hands, consistent with everything observed so far — exact hand
     sizes, proven voids (a player who failed to follow a suit can never be
     dealt that suit), and the trump declaration, which leaks that the
     hakem's hand is trump-dense (`HAKEM_TRUMP_BIAS`).
  2. *Rollout*: for each candidate card, play the rest of the hand to
     completion in a fast rules-identical simulator, with every seat
     following a cheap greedy policy. Playing fourth to a trick, candidates
     that are dominated (same suit, same trick result, higher card) are
     dropped first — see `_prune_last_seat_candidates`.
  3. *Vote*: pick the card with the best mean outcome (team tricks won,
     with hand wins weighted on top) across determinizations.

The rollout policy is a dependency-free re-implementation of
`baselines.HeuristicAgent`'s decision rules operating on plain tuples, so
thousands of rollouts per decision stay affordable without touching torch.
It is written as index scans over `(suit, value)` tuples rather than the
obvious list-and-lambda form; `tests/test_pimc.py` keeps a transcription of
the obvious version as an oracle and differential-tests against it, so the
fast path can be tuned without silently changing how PIMC plays.

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


_SUIT_IDX: Dict[str, int] = {s: i for i, s in enumerate(suits)}


def _to_fast(card: Card) -> FastCard:
    return (_SUIT_IDX[card.suit], card.value)


# ---------------------------------------------------------------------------
# Fast rollout simulator (rules-identical to Hokm's trick logic; verified by
# tests/test_pimc.py against Hokm.determine_trick_winner on random tricks).
# ---------------------------------------------------------------------------

def _legal(hand: Sequence[FastCard], lead_suit: Optional[int]) -> List[FastCard]:
    """Legal subset of `hand` (a fresh list — safe to mutate).

    The rollout hot path does *not* call this: `_policy_index` inlines the
    same legality rule without allocating. Kept as the readable reference
    definition and for callers outside the inner loop.
    """
    if lead_suit is None:
        return list(hand)
    following = [c for c in hand if c[0] == lead_suit]
    return following or list(hand)


def _trick_winner(
    trick: Sequence[Tuple[int, FastCard]], trump: int
) -> Tuple[int, FastCard]:
    """(seat, card) of the winning entry, per Hokm rules: highest trump if
    any trump was played, else highest card of the led suit.

    Single pass — `best_is_trump` makes the running best switch to the
    trump race the moment the first trump lands, which is equivalent to the
    two-phase "was any trump played?" formulation but never rescans.
    """
    lead_suit = trick[0][1][0]
    best_seat, best_card = trick[0]
    best_is_trump = best_card[0] == trump
    for i in range(1, len(trick)):
        seat, card = trick[i]
        suit = card[0]
        if suit == trump:
            if not best_is_trump or card[1] > best_card[1]:
                best_seat, best_card, best_is_trump = seat, card, True
        elif not best_is_trump and suit == lead_suit and card[1] > best_card[1]:
            best_seat, best_card = seat, card
    return best_seat, best_card


def _trick_winner_idx(trick: Sequence[Tuple[int, FastCard]], trump: int) -> int:
    """Seat of the winning (seat, card) entry, per Hokm rules."""
    return _trick_winner(trick, trump)[0]


def _current_winner(
    trick: Sequence[Tuple[int, FastCard]], trump: int
) -> Tuple[int, FastCard]:
    return _trick_winner(trick, trump)


def _dump_index(
    hand: List[FastCard], trump: int, lead_suit: int, following: bool
) -> int:
    """Index of the lowest legal non-trump card, or the lowest legal card
    when every legal card is trump. First minimum wins ties, matching
    `min()` over the legal list in hand order."""
    best_i = -1
    best_v = 99
    for i, c in enumerate(hand):
        suit = c[0]
        if following and suit != lead_suit:
            continue
        if suit == trump:
            continue
        if c[1] < best_v:
            best_v = c[1]
            best_i = i
    if best_i >= 0:
        return best_i
    for i, c in enumerate(hand):
        if following and c[0] != lead_suit:
            continue
        if c[1] < best_v:
            best_v = c[1]
            best_i = i
    return best_i


def _policy_index(
    hand: List[FastCard],
    trick: List[Tuple[int, FastCard]],
    trump: int,
    seat: int,
) -> int:
    """Index into `hand` of the greedy rollout policy's choice.

    Same decision rules as `HeuristicAgent` (lead the long suit high, beat
    cheaply, duck under a winning partner, dump low otherwise) but written
    as index scans over tuples: no legal-list allocation, no `key=lambda`
    call per element, no repeated suit-length recomputation.
    """
    n = len(hand)

    # ---- leading: every card is legal -------------------------------
    if not trick:
        if n == 1:
            return 0
        suit_len = [0, 0, 0, 0]
        for c in hand:
            suit_len[c[0]] += 1
        # Prefer a non-trump suit; fall back to trump only if that is all
        # we hold. Among eligible suits take the longest, ties to the
        # lowest suit index (matches `max()` over a small int set).
        trump_only = suit_len[trump] == n
        best_suit = -1
        best_len = -1
        for s in range(4):
            if suit_len[s] == 0 or (s == trump and not trump_only):
                continue
            if suit_len[s] > best_len:
                best_len = suit_len[s]
                best_suit = s
        best_i = -1
        best_v = -1
        for i, c in enumerate(hand):
            if c[0] == best_suit and c[1] > best_v:
                best_v = c[1]
                best_i = i
        return best_i

    # ---- following ---------------------------------------------------
    lead_suit = trick[0][1][0]
    follow_count = 0
    follow_i = -1
    for i, c in enumerate(hand):
        if c[0] == lead_suit:
            follow_count += 1
            if follow_i < 0:
                follow_i = i
    following = follow_count > 0
    if following:
        if follow_count == 1:
            return follow_i        # forced
    elif n == 1:
        return 0                   # forced

    win_seat, win_card = _trick_winner(trick, trump)
    if win_seat == (seat + 2) % 4:
        return _dump_index(hand, trump, lead_suit, following)

    # Cheapest card that beats the current winner.
    win_suit = win_card[0]
    win_val = win_card[1]
    best_i = -1
    best_v = 99
    for i, c in enumerate(hand):
        suit = c[0]
        if following and suit != lead_suit:
            continue
        if suit == trump:
            if (win_suit != trump or c[1] > win_val) and c[1] < best_v:
                best_v = c[1]
                best_i = i
        elif (
            suit == lead_suit
            and win_suit == lead_suit
            and c[1] > win_val
            and c[1] < best_v
        ):
            best_v = c[1]
            best_i = i
    if best_i >= 0:
        return best_i

    return _dump_index(hand, trump, lead_suit, following)


def _rollout_policy(
    hand: List[FastCard],
    trick: List[Tuple[int, FastCard]],
    trump: int,
    seat: int,
    hand_counts: Optional[Sequence[int]] = None,
) -> FastCard:
    """Greedy rules mirroring HeuristicAgent: lead long suit high, beat
    cheaply, duck under a winning partner, dump low otherwise.

    `hand_counts` is unused (kept for call-site compatibility); the policy
    only ever looked at the acting seat's own hand.
    """
    return hand[_policy_index(hand, trick, trump, seat)]


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
    t0, t1 = tricks_team[0], tricks_team[1]
    seat = next_seat
    cards_left = len(hands[0]) + len(hands[1]) + len(hands[2]) + len(hands[3])
    while True:
        if t0 >= 7 or t1 >= 7:
            break
        if cards_left == 0 and not trick:
            break
        hand = hands[seat]
        i = _policy_index(hand, trick, trump, seat)
        card = hand[i]
        del hand[i]
        cards_left -= 1
        trick.append((seat, card))
        if len(trick) == 4:
            winner = _trick_winner(trick, trump)[0]
            if winner & 1:
                t1 += 1
            else:
                t0 += 1
            trick = []
            seat = winner
        else:
            seat = (seat + 1) & 3
    return t0, t1


# ---------------------------------------------------------------------------
# Candidate pruning
# ---------------------------------------------------------------------------

def _prune_last_seat_candidates(
    valid: Sequence[Card], fast_trick: Sequence[Tuple[int, FastCard]], trump: int
) -> List[Card]:
    """Cut the candidate set for the *fourth* seat of a trick to one
    representative per equivalence class.

    Playing last is the one decision where my card cannot change anything
    except (a) whether this trick is won and (b) which card leaves my hand.
    So two legal cards of the *same suit* that produce the *same* trick
    result are interchangeable for this trick, and of the two it is always
    at least as good to play the lower one and keep the higher: the hand
    that keeps the higher card can do everything the other hand can. Group
    the legal cards by (suit, wins-the-trick) and keep the cheapest of each
    group; everything dropped is dominated.

    Typically 4.0 candidates fall to ~2.2 — and, unlike a "lowest legal
    discard" rule, nothing the search might genuinely want to choose
    between (which suit to discard from, whether to ruff) is removed.

    Returns a subset of `valid`, in `valid`'s order (so score ties break
    exactly as they would without pruning).
    """
    lead_suit = fast_trick[0][1][0]
    win_suit, win_val = _trick_winner(fast_trick, trump)[1]

    # (suit, beats) -> (value, index of the cheapest such card)
    best: Dict[Tuple[int, bool], Tuple[int, int]] = {}
    for i, c in enumerate(valid):
        suit = _SUIT_IDX[c.suit]
        val = c.value
        if suit == trump:
            beats = win_suit != trump or val > win_val
        else:
            beats = suit == lead_suit and win_suit == lead_suit and val > win_val
        key = (suit, beats)
        cur = best.get(key)
        if cur is None or val < cur[0]:
            best[key] = (val, i)

    keep = {i for _, i in best.values()}
    if not keep:
        return list(valid)
    return [c for i, c in enumerate(valid) if i in keep]


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------

_NO_BIAS: Dict[str, float] = {}

# The hakem picked trump after seeing only their first five cards, scoring
# suits by `count * 10 + weighted value` (Hokm.choose_trump_suit). That is a
# real information leak: the declared trump is, in expectation, the hakem's
# longest/strongest suit, so their 13-card hand is trump-denser than a
# uniform deal. Weighting trump toward the hakem seat during determinization
# samples deals from a posterior closer to the truth.
#
# Calibrated, not guessed. Over 300 real mid-hand decision states the hakem
# turned out to hold 42.6% of the trumps still unseen from the observer's
# seat; uniform capacity-weighted sampling assigns them only 33.7% of those
# trumps, an 8.9 pp under-estimate. Measured shares by multiplier:
#
#     1.0 (uniform) 33.7%   1.6  38.8%   2.0  41.2%   2.5  43.5%
#
# 2.2 lands just under the 42.6% target — deliberately the conservative
# side, since an over-confident sampler collapses the deal diversity PIMC
# depends on, while a slightly under-confident one only dilutes the signal.
HAKEM_TRUMP_BIAS = 2.2


def sample_determinization(
    my_seat: int,
    my_hand: Sequence[Card],
    hand_sizes: Dict[int, int],
    unseen: Sequence[Card],
    voids: Dict[int, set],
    rng: random.Random,
    max_tries: int = 200,
    bias: Optional[Dict[int, Dict[str, float]]] = None,
) -> Optional[Dict[int, List[Card]]]:
    """
    Deal `unseen` to the other three seats respecting `hand_sizes` (exact)
    and `voids` (seat -> set of suit names that seat can NOT hold).

    Rejection sampling with a constrained greedy fill: cards that fewer
    seats can legally hold are placed first, which makes dead-ends rare.
    Returns {seat: [Card...]} for the three hidden seats, or None if no
    valid assignment was found (caller falls back to unconstrained).

    `bias` optionally tilts *which* seat gets a card: it maps
    seat -> {suit name: multiplier}. The multiplier scales that seat's
    remaining-capacity weight for cards of that suit, so a seat with
    weight 2.0 on trump is twice as likely to receive any given trump as
    capacity alone would suggest. Constraints are unaffected — biasing
    never deals a card into a proven void nor overfills a hand. When
    `bias` is None the sampler takes the original uniform-by-capacity
    path (identical RNG consumption).
    """
    other_seats = [s for s in range(4) if s != my_seat]
    if bias is not None and not any(s in bias for s in other_seats):
        bias = None

    # --- per-suit tables, built once instead of per card per try ---------
    # Which seats may legally hold each suit, and (for the ordering key) how
    # many of them have capacity at the *start* of a try — nothing has been
    # dealt yet at sort time, so that count depends only on the suit.
    allowed: Dict[str, Tuple[int, ...]] = {}
    order_key: Dict[str, int] = {}
    suit_bias: Dict[str, Tuple[float, ...]] = {}
    for suit in suits:
        seats = tuple(
            s for s in other_seats if suit not in voids.get(s, ())
        )
        allowed[suit] = seats
        order_key[suit] = sum(1 for s in seats if hand_sizes[s] > 0)
        if bias is not None:
            suit_bias[suit] = tuple(
                bias.get(s, _NO_BIAS).get(suit, 1.0) for s in seats
            )
    # A uniform key makes the (stable) sort a no-op; skip it entirely.
    need_sort = len(set(order_key.values())) > 1
    random_ = rng.random

    for _ in range(max_tries):
        remaining = [0, 0, 0, 0]
        for s in other_seats:
            remaining[s] = hand_sizes[s]
        hands: Dict[int, List[Card]] = {s: [] for s in other_seats}
        # Most-constrained cards first, random tie-break.
        cards = list(unseen)
        rng.shuffle(cards)
        if need_sort:
            cards.sort(key=lambda c: order_key[c.suit])
        ok = True
        for card in cards:
            suit = card.suit
            seats = allowed[suit]
            options: List[int] = []
            weights: List[float] = []
            if bias is None:
                for s in seats:
                    r = remaining[s]
                    if r > 0:
                        options.append(s)
                        weights.append(r)
            else:
                mult = suit_bias[suit]
                for j, s in enumerate(seats):
                    r = remaining[s]
                    if r > 0:
                        options.append(s)
                        weights.append(r * mult[j])
                # A pathological all-zero bias would make the draw
                # degenerate; fall back to plain capacity weighting.
                if not any(weights):
                    weights = [remaining[s] for s in options]
            if not options:
                ok = False
                break
            # Inlined `rng.choices(options, weights=weights, k=1)[0]`: the
            # same single `random()` draw and the same cumulative-weight
            # bisection, without the per-call list churn. Keeping the draw
            # even for a single option preserves the RNG stream exactly.
            total = 0.0
            cum = []
            for w in weights:
                total += w
                cum.append(total)
            x = random_() * total
            i = 0
            last = len(options) - 1
            while i < last and cum[i] <= x:
                i += 1
            pick = options[i]
            hands[pick].append(card)
            remaining[pick] -= 1
        if ok and not any(remaining):
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
    prune_last_seat : drop dominated candidates when playing fourth to a
        trick (see `_prune_last_seat_candidates`). On by default; the
        rollouts it saves are spent on the candidates that matter.
    """

    def __init__(
        self,
        name: str,
        *,
        determinizations: int = 24,
        win_weight: float = 4.0,
        rng: Optional[random.Random] = None,
        prune_last_seat: bool = True,
    ):
        super().__init__(name)
        self.learning_enabled = False
        self.epsilon = 0.0
        self.eta = 0.0
        self.determinizations = determinizations
        self.win_weight = win_weight
        self._rng = rng or random.Random()
        self.prune_last_seat = prune_last_seat

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

    def _hakem_bias(
        self, game, my_seat: int, trump_name: str
    ) -> Optional[Dict[int, Dict[str, float]]]:
        """Per-seat suit weights for `sample_determinization`, encoding the
        one thing the trump declaration tells us for free: the hakem's hand
        is trump-dense. None when I am the hakem (no inference to make) or
        when there is no live game to read the hakem from."""
        if game is None or not self.trump_suit:
            return None
        hakem = getattr(game, "hakem", None)
        if hakem is None or hakem is self:
            return None
        try:
            hakem_seat = game.players.index(hakem)
        except (ValueError, AttributeError):
            return None
        if hakem_seat == my_seat:
            return None
        return {hakem_seat: {trump_name: HAKEM_TRUMP_BIAS}}

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

        # Hakem-trump inference: if someone else declared trump, tilt the
        # sampler so that seat holds more of it (see HAKEM_TRUMP_BIAS).
        bias = self._hakem_bias(game, my_seat, trump_name)

        # Fourth to a trick: dominated candidates cannot change the outcome,
        # so spend the rollouts only on distinguishable choices.
        if self.prune_last_seat and len(fast_trick) == 3 and len(valid) > 1:
            valid = _prune_last_seat_candidates(valid, fast_trick, trump)

        # Everything that does not depend on the sampled deal is built once
        # per decision rather than once per (deal × candidate) rollout: the
        # fast form of my own hand, my hand minus each candidate, and the
        # extended trick for each candidate.
        my_fast = [_to_fast(c) for c in self.hand]
        pos_of = {id(c): i for i, c in enumerate(self.hand)}
        rest_of: List[List[FastCard]] = []
        trick_of: List[List[Tuple[int, FastCard]]] = []
        for cand in valid:
            i = pos_of.get(id(cand))
            if i is None:
                rest = [_to_fast(c) for c in self.hand if c is not cand]
                cand_fast = _to_fast(cand)
            else:
                rest = my_fast[:i] + my_fast[i + 1:]
                cand_fast = my_fast[i]
            rest_of.append(rest)
            trick_of.append(fast_trick + [(my_seat, cand_fast)])

        n_cand = len(valid)
        scores: List[float] = [0.0] * n_cand
        samples = 0
        next_seat = (my_seat + 1) & 3
        my_team_is_0 = my_seat % 2 == 0

        # Rollout counts teams by seat parity with team0 = seats {0, 2},
        # which is exactly the engine's "Team 1"; no re-mapping needed.
        base_tricks = [0, 0]
        if game is not None:
            base_tricks = [game.scores[1], game.scores[2]]
        b0, b1 = base_tricks

        for _ in range(self.determinizations):
            deal = sample_determinization(
                my_seat, self.hand, hand_sizes, unseen, voids, self._rng,
                bias=bias,
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
            # One conversion of the sampled hands per deal, shared (by copy)
            # across every candidate rollout.
            deal_fast: List[List[FastCard]] = [[], [], [], []]
            for seat, cards in deal.items():
                deal_fast[seat] = [_to_fast(c) for c in cards]

            for ci in range(n_cand):
                hands: List[List[FastCard]] = [
                    deal_fast[0][:], deal_fast[1][:],
                    deal_fast[2][:], deal_fast[3][:],
                ]
                hands[my_seat] = rest_of[ci][:]
                trick_now = trick_of[ci]
                if len(trick_now) == 4:
                    winner = _trick_winner(trick_now, trump)[0]
                    tricks = [b0, b1]
                    tricks[winner & 1] += 1
                    t0, t1 = _rollout(hands, [], trump, winner, tricks)
                else:
                    t0, t1 = _rollout(hands, trick_now, trump, next_seat, [b0, b1])
                mine, theirs = (t0, t1) if my_team_is_0 else (t1, t0)
                score = mine - theirs + (self.win_weight if mine >= 7 else 0.0)
                scores[ci] += score

        if samples == 0:
            return max(valid, key=lambda c: c.value)
        best_i = 0
        best_score = scores[0]
        for ci in range(1, n_cand):
            if scores[ci] > best_score:
                best_score = scores[ci]
                best_i = ci
        return valid[best_i]

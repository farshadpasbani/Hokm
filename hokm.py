# hokm.py

import random
import sys
import traceback
import torch
import pandas as pd
import os
from datetime import datetime
from typing import Optional
from game_constants import (
    ACTION_DIM,
    Card,
    card_to_index,
    suits,
    ranks,
    rank_values,
)
from enhanced_player import EnhancedPlayer, TeamStrategy
import time
import csv


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        if len(self.cards) != 52:
            raise ValueError(
                f"Deck initialized with {len(self.cards)} cards, expected 52"
            )

    def shuffle(self, rng: "random.Random | None" = None):
        """Shuffle using the given seeded RNG if provided (reproducible eval)."""
        if rng is not None:
            rng.shuffle(self.cards)
        else:
            random.shuffle(self.cards)

    def deal(self, num_cards):
        # print(f"Deck size before dealing: {len(self.cards)}")
        if len(self.cards) < num_cards:
            raise ValueError(f"Not enough cards in deck to deal {num_cards} cards")
        dealt_cards = [self.cards.pop() for _ in range(num_cards)]
        # print(f"Deck size after dealing: {len(self.cards)}")
        for card in dealt_cards:
            if not isinstance(card, Card):
                raise ValueError(f"Invalid card dealt: {card}")
        return dealt_cards


class Hokm:
    def __init__(
        self,
        players,
        trick_csv_path=None,
        minimal_logging: bool = False,
        rng: "random.Random | None" = None,
        hakem_stays_on_win: bool = False,
    ):
        """
        players: list of 4 agent objects implementing the EnhancedPlayer-compatible
                 interface (hand, draw, reset, play_card, get_state,
                 store_experience, optimize_model, evaluate_play, and bookkeeping attrs).
        trick_csv_path: optional per-trick review CSV path.
        minimal_logging: if True, skip pandas logging on the hot path (for training).
        rng: a seeded `random.Random` instance used for deck shuffling and first-time
             Hakem selection. If None, the module-level `random` is used (legacy).
             Use this for reproducible evaluation.
        hakem_stays_on_win: rule-authenticity switch for `rotate_hakem()`, see RULES.md §8.
             False (default, unchanged legacy behavior): on a Hakem-team win the
             Hakem seat toggles to the partner. True (traditional rule): a winning
             Hakem keeps the Hakemship; only on a loss does it pass, to the first
             winning-team seat clockwise after the old Hakem.
        """
        self.players = players
        self.trick_csv_path = trick_csv_path
        self.minimal_logging = minimal_logging
        self.rng = rng
        self.hakem_stays_on_win = hakem_stays_on_win
        self._trick_csv_fh = None
        self._trick_csv_writer = None
        self.deck = Deck()
        self.current_trick = []
        self.lead_suit = None
        self.trump_suit = None
        self.hakem = None
        self.scores = {1: 0, 2: 0}
        self.game_log = pd.DataFrame()
        self.difficulty_level = 1
        self.hakem_cards = None
        self.game_count = 0
        self.round_count = 0
        self.trick_count = 0
        self.trick_starter_index = 0  # index into self.players; leads the current trick
        self.last_trick_winner = None
        self.team1 = [self.players[0], self.players[2]]
        self.team2 = [self.players[1], self.players[3]]
        self.team_strategy = TeamStrategy()
        self.tricks_won = {player: 0 for player in self.players}
        self.last_winning_team = self.team1
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Public-info bookkeeping read by every player's get_state().
        # Reset per hand in start_game(). Order in cards_played_this_hand is
        # the order-of-play over the whole hand.
        self.cards_played_this_hand: list = []
        self.void_map: dict = {player: set() for player in self.players}

        # Kot latch. `is_kot()` is the *live* view of the scores currently in
        # the engine, so it goes False again the moment the next hand resets
        # them. Match-play callers need the value to survive that boundary, so
        # every hand completion snapshots it here and `last_hand_kot` reads
        # only the snapshot. Deliberately never reset: it always describes the
        # most recently *completed* hand, for the whole life of the instance.
        self._kot_latched: bool = False

        # Failure accounting. `play_game` catches exceptions from `play_round`
        # so a bad game doesn't kill a training run, but the caller still needs
        # to know that a game aborted (otherwise silent bugs like a state-dim
        # mismatch are indistinguishable from a healthy run). We always surface
        # the first error to stderr, and expose cumulative counters that
        # TrainBackend / evaluators can report in their summaries.
        self.aborted_games: int = 0
        self.last_error: Optional[BaseException] = None
        self._error_printed_once: bool = False

        for player in self.players:
            player.team = self.team1 if player in self.team1 else self.team2
            player.tricks_won = self.tricks_won
            player.team_strategy = self.team_strategy

    def _init_trick_csv_if_needed(self):
        if not self.trick_csv_path or self._trick_csv_fh is not None:
            return
        path = os.path.abspath(self.trick_csv_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._trick_csv_fh = open(path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "game",
            "trick",
            "hakem",
            "trump_suit",
            "lead_suit",
            "team1_players",
            "team2_players",
            "seat0_player",
            "seat1_player",
            "seat2_player",
            "seat3_player",
            "seat0_card",
            "seat1_card",
            "seat2_card",
            "seat3_card",
            "play_order",
            "trick_winner_player",
            "trick_winner_team",
            "team1_tricks",
            "team2_tricks",
            "game_over",
            "game_winner_team",
            "timestamp",
        ]
        self._trick_csv_writer = csv.DictWriter(
            self._trick_csv_fh, fieldnames=fieldnames
        )
        self._trick_csv_writer.writeheader()

    def _append_trick_review_csv(self, winner):
        """One row per completed trick; only used when trick_csv_path is set."""
        self._init_trick_csv_if_needed()
        if not self._trick_csv_writer:
            return
        lead_suit = self.current_trick[0][1].suit
        cards_by_seat = [""] * 4
        for p, c in self.current_trick:
            cards_by_seat[self.players.index(p)] = str(c)
        play_order = " -> ".join(f"{p.name}:{c}" for p, c in self.current_trick)
        game_over = self.scores[1] >= 7 or self.scores[2] >= 7
        game_winner = ""
        if self.scores[1] >= 7:
            game_winner = "Team 1"
        elif self.scores[2] >= 7:
            game_winner = "Team 2"
        row = {
            "game": self.game_count,
            "trick": self.round_count + 1,
            "hakem": self.hakem.name if self.hakem else "",
            "trump_suit": self.trump_suit or "",
            "lead_suit": lead_suit,
            "team1_players": f"{self.team1[0].name} & {self.team1[1].name}",
            "team2_players": f"{self.team2[0].name} & {self.team2[1].name}",
            "seat0_player": self.players[0].name,
            "seat1_player": self.players[1].name,
            "seat2_player": self.players[2].name,
            "seat3_player": self.players[3].name,
            "seat0_card": cards_by_seat[0],
            "seat1_card": cards_by_seat[1],
            "seat2_card": cards_by_seat[2],
            "seat3_card": cards_by_seat[3],
            "play_order": play_order,
            "trick_winner_player": winner.name,
            "trick_winner_team": "Team 1" if winner in self.team1 else "Team 2",
            "team1_tricks": self.scores[1],
            "team2_tricks": self.scores[2],
            "game_over": "yes" if game_over else "no",
            "game_winner_team": game_winner,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._trick_csv_writer.writerow(row)
        self._trick_csv_fh.flush()

    def close_trick_csv(self):
        if self._trick_csv_fh:
            self._trick_csv_fh.close()
            self._trick_csv_fh = None
            self._trick_csv_writer = None

    def reset_players(self):
        for player in self.players:
            player.reset()
            if player.hand:
                # print(f"Clearing {player.name}'s hand: {[str(c) for c in player.hand]}")
                player.hand = []  # Force clear the hand
        # CRITICAL: every player aliases this dict (set in __init__ /
        # _maybe_swap_opponents), and `compute_terminal_reward` reads it via
        # `self.tricks_won` on the *player*. If we replace this attribute with
        # a fresh dict, the players keep pointing at the old (stale) one and
        # every terminal reward collapses to "I lost" regardless of outcome.
        # Mutate in place so all aliasing references stay valid.
        self.tricks_won.clear()
        for player in self.players:
            self.tricks_won[player] = 0
        self.scores = {1: 0, 2: 0}
        self.current_trick = []
        self.lead_suit = None
        self.round_count = 0
        self.trick_count = 0
        self.trick_starter_index = 0
        self.last_trick_winner = None

    def start_game(self):
        self.deck = Deck()
        self.deck.shuffle(self.rng)
        self.reset_players()
        if not self.hakem:
            self.hakem = (self.rng or random).choice(self.players)
            # print(f"{self.hakem.name} is the Hakem for this game")
        # Fresh public-info bookkeeping per hand. Must happen before players
        # start calling get_state() so they see empty history / no voids.
        self.cards_played_this_hand = []
        # Same events with exact seat attribution: [(seat_index, card), ...]
        # in play order. Consumed by sequence-model agents (dmc.py).
        self.play_log_this_hand = []
        self.void_map = {p: set() for p in self.players}
        for p in self.players:
            if hasattr(p, "_sync_seats"):
                p._sync_seats(self)
        self.hakem_cards = self.deck.deal(5)
        self.hakem.hand = self.hakem_cards.copy()
        # print(
        #     f"Hakem {self.hakem.name} received cards: {[str(c) for c in self.hakem.hand]}"
        # )
        self.log_game_state("Game initialized", player_hands=True)
        return self.hakem_cards

    def set_trump_suit(self, trump_suit):
        if trump_suit not in suits:
            raise ValueError(f"Invalid trump suit: {trump_suit}")
        self.trump_suit = trump_suit
        for player in self.players:
            player.update_trump_suit(trump_suit)
        # print(f"Trump suit set to: {self.trump_suit}")
        cards_per_player = 8 if self.hakem else 13
        for player in self.players:
            num_cards = 8 if player == self.hakem else 13
            player.draw(self.deck, num_cards)
            # print(f"{player.name} hand after draw: {[str(c) for c in player.hand]}")
        self.trick_starter_index = self.players.index(self.hakem)
        self._sync_player_trick_context()
        self.log_game_state("Game started", player_hands=True)

    def choose_trump_suit(self):
        suit_counts = {suit: 0 for suit in suits}
        suit_values = {suit: 0 for suit in suits}
        for card in self.hakem.hand:
            suit_counts[card.suit] += 1
            suit_values[card.suit] += card.value * (2 if card.value >= 10 else 1)
        best_suit = max(
            suits, key=lambda s: suit_counts[s] * 10 + suit_values[s]
        )  # Best strategy for selecting the trump suit?
        self.set_trump_suit(best_suit)

    def play_round(self):
        """
        Play one trick. Experience storage is deferred until after the trick
        resolves so that:
          * `done` correctly reflects end-of-hand (game_over OR hand empty),
          * `next_state` sees the resolved trick (updated scores / trick_winner),
          * terminal-reward bonuses can be injected on the final transition.
        """
        self.current_trick = []
        self.lead_suit = None
        # Re-point every player at the NEW trick list. Without this, players
        # keep referencing the previous trick's list object and get_state()
        # sees a stale (full) trick until their own next play re-syncs them —
        # which corrupted the trick/lead/position/winner observation blocks
        # for every not-yet-synced seat.
        self._sync_player_trick_context()
        hakem_index = self.players.index(self.hakem)

        if self.round_count == 0:
            self.trick_starter_index = hakem_index
        else:
            if self.last_trick_winner is None:
                raise RuntimeError(
                    "last_trick_winner must be set before non-first tricks"
                )
            self.trick_starter_index = self.players.index(self.last_trick_winner)
        starting_player_index = self.trick_starter_index
        current_player = self.players[starting_player_index]

        # Pending transitions, recorded in play order, flushed after trick resolves.
        pending = []

        player_rewards = {}
        player_action_indices = {}
        player_valid_cards = {}
        play_order = []

        for _ in range(4):
            try:
                state = current_player.get_state()
                valid_cards = (
                    current_player.hand
                    if self.lead_suit is None
                    else [
                        card
                        for card in current_player.hand
                        if card.suit == self.lead_suit
                    ]
                    or current_player.hand
                )
                player_valid_cards[current_player.name] = [str(c) for c in valid_cards]

                result = current_player.play_card(self.lead_suit)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError(
                        f"Invalid return from play_card for {current_player.name}: {result}"
                    )
                card, action_index = result
                if not isinstance(card, Card):
                    raise ValueError(
                        f"Invalid card returned by {current_player.name}: {card}"
                    )

                if card not in current_player.hand:
                    error_msg = (
                        f"{current_player.name} attempted to play {card}, "
                        f"not in hand: {[str(c) for c in current_player.hand]}"
                    )
                    self.log_game_state(error_msg, player_hands=True)
                    raise ValueError(error_msg)

                current_player.hand.remove(card)
                self.current_trick.append((current_player, card))
                play_order.append(current_player.name)
                current_player.current_trick = self.current_trick
                prev_lead_suit = self.lead_suit
                if not self.lead_suit:
                    self.lead_suit = card.suit

                current_player.actions_taken.append(action_index)
                current_player.played_cards_memory.add(str(card))
                current_player.team_strategy.update_card_count(card)
                current_player.played_suit_counts[suits.index(card.suit)] += 1

                # ---- public-info bookkeeping for get_state() ---------------
                # Every seat's observation shares these structures, so the
                # network gets rank-level memory (lemma #1) and void flags
                # (lemma #2) without any per-player duplication.
                self.cards_played_this_hand.append(card)
                self.play_log_this_hand.append(
                    (self.players.index(current_player), card)
                )
                # A non-leader failing to follow the led suit proves they are
                # void in that suit for the rest of the hand.
                if prev_lead_suit is not None and card.suit != prev_lead_suit:
                    self.void_map.setdefault(current_player, set()).add(prev_lead_suit)

                shaping_reward = self.evaluate_play(
                    current_player, card, self.lead_suit, self.round_count
                )
                player_rewards[current_player.name] = shaping_reward
                player_action_indices[current_player.name] = action_index

                pending.append(
                    {
                        "player": current_player,
                        "state": state,
                        "action": action_index,
                        "shaping_reward": shaping_reward,
                        "rl_eligible": getattr(
                            current_player, "last_rl_eligible", True
                        ),
                    }
                )
            except Exception:
                error_msg = (
                    f"Error in play_round for {current_player.name}: "
                    f"see traceback"
                )
                self.log_game_state(error_msg, player_hands=True)
                raise
            current_player_index = (self.players.index(current_player) + 1) % 4
            current_player = self.players[current_player_index]

        # Resolve trick (updates self.scores) before storing experiences so
        # `next_state` sees the outcome and we can flag game-over correctly.
        winner = self.determine_trick_winner()
        self.last_trick_winner = winner
        self.trick_starter_index = self.players.index(winner)
        team = 1 if winner in self.team1 else 2
        self.scores[team] += 1
        self._append_trick_review_csv(winner)

        game_over = self.scores[1] >= 7 or self.scores[2] >= 7

        # Flush pending transitions with correct `done` + terminal reward.
        # We compute `next_legal_mask` from the player's hand at flush time
        # — by then the played card has already been removed from `p.hand`,
        # so the mask reflects the actual cards available at the player's
        # next decision point. This is an over-approximation of legality
        # (it doesn't yet incorporate the next trick's lead suit, which
        # isn't known yet), but it bounds argmax to *cards in hand*, which
        # is the critical invariant: every card in hand will eventually be
        # played, so its Q-value is anchored by real-world feedback. Cards
        # not in hand (already played, never dealt) are the dangerous
        # phantom actions whose Q-values drift unboundedly without this
        # mask. See SharedNFSPLearner.push_transition for full rationale.
        for rec in pending:
            p = rec["player"]
            hand_empty = len(p.hand) == 0
            done = game_over or hand_empty
            reward = rec["shaping_reward"]
            if done:
                terminal = 0.0
                if hasattr(p, "compute_terminal_reward"):
                    try:
                        terminal = float(p.compute_terminal_reward())
                    except Exception:
                        terminal = 0.0
                reward += terminal
            next_state = p.get_state()
            next_legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
            if not done:
                for c in p.hand:
                    next_legal_mask[card_to_index(c)] = True
            p.store_experience(
                rec["state"],
                rec["action"],
                reward,
                next_state,
                done,
                rl_eligible=rec["rl_eligible"],
                next_legal_mask=next_legal_mask,
            )
            p.optimize_model()

        self.log_round(
            self.round_count,
            self.lead_suit,
            play_order,
            self.current_trick,
            winner,
            player_rewards,
            player_action_indices,
            player_valid_cards,
        )
        self.log_game_state("Trick completed")
        self.round_count += 1
        return winner

    def _sync_player_trick_context(self):
        """Keep per-player trick/lead mirrors in sync for heuristics and the DQN."""
        for player in self.players:
            player.current_trick = self.current_trick
            player.lead_suit = self.lead_suit

    def get_next_to_play(self):
        if not self.current_trick:
            return self.players[self.trick_starter_index]
        last_player, _ = self.current_trick[-1]
        return self.players[(self.players.index(last_player) + 1) % 4]

    def legal_cards_for_player(self, player):
        if not self.lead_suit:
            return list(player.hand)
        following = [c for c in player.hand if c.suit == self.lead_suit]
        return following or list(player.hand)

    def apply_play(self, player, card):
        """Play one card if legal and it is this player's turn. Returns an error message or None."""
        if player != self.get_next_to_play():
            return "Not this player's turn"
        if card not in player.hand:
            return "Card not in hand"
        legal = self.legal_cards_for_player(player)
        if card not in legal:
            return "Illegal card for this trick"
        player.hand.remove(card)
        self.current_trick.append((player, card))
        prev_lead_suit = self.lead_suit
        if self.lead_suit is None:
            self.lead_suit = card.suit
        # Mirror play_round()'s public-info bookkeeping so the web-app path
        # feeds the NFSP observation the same card-memory + voids features.
        self.cards_played_this_hand.append(card)
        self.play_log_this_hand.append((self.players.index(player), card))
        if prev_lead_suit is not None and card.suit != prev_lead_suit:
            self.void_map.setdefault(player, set()).add(prev_lead_suit)
        self._sync_player_trick_context()
        return None

    def resolve_trick_if_complete(self):
        """
        If the trick has four cards, pick a winner, update scores, and reset trick state.
        Returns (winner, trick_snapshot) or (None, None).
        trick_snapshot is a list of dicts: {"player": name, "card": str(card)}.
        """
        if len(self.current_trick) < 4:
            return None, None
        snapshot = [{"player": p.name, "card": str(c)} for p, c in self.current_trick]
        winner = self.determine_trick_winner()
        self.last_trick_winner = winner
        self.trick_starter_index = self.players.index(winner)
        team = 1 if winner in self.team1 else 2
        self.scores[team] += 1
        self.current_trick = []
        self.lead_suit = None
        self._sync_player_trick_context()
        # Step-API hand completion: this trick may have taken a team to 7 (or
        # exhausted the last cards). Latch Kot here so the web/service layer
        # sees it even though it never calls play_game().
        self._latch_kot_if_hand_complete()
        return winner, snapshot

    def play_game(self, save_excel_log=True):
        self.game_count += 1
        # print(f"Starting game {self.game_count}")
        self.start_game()
        self.choose_trump_suit()
        self.round_count = 0  # Reset round_count at start of game
        current_player = self.hakem  # Start with Hakem for first trick
        while any(len(player.hand) > 0 for player in self.players):
            try:
                winner = self.play_round()
                current_player = winner  # Winner of the trick starts the next one
                if self.scores[1] >= 7 or self.scores[2] >= 7:
                    break
            except Exception as e:
                # A play_round exception means this hand is unrecoverable. We
                # break out and let the caller (train loop / evaluator) decide
                # whether to continue, but we must NOT silently drop this on
                # the floor — silent failures here previously masked a
                # state-dim mismatch that produced 100k all-zero games.
                self.aborted_games += 1
                self.last_error = e
                self.log_game_state(f"Round error: {str(e)}", player_hands=True)
                if not self._error_printed_once:
                    self._error_printed_once = True
                    print(
                        f"[Hokm] play_round aborted game {self.game_count} "
                        f"on round {self.round_count}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    print(
                        "[Hokm] further per-game aborts will be counted silently; "
                        "see `Hokm.aborted_games` / `Hokm.last_error`.",
                        file=sys.stderr,
                    )
                break
        # Latch Kot before anything resets the scores. Guarded by
        # _hand_is_complete(), so a hand aborted by the exception handler
        # above leaves the previous hand's value in place.
        self._latch_kot_if_hand_complete()
        self.update_last_winning_team()
        self.rotate_hakem()
        self.adjust_difficulty()
        self.log_game_state("Game ended", player_hands=True)
        if save_excel_log:
            self.save_game_log()

    def determine_trick_winner(self):
        winning_card = self.current_trick[0][1]
        winner = self.current_trick[0][0]
        has_trump = any(card.suit == self.trump_suit for _, card in self.current_trick)
        for player, card in self.current_trick[1:]:
            if has_trump:
                if card.suit == self.trump_suit:
                    if (
                        winning_card.suit != self.trump_suit
                        or card.value > winning_card.value
                    ):
                        winning_card = card
                        winner = player
            else:
                if card.suit == self.lead_suit:
                    if (
                        winning_card.suit != self.lead_suit
                        or card.value > winning_card.value
                    ):
                        winning_card = card
                        winner = player
        self.tricks_won[winner] += 1
        return winner

    def evaluate_play(self, player, card, lead_suit, round_num):
        return player.evaluate_play(card, lead_suit, round_num)

    def is_kot(self) -> bool:
        """
        **Live** view: True iff the hand currently reflected in `self.scores`
        is a Kot (کت) — the winning team took 7+ tricks and the losing team
        took none.

        Because it reads `self.scores`, it is correct on both play paths
        (`play_game`'s loop and the incremental `apply_play` /
        `resolve_trick_if_complete` step API used by the web app) without
        either having to set a flag — but it also reverts to False as soon as
        the next hand resets the scores. Use `last_hand_kot` if you need the
        value after the hand boundary.

        Engine-level detection only — Kot carries **no** scoring consequence
        here (see RULES.md §7).
        """
        t1 = self.scores.get(1, 0)
        t2 = self.scores.get(2, 0)
        return (t1 >= 7 and t2 == 0) or (t2 >= 7 and t1 == 0)

    def _hand_is_complete(self) -> bool:
        """A hand is over once a team reaches 7 tricks or all cards are gone."""
        return (
            self.scores.get(1, 0) >= 7
            or self.scores.get(2, 0) >= 7
            or all(len(p.hand) == 0 for p in self.players)
        )

    def _latch_kot_if_hand_complete(self) -> None:
        """Snapshot `is_kot()` at hand completion. Called from both play paths."""
        if self._hand_is_complete():
            self._kot_latched = self.is_kot()

    @property
    def last_hand_kot(self) -> bool:
        """
        Kot status of the most recently **completed** hand — latched, so it
        survives `start_game()` / `reset_players()` into the next hand.

        False until the first hand completes (and False for a hand that ended
        without a Kot — the two cases are not distinguished here; check the
        scores or your own hand counter if you need to tell them apart). A
        hand aborted mid-play by `play_game`'s exception handler does not
        latch, so the value keeps describing the last hand that really
        finished.
        """
        return self._kot_latched

    def update_last_winning_team(self):
        team1_tricks = sum(self.tricks_won[player] for player in self.team1)
        team2_tricks = sum(self.tricks_won[player] for player in self.team2)
        # A completed hand always has exactly one team at >= 7 tricks, so
        # "more tricks" is identical to the old ">= 7" test there. The
        # difference is the aborted-hand path (`play_game` swallows a
        # `play_round` exception): previously *any* unfinished hand silently
        # credited team 2. Now the team that actually led on tricks wins, and
        # a dead tie leaves `last_winning_team` untouched.
        if team1_tricks > team2_tricks:
            self.last_winning_team = self.team1
        elif team2_tricks > team1_tricks:
            self.last_winning_team = self.team2
        # else: tie — keep the previous value (team1 if never set).

    def rotate_hakem(self):
        current_team = self.team1 if self.hakem in self.team1 else self.team2
        if self.hakem_stays_on_win:
            # Traditional rule: a winning Hakem keeps the Hakemship.
            if self.last_winning_team == current_team:
                return
            # Lost: Hakemship passes to the winning team — specifically to the
            # first winning-team player in play order (clockwise) after the
            # outgoing Hakem, which is the seat immediately to the Hakem's left.
            start = self.players.index(self.hakem)
            for step in range(1, 5):
                candidate = self.players[(start + step) % 4]
                if candidate in self.last_winning_team:
                    self.hakem = candidate
                    return
            return
        if self.last_winning_team != current_team:
            self.hakem = self.last_winning_team[0]
        else:
            current_idx = current_team.index(self.hakem)
            next_idx = (current_idx + 1) % 2
            self.hakem = current_team[next_idx]

    def log_round(
        self,
        round_num,
        lead_suit,
        play_order,
        current_trick,
        winner,
        player_rewards,
        player_action_indices,
        player_valid_cards,
    ):
        if self.minimal_logging:
            self.trick_count += 1
            return
        played_cards = {player.name: "None" for player in self.players}
        for player, card in current_trick:
            played_cards[player.name] = str(card)

        hand_sizes = {player.name: len(player.hand) for player in self.players}
        hands = {
            player.name: [str(card) for card in player.hand] for player in self.players
        }
        team1_score = sum(self.tricks_won[player] for player in self.team1)
        team2_score = sum(self.tricks_won[player] for player in self.team2)
        game_winner = (
            "Team 1" if team1_score >= 7 else "Team 2" if team2_score >= 7 else None
        )

        trick_contents = ", ".join([f"{p.name}: {str(c)}" for p, c in current_trick])

        row = {
            "Game": self.game_count,
            "Round": round_num,
            "Trick Number": self.trick_count,
            "Hakem": self.hakem.name,
            "Trump Suit": self.trump_suit,
            "Lead Suit": lead_suit,
            "Play Order": ", ".join(play_order),
            "Trick Contents": trick_contents,
            "Player 1 Card": played_cards.get("Player 1", "None"),
            "Player 2 Card": played_cards.get("Player 2", "None"),
            "Player 3 Card": played_cards.get("Player 3", "None"),
            "Player 4 Card": played_cards.get("Player 4", "None"),
            "Trick Winner": winner.name,
            "Winning Team": "Team 1" if winner in self.team1 else "Team 2",
            "Player 1 Hand": ", ".join(hands.get("Player 1", [])),
            "Player 2 Hand": ", ".join(hands.get("Player 2", [])),
            "Player 3 Hand": ", ".join(hands.get("Player 3", [])),
            "Player 4 Hand": ", ".join(hands.get("Player 4", [])),
            "Player 1 Hand Size": hand_sizes.get("Player 1", 0),
            "Player 2 Hand Size": hand_sizes.get("Player 2", 0),
            "Player 3 Hand Size": hand_sizes.get("Player 3", 0),
            "Player 4 Hand Size": hand_sizes.get("Player 4", 0),
            "Player 1 Reward": player_rewards.get("Player 1", 0.0),
            "Player 2 Reward": player_rewards.get("Player 2", 0.0),
            "Player 3 Reward": player_rewards.get("Player 3", 0.0),
            "Player 4 Reward": player_rewards.get("Player 4", 0.0),
            "Player 1 Action Index": player_action_indices.get("Player 1", -1),
            "Player 2 Action Index": player_action_indices.get("Player 2", -1),
            "Player 3 Action Index": player_action_indices.get("Player 3", -1),
            "Player 4 Action Index": player_action_indices.get("Player 4", -1),
            "Player 1 Valid Cards": ", ".join(player_valid_cards.get("Player 1", [])),
            "Player 2 Valid Cards": ", ".join(player_valid_cards.get("Player 2", [])),
            "Player 3 Valid Cards": ", ".join(player_valid_cards.get("Player 3", [])),
            "Player 4 Valid Cards": ", ".join(player_valid_cards.get("Player 4", [])),
            "Game Winner": game_winner,
            "Team 1 Score": team1_score,
            "Team 2 Score": team2_score,
            "Difficulty Level": self.difficulty_level,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.game_log = pd.concat(
            [self.game_log, pd.DataFrame([row])], ignore_index=True
        )
        self.trick_count += 1

    def log_game_state(self, event, player_hands=False):
        if self.minimal_logging:
            return
        hand_sizes = {player.name: len(player.hand) for player in self.players}
        hands = {
            player.name: [str(card) for card in player.hand] for player in self.players
        }
        team1_score = sum(self.tricks_won[player] for player in self.team1)
        team2_score = sum(self.tricks_won[player] for player in self.team2)
        row = {
            "Game": self.game_count,
            "Round": self.round_count,
            "Trick Number": self.trick_count,
            "Hakem": self.hakem.name if self.hakem else "None",
            "Trump Suit": self.trump_suit,
            "Lead Suit": self.lead_suit,
            "Event": event,
            "Player 1 Hand Size": hand_sizes.get("Player 1", 0),
            "Player 2 Hand Size": hand_sizes.get("Player 2", 0),
            "Player 3 Hand Size": hand_sizes.get("Player 3", 0),
            "Player 4 Hand Size": hand_sizes.get("Player 4", 0),
            "Player 1 Hand": (
                ", ".join(hands.get("Player 1", [])) if player_hands else ""
            ),
            "Player 2 Hand": (
                ", ".join(hands.get("Player 2", [])) if player_hands else ""
            ),
            "Player 3 Hand": (
                ", ".join(hands.get("Player 3", [])) if player_hands else ""
            ),
            "Player 4 Hand": (
                ", ".join(hands.get("Player 4", [])) if player_hands else ""
            ),
            "Team 1 Score": team1_score,
            "Team 2 Score": team2_score,
            "Difficulty Level": self.difficulty_level,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.game_log = pd.concat(
            [self.game_log, pd.DataFrame([row])], ignore_index=True
        )

    def save_game_log(self, file_name=None, **_kwargs):
        """
        Write game logs as CSV (no Excel). Produces three files:
        {stem}.csv, {stem}_summary.csv, {stem}_team_stats.csv under game_logs/.
        """
        if file_name is None:
            stem = f"game_log_{self.session_id}_game_{self.game_count}"
        else:
            stem = os.path.splitext(os.path.basename(file_name))[0]
        os.makedirs("game_logs", exist_ok=True)
        main_path = os.path.join("game_logs", f"{stem}.csv")
        try:
            self.game_log.to_csv(main_path, index=False)
            summary = self._create_summary_statistics()
            summary.to_csv(
                os.path.join("game_logs", f"{stem}_summary.csv"), index=False
            )
            team_stats = self._create_team_statistics()
            team_stats.to_csv(
                os.path.join("game_logs", f"{stem}_team_stats.csv"), index=False
            )
        except Exception:
            pass

    def _training_summary_from_state(self):
        """Lightweight metrics for training when game_log is disabled."""
        t1 = self.scores[1]
        t2 = self.scores[2]
        n_tricks = t1 + t2
        team1_win_rate = 1.0 if t1 >= 7 else 0.0
        team2_win_rate = 1.0 if t2 >= 7 else 0.0
        player_stats = {}
        for i in range(1, 5):
            pname = f"Player {i}"
            p = self.players[i - 1]
            actions = getattr(p, "actions_taken", []) or []
            n_act = max(1, len(actions))
            player_stats[f"{pname} Avg Reward"] = getattr(p, "total_reward", 0.0) / n_act
            player_stats[f"{pname} Trick Wins"] = self.tricks_won.get(p, 0)
        return pd.DataFrame(
            {
                "Total Games Played": [1],
                "Total Tricks Played": [n_tricks],
                "Average Tricks per Game": [float(n_tricks)],
                "Most Common Trump Suit": [self.trump_suit or "N/A"],
                "Most Winning Team": [
                    "Team 1" if t1 >= 7 else ("Team 2" if t2 >= 7 else "N/A")
                ],
                "Team 1 Win Rate": [team1_win_rate],
                "Team 2 Win Rate": [team2_win_rate],
                **player_stats,
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
        )

    def _create_summary_statistics(self):
        if self.minimal_logging and self.game_log.empty:
            return self._training_summary_from_state()
        if self.game_log.empty:
            return pd.DataFrame()
        unique_games = self.game_log["Game"].unique()
        total_games = len(unique_games)
        total_tricks = len(self.game_log[self.game_log["Event"] == "Trick completed"])
        trump_counts = (
            self.game_log.groupby("Game")["Trump Suit"].first().value_counts()
        )
        most_common_trump = trump_counts.idxmax() if not trump_counts.empty else "N/A"
        game_winners = (
            self.game_log.groupby("Game")["Game Winner"].last().value_counts()
        )
        most_winning_team = game_winners.idxmax() if not game_winners.empty else "N/A"

        # Calculate team win rates
        team1_wins = len(self.game_log[self.game_log["Game Winner"] == "Team 1"])
        team2_wins = len(self.game_log[self.game_log["Game Winner"] == "Team 2"])
        team1_win_rate = team1_wins / total_games if total_games > 0 else 0
        team2_win_rate = team2_wins / total_games if total_games > 0 else 0

        # Calculate player rewards and trick wins
        player_stats = {}
        for i in range(1, 5):
            player_name = f"Player {i}"
            reward_col = f"{player_name} Reward"
            if reward_col in self.game_log.columns:
                player_stats[f"{player_name} Avg Reward"] = self.game_log[
                    reward_col
                ].mean()
            else:
                player_stats[f"{player_name} Avg Reward"] = 0.0
            p_obj = self.players[i - 1] if i - 1 < len(self.players) else None
            player_stats[f"{player_name} Trick Wins"] = (
                self.tricks_won.get(p_obj, 0) if p_obj is not None else 0
            )

        return pd.DataFrame(
            {
                "Total Games Played": [total_games],
                "Total Tricks Played": [total_tricks],
                "Average Tricks per Game": [
                    total_tricks / total_games if total_games > 0 else 0
                ],
                "Most Common Trump Suit": [most_common_trump],
                "Most Winning Team": [most_winning_team],
                "Team 1 Win Rate": [team1_win_rate],
                "Team 2 Win Rate": [team2_win_rate],
                **player_stats,
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
        )

    def _create_team_statistics(self):
        if self.game_log.empty:
            return pd.DataFrame()
        game_stats = (
            self.game_log.groupby("Game")
            .agg(
                {
                    "Game Winner": "last",
                    "Winning Team": "count",
                    "Team 1 Score": "last",
                    "Team 2 Score": "last",
                }
            )
            .reset_index()
        )
        team_stats = pd.DataFrame(
            {
                "Team": ["Team 1", "Team 2"],
                "Total Wins": [
                    len(game_stats[game_stats["Game Winner"] == "Team 1"]),
                    len(game_stats[game_stats["Game Winner"] == "Team 2"]),
                ],
                "Total Tricks Won": [
                    len(self.game_log[self.game_log["Winning Team"] == "Team 1"]),
                    len(self.game_log[self.game_log["Winning Team"] == "Team 2"]),
                ],
                "Average Tricks per Game": [
                    game_stats[game_stats["Game Winner"] == "Team 1"][
                        "Winning Team"
                    ].mean()
                    or 0,
                    game_stats[game_stats["Game Winner"] == "Team 2"][
                        "Winning Team"
                    ].mean()
                    or 0,
                ],
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
        )
        return team_stats

    def adjust_difficulty(self):
        """Uses rows that include trick-level stats (`Game Winner`); event-only rows are skipped."""
        if self.game_log.empty or "Game Winner" not in self.game_log.columns:
            return
        gl = self.game_log
        dl = self.difficulty_level
        team1_wins = int(
            ((gl["Game Winner"] == "Team 1") & (gl["Difficulty Level"] == dl)).sum()
        )
        team2_wins = int(
            ((gl["Game Winner"] == "Team 2") & (gl["Difficulty Level"] == dl)).sum()
        )
        total_games = team1_wins + team2_wins
        if total_games >= 10:
            win_rate = team1_wins / total_games if total_games > 0 else 0
            if win_rate > 0.7 and self.difficulty_level < 3:
                self.difficulty_level += 1
                # print(f"Difficulty increased to level {self.difficulty_level}")
            elif win_rate < 0.3 and self.difficulty_level > 1:
                self.difficulty_level -= 1
                # print(f"Difficulty decreased to level {self.difficulty_level}")

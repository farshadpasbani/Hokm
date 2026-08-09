"""
Multi-session Hokm game service for the Telegram Mini App backend.

`app.py` keeps a single global game for local development; this module is
the production counterpart: one `GameSession` per authenticated Telegram
user, with per-session locks (gunicorn runs threaded), TTL eviction, and a
cap on concurrent sessions.

AI seats (strongest first), selected via AI_KIND:
  * "pimc" (default): `PIMCPlayer` — determinized Monte-Carlo search over
    sampled opponent hands (respecting proven voids), heuristic rollouts.
    Measured stronger than the plain heuristic and far stronger than any
    NFSP checkpoint to date (see TRAINING_REPORT.md). PIMC_DETERMINIZATIONS
    tunes strength vs latency (default 32; ~0.02-0.05 s per decision).
  * "heuristic": rule-based `HeuristicAgent`.
  * "checkpoint": greedy trained agent from MODEL_PATH (or the newest .pth
    in models_release/). Falls back to heuristic when no checkpoint exists.

Match play (multi-hand):
  A session runs a *match*, not a single hand. `new_game()` starts a fresh
  match at 0–0; each hand ends when a team takes 7 tricks, its result is
  folded into `match_score`, the engine rotates the Hakem, and `next_hand()`
  deals the following hand with the same four seats. The match ends when a
  team reaches `match_target` (default 7, `MATCH_TARGET` env). Hand-level
  scoring — including the Kot (7–0) rule — is implemented here at the
  service layer; `hokm.py` is untouched and still models exactly one hand.

Seat ownership:
  Every seat is owned either by a human (`human_seats`: seat index →
  display name) or by AI. Solo play is the degenerate case
  `{0: display_name}`; a multi-human table (see `table_service.py`) passes
  two to four entries. Every acting method therefore takes the acting seat,
  defaulting to 0 so the solo call sites are unchanged, and every payload is
  built *for one seat*: shared table facts plus that seat's own hand and
  legal cards, never anyone else's cards.

  `auto_seats` is the set of human seats the AI should cover right now
  (their owner has gone idle). The session obeys the set; whoever owns the
  clock — the table layer — decides what goes in it.
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from baselines import HeuristicAgent
from enhanced_player import EnhancedPlayer
from game_constants import Card, STATE_DIM, ACTION_DIM, ranks, suits
from game_recorder import GameRecorder
from hokm import Hokm
from pimc import PIMCPlayer

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(2 * 60 * 60)))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))
DEFAULT_MATCH_TARGET = 7


def _default_model_path() -> str:
    """MODEL_PATH env wins; otherwise auto-detect a shipped release checkpoint."""
    explicit = os.getenv("MODEL_PATH", "")
    if explicit:
        return explicit
    release_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_release")
    if os.path.isdir(release_dir):
        candidates = sorted(
            f for f in os.listdir(release_dir) if f.endswith(".pth")
        )
        if candidates:
            return os.path.join(release_dir, candidates[0])
    return ""


MODEL_PATH = _default_model_path()
AI_KIND = os.getenv("AI_KIND", "pimc").strip().lower()
PIMC_DETERMINIZATIONS = int(os.getenv("PIMC_DETERMINIZATIONS", "32"))

# Seat order in Hokm(players) is south, east, north, west; partners sit two
# apart (Team 1 = seats 0 and 2, Team 2 = seats 1 and 3). An AI seat is
# labelled by the direction it occupies. In solo play seats 1-3 are AI, which
# reproduces the original "AI East / AI North / AI West" labels exactly.
_AI_SEAT_LABELS = ["AI South", "AI East", "AI North", "AI West"]

SEAT_COUNT = 4

TEAM1, TEAM2 = "Team 1", "Team 2"


class GameServiceError(Exception):
    """User-visible game/service errors (turned into JSON error responses)."""


# Every finished hand is appended to GAME_DATA_DIR as training material.
# Module-level so tests can swap it; failures inside never break a game.
RECORDER = GameRecorder()


def _match_target() -> int:
    """Hands needed to win a match. `MATCH_TARGET` env overrides the default."""
    try:
        target = int(os.getenv("MATCH_TARGET", str(DEFAULT_MATCH_TARGET)))
    except ValueError:
        return DEFAULT_MATCH_TARGET
    return target if target >= 1 else DEFAULT_MATCH_TARGET


def _build_ai_seat(label: str) -> EnhancedPlayer:
    if AI_KIND == "pimc":
        return PIMCPlayer(label, determinizations=PIMC_DETERMINIZATIONS)
    if AI_KIND == "checkpoint" and MODEL_PATH and os.path.isfile(MODEL_PATH):
        p = EnhancedPlayer(label, STATE_DIM, ACTION_DIM, epsilon=0.0, eta=0.0)
        p.learning_enabled = False
        p.load_policy_state(MODEL_PATH)
        return p
    return HeuristicAgent(label)


class GameSession:
    """
    One four-seat **match**. Seats are owned by humans or by AI.

    A match is a sequence of hands played by the same four seats. Hand state
    lives on `self.game` (the `Hokm` engine instance, which models exactly
    one hand); match state lives here.

    `human_seats` maps seat index → display name for every human-owned seat.
    Solo play (the default) is one human at seat 0 and AI at seats 1-3.
    """

    def __init__(
        self,
        user_id: str,
        display_name: str,
        human_seats: Optional[Dict[int, str]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.user_id = user_id
        self.display_name = display_name or "You"
        # A solo session follows its (refreshable) display name; a table
        # session carries a fixed per-seat name map handed to it at creation.
        self._solo = human_seats is None
        self.human_seats: Dict[int, str] = (
            {0: self.display_name} if self._solo else dict(human_seats)
        )
        # Human seats the AI should play right now because their owner has
        # gone idle. Owned by the caller (see the module docstring).
        self.auto_seats: Set[int] = set()
        # Bumped on every state-changing step (deal, trump, each card played).
        # Lets a caller detect "the game moved" without diffing payloads.
        self.revision: int = 0
        # Handed to `Hokm(rng=...)` (hokm.py), which uses it for the deck
        # shuffle and the first-time Hakem draw. None — every production
        # caller — leaves the engine on module-level `random`, unchanged.
        # Tests pass a seeded Random to get reproducible deals.
        self.rng = rng
        self.lock = threading.Lock()
        self.last_seen = time.time()
        self.game: Optional[Hokm] = None
        self.human: Optional[EnhancedPlayer] = None
        # Rule-based stand-in used to play an idle human's seat; built on
        # first use so solo sessions never pay for it (see `_cover_choice`).
        self._stand_in: Optional[HeuristicAgent] = None
        self.match_target: int = _match_target()
        self.match_score: Dict[str, int] = {TEAM1: 0, TEAM2: 0}
        self.match_over: bool = False
        self.match_winner: Optional[str] = None
        # Result of the hand currently on the table once it has ended:
        # {"hand_result": "Team 1"|"Team 2"|None, "kot": bool}. None while a
        # hand is still in progress.
        self.hand_result: Optional[Dict[str, Any]] = None
        # Training-data capture: snapshot of the deal taken the moment trump
        # is fixed (all four hands complete, nothing played yet). Consumed by
        # _finish_hand_if_over → RECORDER. See game_recorder.py.
        self._hand_snapshot: Optional[Dict[str, Any]] = None
        self._hand_index: int = 0
        # Tester feedback for the hand on the table: 0-based indices of tricks
        # flagged as bad AI play. Cleared when the next hand is dealt.
        self._flagged_tricks: Set[int] = set()
        # hand_id of the hand already written to disk, while its end-of-hand
        # screen is still up — late flags amend that record (see flag_trick).
        self._recorded_hand_id: Optional[str] = None

    # ---------- lifecycle ----------

    def touch(self) -> None:
        self.last_seen = time.time()

    def new_game(self, seat: int = 0) -> Dict[str, Any]:
        """Start a brand-new match (match score 0–0, Hakem drawn at random)."""
        if self._solo:
            self.human_seats = {0: self.display_name}
        players: List[EnhancedPlayer] = []
        for i in range(SEAT_COUNT):
            name = self.human_seats.get(i)
            if name is None:
                players.append(_build_ai_seat(_AI_SEAT_LABELS[i]))
                continue
            p = EnhancedPlayer(name, STATE_DIM, ACTION_DIM, is_human=True)
            p.learning_enabled = False
            players.append(p)
        # minimal_logging: a session now spans a whole match, and nothing here
        # reads `game_log`; skip the per-hand pandas concat.
        game = Hokm(players, minimal_logging=True, rng=self.rng)
        self.game = game
        # Kept for the solo call sites and their tests: the seat this session
        # was created for. Multi-human paths address seats by index.
        self.human = players[min(self.human_seats)] if self.human_seats else None
        self.match_target = _match_target()
        self.match_score = {TEAM1: 0, TEAM2: 0}
        self.match_over = False
        self.match_winner = None
        self.hand_result = None
        self._hand_snapshot = None
        self._hand_index = 0
        return self._deal_hand(seat)

    def next_hand(self, seat: int = 0) -> Dict[str, Any]:
        """
        Deal the next hand of the current match.

        The `Hokm` instance is reused so the Hakem rotation computed at the
        end of the previous hand (`update_last_winning_team` + `rotate_hakem`,
        engine behaviour) carries over: `start_game()` keeps `self.hakem` when
        it is already set.
        """
        self._require_game()
        if not self._hand_over():
            raise GameServiceError("The current hand is still in progress.")
        self._finish_hand_if_over()  # idempotent; normally already recorded
        if self.match_over:
            raise GameServiceError("The match is over — start a new match.")
        self.hand_result = None
        return self._deal_hand(seat)

    def _deal_hand(self, seat: int) -> Dict[str, Any]:
        """Shuffle, deal, and resolve the trump decision for one hand."""
        game = self._require_game()
        # `Hokm.start_game()` does not clear the previous hand's trump; the
        # service uses `trump_suit is None` as its "waiting for Hakem" signal,
        # so reset it here rather than patching the engine.
        game.trump_suit = None
        game.game_count += 1
        game.start_game()
        self.revision += 1
        events = self._advance()
        if not game.trump_suit:
            body = self._choose_trump_payload(seat)
            body["message"] = (
                "You are Hakem — choose the trump suit."
                if seat == self._hakem_seat()
                else f"{game.hakem.name} is Hakem — waiting for the trump suit."
            )
            return {"status": "success", **body}
        return {
            "status": "success",
            "phase": "playing",
            "message": f"{game.hakem.name} is Hakem; trump is {game.trump_suit}.",
            **self._payload(seat, last_events=events, trick_before_ai=[]),
        }

    def set_trump(self, trump_suit: str, seat: int = 0) -> Dict[str, Any]:
        game = self._require_game()
        self._require_human_seat(seat)
        if game.trump_suit:
            raise GameServiceError("Trump has already been chosen.")
        if self._hakem_seat() != seat:
            raise GameServiceError("Only the Hakem chooses trump.")
        if trump_suit not in suits:
            raise GameServiceError(f"Invalid suit: {trump_suit}")
        hand = game.players[seat].hand
        if trump_suit not in {c.suit for c in hand}:
            raise GameServiceError("Trump must be one of your dealt suits.")
        game.set_trump_suit(trump_suit)
        self._snapshot_deal(trump_chosen_by_human=True)
        self.revision += 1
        events = self._run_ai_turns()
        return {
            "status": "success",
            "phase": "playing",
            "message": f"Trump suit set to {trump_suit}.",
            **self._payload(seat, last_events=events, trick_before_ai=[]),
        }

    def play_card(self, card_str: str, seat: int = 0) -> Dict[str, Any]:
        game = self._require_game()
        self._require_human_seat(seat)
        if not game.trump_suit:
            raise GameServiceError("Choose the trump suit first.")
        if self._hand_over():
            raise GameServiceError(
                "This hand is over — start the next hand."
                if not self.match_over
                else "The match is over — start a new match."
            )
        try:
            card = Card.from_string(card_str)
        except ValueError as e:
            raise GameServiceError(str(e)) from e
        me = game.players[seat]
        if game.get_next_to_play() is not me:
            raise GameServiceError("Not your turn.")
        err = game.apply_play(me, card)
        if err:
            raise GameServiceError(err)
        self.revision += 1

        trick_before_ai = [
            {"player": p.name, "card": str(c)} for p, c in game.current_trick
        ]
        events = self._run_ai_turns()
        body = self._payload(
            seat, last_events=events, trick_before_ai=trick_before_ai
        )
        body["status"] = "success"
        body["phase"] = "ended" if body["game_over"] else "playing"
        return body

    def state(self, seat: int = 0) -> Dict[str, Any]:
        if not self.game:
            return {"status": "success", "phase": "idle"}
        game = self._require_game()
        events = self._advance()
        if not game.trump_suit:
            return {"status": "success", **self._choose_trump_payload(seat)}
        body = self._payload(seat, last_events=events, trick_before_ai=[])
        body["status"] = "success"
        body["phase"] = "ended" if body["game_over"] else "playing"
        return body

    def flag_trick(self, trick_index: Any) -> Dict[str, Any]:
        """
        Mark one trick of the hand on the table as bad AI play ("AI is stupid").

        `trick_index` is the trick the CLIENT is displaying, not the server's
        current trick. The server resolves every AI turn the instant the human
        plays, while the client animates those cards a beat later, so the two
        run up to a trick apart — taking the server's own position here would
        attach the flag to the wrong trick.

        Flags are deduped and land in the hand's training record. The record is
        written the moment the hand ends, so a flag pressed while the
        end-of-hand screen is still up is persisted as an amendment line that
        `game_recorder.load_hands` folds back in.
        """
        game = self._require_game()
        if not game.trump_suit:
            raise GameServiceError("No hand in progress to flag.")
        try:
            index = int(trick_index)
        except (TypeError, ValueError):
            raise GameServiceError("trick_index must be an integer.") from None

        resolved = game.scores[1] + game.scores[2]
        # While the hand runs, the trick on the table (index == resolved) is
        # flaggable too; once it ends only completed tricks exist.
        highest = resolved - 1 if self._hand_over() else resolved
        if index < 0 or index > highest:
            raise GameServiceError(
                f"trick_index {index} is outside this hand (0–{highest})."
            )

        already = index in self._flagged_tricks
        self._flagged_tricks.add(index)
        if not already and self._recorded_hand_id:
            try:
                RECORDER.record_flags(
                    self._recorded_hand_id, sorted(self._flagged_tricks)
                )
            except Exception:
                # Like recording, feedback must never break live play.
                pass
        return {
            "status": "success",
            "trick_index": index,
            "already_flagged": already,
            "flagged_tricks": sorted(self._flagged_tricks),
        }

    # ---------- internals (mirrors app.py's single-game helpers) ----------

    def _require_game(self) -> Hokm:
        if not self.game:
            raise GameServiceError("No game in progress — start a new game.")
        return self.game

    def _require_human_seat(self, seat: int) -> None:
        """Reject an action aimed at a seat this session does not own."""
        if seat not in self.human_seats:
            raise GameServiceError("That seat is not yours to play.")

    def _hakem_seat(self) -> int:
        return self.game.players.index(self.game.hakem)

    def _ai_controls(self, seat: int) -> bool:
        """True when the AI acts for `seat`: an AI seat, or an idle human's."""
        return seat not in self.human_seats or seat in self.auto_seats

    def _advance(self) -> List[Dict[str, Any]]:
        """
        Let the AI act until the next decision belongs to a live human.

        Covers the trump decision too: an AI Hakem — or a human Hakem whose
        seat the AI is currently covering — picks trump so the table cannot
        stall on an absent player.
        """
        game = self._require_game()
        if not game.trump_suit:
            if not self._ai_controls(self._hakem_seat()):
                return []
            game.choose_trump_suit()
            self._snapshot_deal(trump_chosen_by_human=False)
            self.revision += 1
        return self._run_ai_turns()

    def _scores(self) -> dict:
        return {"Team 1": self.game.scores[1], "Team 2": self.game.scores[2]}

    def _seats(self) -> dict:
        p = self.game.players
        return {
            "south": p[0].name,
            "east": p[1].name,
            "north": p[2].name,
            "west": p[3].name,
        }

    def _hand_over(self) -> bool:
        """True once the current hand is decided (7 tricks, or cards exhausted)."""
        g = self.game
        return bool(
            g
            and (
                g.scores[1] >= 7
                or g.scores[2] >= 7
                or all(len(pl.hand) == 0 for pl in g.players)
            )
        )

    def _finish_hand_if_over(self) -> None:
        """
        Fold a finished hand into the match score and rotate the Hakem.

        Idempotent: the first call for a given hand records it, later calls
        (e.g. repeated `/api/state` polls) are no-ops.

        Hakem rotation is *engine* behaviour, but the step API used by the web
        app (`apply_play` / `resolve_trick_if_complete`) never runs the tail of
        `Hokm.play_game()`, so `update_last_winning_team()` + `rotate_hakem()`
        are invoked here — before the next `start_game()`, which clears the
        per-player `tricks_won` counters those methods read.
        """
        if self.hand_result is not None or not self._hand_over():
            return
        g = self.game
        t1, t2 = g.scores[1], g.scores[2]
        winner = TEAM1 if t1 > t2 else TEAM2 if t2 > t1 else None
        # Kot (کت): the losing team took zero tricks. Standard tables score a
        # Kot as two points; ties/unfinished hands never qualify.
        kot = bool(winner and min(t1, t2) == 0 and max(t1, t2) >= 7)
        if winner:
            self.match_score[winner] += 2 if kot else 1
            if self.match_score[winner] >= self.match_target:
                self.match_over = True
                self.match_winner = winner
        self.hand_result = {
            "hand_result": winner,
            "kot": kot,
            "hakem": g.hakem.name if g.hakem else None,
        }
        # Persist the finished hand as training material — before rotate_hakem
        # (the record needs THIS hand's Hakem) and before the next start_game
        # clears play_log_this_hand.
        self._record_hand(winner, kot)
        g.update_last_winning_team()
        g.rotate_hakem()

    def _snapshot_deal(self, *, trump_chosen_by_human: bool) -> None:
        """Capture the deal at the moment trump is fixed: all four hands
        complete, nothing played. This plus the play log is a complete,
        replayable description of the hand (see game_recorder.replay_hand)."""
        g = self.game
        self._hand_index += 1
        # A new hand starts with no flags, and closes the amendment window on
        # the previous hand — flags from here on belong to this hand.
        self._flagged_tricks = set()
        self._recorded_hand_id = None
        self._hand_snapshot = {
            "initial_hands": [[str(c) for c in p.hand] for p in g.players],
            "trump_chosen_by_human": trump_chosen_by_human,
            "hakem_seat": g.players.index(g.hakem),
            "hand_index": self._hand_index,
            "hand_id": uuid.uuid4().hex,
        }

    def _record_hand(self, winner: Optional[str], kot: bool) -> None:
        snap = self._hand_snapshot
        self._hand_snapshot = None
        if snap is None:
            return
        g = self.game
        try:
            record = {
                "user": self.user_id,
                "player_name": self.display_name,
                "ai_kind": AI_KIND,
                # Which seats a person played. Solo records {"0": name}; a
                # table records two to four. Without it a replayed multi-human
                # hand cannot tell human decisions from AI ones.
                "human_seats": {
                    str(s): n for s, n in sorted(self.human_seats.items())
                },
                "hakem_seat": snap["hakem_seat"],
                "trump": g.trump_suit,
                "trump_chosen_by_human": snap["trump_chosen_by_human"],
                "initial_hands": snap["initial_hands"],
                "plays": [[seat, str(c)] for seat, c in g.play_log_this_hand],
                "scores": {"team1": g.scores[1], "team2": g.scores[2]},
                "winner_team": 1 if winner == TEAM1 else 2 if winner == TEAM2 else 0,
                "kot": kot,
                "match_score": {
                    "team1": self.match_score[TEAM1],
                    "team2": self.match_score[TEAM2],
                },
                "hand_index": snap["hand_index"],
                "hand_id": snap["hand_id"],
            }
            if self._flagged_tricks:
                record["flagged_tricks"] = sorted(self._flagged_tricks)
            # Remember the hand only if it actually reached disk; otherwise a
            # late flag has no record to amend and just stays in memory.
            if RECORDER.record_hand(record):
                self._recorded_hand_id = snap["hand_id"]
        except Exception:
            # Recording must never break live play.
            pass

    def _sort_human_hand(self) -> None:
        """Sort every human seat's hand (suit, then rank) for display."""
        if not self.game:
            return
        for seat in self.human_seats:
            hand = self.game.players[seat].hand
            if hand:
                hand.sort(key=lambda c: (suits.index(c.suit), ranks.index(c.rank)))

    def _cover_choice(self, seat: int):
        """
        Choose a card for an idle human's seat.

        That seat's player object is an `EnhancedPlayer(is_human=True)`, whose
        `play_card` refuses to decide for itself, so a rule-based stand-in is
        bound to the seat's live context for the one decision. `_get_teammate`
        returns `team[1]` when `team[0]` is the caller, so the stand-in goes
        first and the seat's real partner (two seats along) second.
        """
        g, player = self.game, self.game.players[seat]
        if self._stand_in is None:
            self._stand_in = HeuristicAgent("idle-cover")
        stand_in = self._stand_in
        stand_in.hand = player.hand
        stand_in.trump_suit = g.trump_suit
        stand_in.current_trick = g.current_trick
        stand_in.lead_suit = g.lead_suit
        stand_in.team = [stand_in, g.players[(seat + 2) % SEAT_COUNT]]
        card, _ = stand_in.play_card(g.lead_suit)
        return card

    def _run_ai_turns(self) -> List[Dict[str, Any]]:
        g = self.game
        events: List[Dict[str, Any]] = []
        for _ in range(200):
            # Resolve a completed trick *before* testing for hand end: after
            # the 13th trick every hand is empty, and checking first used to
            # drop that trick on the floor (a 6-6 hand then looked like a draw).
            if len(g.current_trick) == 4:
                winner, snapshot = g.resolve_trick_if_complete()
                if winner is not None:
                    events.append(
                        {"type": "trick", "winner": winner.name, "trick": snapshot}
                    )
                    self.revision += 1
                continue
            if self._hand_over():
                return events
            nxt = g.get_next_to_play()
            seat = g.players.index(nxt)
            if not self._ai_controls(seat):
                return events
            if seat in self.human_seats:
                card = self._cover_choice(seat)
            else:
                card, _ = nxt.play_card(g.lead_suit)
            err = g.apply_play(nxt, card)
            if err:
                raise RuntimeError(f"AI play failed ({nxt.name}): {err}")
            self.revision += 1
            events.append({"type": "play", "player": nxt.name, "card": str(card)})
        raise RuntimeError("AI turn loop exceeded safety limit")

    def _match_payload(self) -> Dict[str, Any]:
        return {
            "match_score": dict(self.match_score),
            "match_target": self.match_target,
            "match_over": self.match_over,
            "match_winner": self.match_winner,
        }

    def _choose_trump_payload(self, seat: int) -> Dict[str, Any]:
        """
        The waiting-for-the-Hakem view, as seen from `seat`.

        Only the Hakem's own seat is told the Hakem's five cards; every other
        seat gets empty lists, so the deal never leaks across the table.
        """
        game = self.game
        self._sort_human_hand()
        hakem_hand = game.players[self._hakem_seat()].hand
        mine = seat == self._hakem_seat()
        return {
            "phase": "choose_trump",
            "hakem": game.hakem.name,
            "hakem_cards": [str(c) for c in hakem_hand] if mine else [],
            "trump_options": (
                sorted({c.suit for c in hakem_hand}, key=suits.index) if mine else []
            ),
            "scores": self._scores(),
            "seats": self._seats(),
            **self._match_payload(),
        }

    def _payload(
        self,
        seat: int = 0,
        *,
        last_events: Optional[List[Dict[str, Any]]] = None,
        trick_before_ai: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Build the play-phase payload as seen from `seat`.

        `scores` are the *tricks* of the hand in progress; `match_score` counts
        *hands* won in the match so far. When the hand is over the payload also
        carries `hand_result` (winning team of the hand), `kot`, `match_over`
        and `match_winner`.

        **Kot scoring decision.** A 7–0 sweep ("Kot", کت) is scored as **2
        points** in `match_score` — the most widely played convention. Some
        tables score a Kot as 3, or double only when the losing side is the
        Hakem's team; this service deliberately implements the simple ×2 rule
        and exposes `kot` so clients can badge it. The engine (`hokm.py`) is
        untouched and remains Kot-agnostic.
        """
        g = self.game
        me = g.players[seat]
        self._sort_human_hand()
        if self._hand_over():
            self._finish_hand_if_over()
        your_turn = (not self._hand_over()) and g.get_next_to_play() is me
        legal = [str(c) for c in g.legal_cards_for_player(me)] if your_turn else []
        hand_over = self._hand_over()
        result = self.hand_result or {}
        payload: Dict[str, Any] = {
            "scores": self._scores(),
            "seats": self._seats(),
            "hand": [str(c) for c in me.hand],
            "legal_cards": legal,
            "current_trick": [
                {"player": p.name, "card": str(c)} for p, c in g.current_trick
            ],
            "trump_suit": g.trump_suit,
            # While the hand runs this is the Hakem; once it ends the engine has
            # already rotated, so report the hand's Hakem and the next one apart.
            "hakem": result.get("hakem") or (g.hakem.name if g.hakem else None),
            "next_player": None if hand_over else g.get_next_to_play().name,
            "your_turn": your_turn,
            "game_over": hand_over,  # "game" == one hand, kept for compatibility
            "last_events": last_events or [],
            "opponent_card_counts": {
                seat: len(p.hand)
                for seat, p in zip(
                    ["south", "east", "north", "west"], g.players
                )
            },
            **self._match_payload(),
        }
        if trick_before_ai is not None:
            payload["trick_before_ai"] = trick_before_ai
        if hand_over:
            t1, t2 = g.scores[1], g.scores[2]
            payload["result"] = (
                "Team 1 wins" if t1 > t2 else "Team 2 wins" if t2 > t1 else "Draw"
            )
            # Team 1 is seats 0 and 2, Team 2 is seats 1 and 3 (engine order),
            # so "did I win" depends on the seat asking, not on a fixed side.
            payload["you_won"] = t1 > t2 if seat % 2 == 0 else t2 > t1
            payload["hand_result"] = result.get("hand_result")
            payload["kot"] = bool(result.get("kot"))
            payload["next_hakem"] = g.hakem.name if g.hakem else None
        return payload


class SessionStore:
    """Thread-safe user_id → GameSession map with TTL eviction."""

    def __init__(self):
        self._sessions: Dict[str, GameSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, user_id: str, display_name: str) -> GameSession:
        with self._lock:
            self._evict_locked()
            sess = self._sessions.get(user_id)
            if sess is None:
                if len(self._sessions) >= MAX_SESSIONS:
                    raise GameServiceError(
                        "Server is at capacity — please try again in a few minutes."
                    )
                sess = GameSession(user_id, display_name)
                self._sessions[user_id] = sess
            sess.touch()
            if display_name:
                sess.display_name = display_name
            return sess

    def _evict_locked(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        stale = [k for k, s in self._sessions.items() if s.last_seen < cutoff]
        for k in stale:
            del self._sessions[k]

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

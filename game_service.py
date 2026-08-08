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
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from baselines import HeuristicAgent
from enhanced_player import EnhancedPlayer
from game_constants import Card, STATE_DIM, ACTION_DIM, ranks, suits
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

# Seat order in Hokm([human, a, b, c]) is south, east, north, west — the
# human's partner is players[2] (north). Labels must match those seats.
_AI_LABELS = ["AI East", "AI North", "AI West"]

TEAM1, TEAM2 = "Team 1", "Team 2"


class GameServiceError(Exception):
    """User-visible game/service errors (turned into JSON error responses)."""


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
    One human-vs-3-AI **match** bound to a single user.

    A match is a sequence of hands played by the same four seats. Hand state
    lives on `self.game` (the `Hokm` engine instance, which models exactly
    one hand); match state lives here.
    """

    def __init__(self, user_id: str, display_name: str):
        self.user_id = user_id
        self.display_name = display_name or "You"
        self.lock = threading.Lock()
        self.last_seen = time.time()
        self.game: Optional[Hokm] = None
        self.human: Optional[EnhancedPlayer] = None
        self.match_target: int = _match_target()
        self.match_score: Dict[str, int] = {TEAM1: 0, TEAM2: 0}
        self.match_over: bool = False
        self.match_winner: Optional[str] = None
        # Result of the hand currently on the table once it has ended:
        # {"hand_result": "Team 1"|"Team 2"|None, "kot": bool}. None while a
        # hand is still in progress.
        self.hand_result: Optional[Dict[str, Any]] = None

    # ---------- lifecycle ----------

    def touch(self) -> None:
        self.last_seen = time.time()

    def new_game(self) -> Dict[str, Any]:
        """Start a brand-new match (match score 0–0, Hakem drawn at random)."""
        human = EnhancedPlayer(
            self.display_name, STATE_DIM, ACTION_DIM, is_human=True
        )
        human.learning_enabled = False
        ai_players = [_build_ai_seat(label) for label in _AI_LABELS]
        # minimal_logging: a session now spans a whole match, and nothing here
        # reads `game_log`; skip the per-hand pandas concat.
        game = Hokm([human] + ai_players, minimal_logging=True)
        self.game, self.human = game, human
        self.match_target = _match_target()
        self.match_score = {TEAM1: 0, TEAM2: 0}
        self.match_over = False
        self.match_winner = None
        self.hand_result = None
        return self._deal_hand()

    def next_hand(self) -> Dict[str, Any]:
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
        return self._deal_hand()

    def _deal_hand(self) -> Dict[str, Any]:
        """Shuffle, deal, and resolve the trump decision for one hand."""
        game, human = self._require_game()
        # `Hokm.start_game()` does not clear the previous hand's trump; the
        # service uses `trump_suit is None` as its "waiting for Hakem" signal,
        # so reset it here rather than patching the engine.
        game.trump_suit = None
        game.game_count += 1
        game.start_game()
        if game.hakem == human:
            self._sort_human_hand()
            return {
                "status": "success",
                "phase": "choose_trump",
                "hakem": game.hakem.name,
                "hakem_cards": [str(c) for c in human.hand],
                "trump_options": sorted(
                    {c.suit for c in human.hand}, key=suits.index
                ),
                "scores": self._scores(),
                "seats": self._seats(),
                **self._match_payload(),
                "message": "You are Hakem — choose the trump suit.",
            }

        game.choose_trump_suit()
        events = self._run_ai_turns()
        return {
            "status": "success",
            "phase": "playing",
            "message": f"{game.hakem.name} is Hakem; trump is {game.trump_suit}.",
            **self._payload(last_events=events, trick_before_ai=[]),
        }

    def set_trump(self, trump_suit: str) -> Dict[str, Any]:
        game, human = self._require_game()
        if game.trump_suit:
            raise GameServiceError("Trump has already been chosen.")
        if game.hakem != human:
            raise GameServiceError("Only the Hakem chooses trump.")
        if trump_suit not in suits:
            raise GameServiceError(f"Invalid suit: {trump_suit}")
        if trump_suit not in {c.suit for c in human.hand}:
            raise GameServiceError("Trump must be one of your dealt suits.")
        game.set_trump_suit(trump_suit)
        events = self._run_ai_turns()
        return {
            "status": "success",
            "phase": "playing",
            "message": f"Trump suit set to {trump_suit}.",
            **self._payload(last_events=events, trick_before_ai=[]),
        }

    def play_card(self, card_str: str) -> Dict[str, Any]:
        game, human = self._require_game()
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
        if game.get_next_to_play() != human:
            raise GameServiceError("Not your turn.")
        err = game.apply_play(human, card)
        if err:
            raise GameServiceError(err)

        trick_before_ai = [
            {"player": p.name, "card": str(c)} for p, c in game.current_trick
        ]
        events = self._run_ai_turns()
        body = self._payload(
            last_events=events, trick_before_ai=trick_before_ai
        )
        body["status"] = "success"
        body["phase"] = "ended" if body["game_over"] else "playing"
        return body

    def state(self) -> Dict[str, Any]:
        if not self.game:
            return {"status": "success", "phase": "idle"}
        game, human = self._require_game()
        if not game.trump_suit:
            self._sort_human_hand()
            return {
                "status": "success",
                "phase": "choose_trump",
                "hakem": game.hakem.name,
                "hakem_cards": [str(c) for c in human.hand],
                "trump_options": sorted(
                    {c.suit for c in human.hand}, key=suits.index
                ),
                "scores": self._scores(),
                "seats": self._seats(),
                **self._match_payload(),
            }
        events = self._run_ai_turns()
        body = self._payload(last_events=events, trick_before_ai=[])
        body["status"] = "success"
        body["phase"] = "ended" if body["game_over"] else "playing"
        return body

    # ---------- internals (mirrors app.py's single-game helpers) ----------

    def _require_game(self):
        if not self.game or not self.human:
            raise GameServiceError("No game in progress — start a new game.")
        return self.game, self.human

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
        g.update_last_winning_team()
        g.rotate_hakem()

    def _sort_human_hand(self) -> None:
        if self.human and self.human.hand:
            self.human.hand.sort(
                key=lambda c: (suits.index(c.suit), ranks.index(c.rank))
            )

    def _run_ai_turns(self) -> List[Dict[str, Any]]:
        g, human = self.game, self.human
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
                continue
            if self._hand_over():
                return events
            nxt = g.get_next_to_play()
            if nxt == human:
                return events
            card, _ = nxt.play_card(g.lead_suit)
            err = g.apply_play(nxt, card)
            if err:
                raise RuntimeError(f"AI play failed ({nxt.name}): {err}")
            events.append({"type": "play", "player": nxt.name, "card": str(card)})
        raise RuntimeError("AI turn loop exceeded safety limit")

    def _match_payload(self) -> Dict[str, Any]:
        return {
            "match_score": dict(self.match_score),
            "match_target": self.match_target,
            "match_over": self.match_over,
            "match_winner": self.match_winner,
        }

    def _payload(
        self,
        *,
        last_events: Optional[List[Dict[str, Any]]] = None,
        trick_before_ai: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Build the play-phase payload.

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
        g, human = self.game, self.human
        self._sort_human_hand()
        if self._hand_over():
            self._finish_hand_if_over()
        your_turn = (not self._hand_over()) and g.get_next_to_play() == human
        legal = (
            [str(c) for c in g.legal_cards_for_player(human)] if your_turn else []
        )
        hand_over = self._hand_over()
        result = self.hand_result or {}
        payload: Dict[str, Any] = {
            "scores": self._scores(),
            "seats": self._seats(),
            "hand": [str(c) for c in human.hand],
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
            payload["you_won"] = t1 > t2  # human is always on Team 1
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

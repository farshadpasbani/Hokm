"""
Multi-session Hokm game service for the Telegram Mini App backend.

`app.py` keeps a single global game for local development; this module is
the production counterpart: one `GameSession` per authenticated Telegram
user, with per-session locks (gunicorn runs threaded), TTL eviction, and a
cap on concurrent sessions.

AI seats:
  * With no checkpoint configured, seats use `HeuristicAgent` — a
    deterministic rule-based opponent that plays sensible Hokm. This is the
    production default because untrained NFSP weights play near-randomly.
  * Set MODEL_PATH to an NFSP checkpoint (.pth) to seat greedy
    (ε = η = 0, learning disabled) trained agents instead, matching the
    evaluation-time protocol.
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

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(2 * 60 * 60)))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))


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

# Seat order in Hokm([human, a, b, c]) is south, east, north, west — the
# human's partner is players[2] (north). Labels must match those seats.
_AI_LABELS = ["AI East", "AI North", "AI West"]


class GameServiceError(Exception):
    """User-visible game/service errors (turned into JSON error responses)."""


def _build_ai_seat(label: str) -> EnhancedPlayer:
    if MODEL_PATH and os.path.isfile(MODEL_PATH):
        p = EnhancedPlayer(label, STATE_DIM, ACTION_DIM, epsilon=0.0, eta=0.0)
        p.learning_enabled = False
        p.load_policy_state(MODEL_PATH)
        return p
    p = HeuristicAgent(label)
    return p


class GameSession:
    """One human-vs-3-AI game bound to a single user."""

    def __init__(self, user_id: str, display_name: str):
        self.user_id = user_id
        self.display_name = display_name or "You"
        self.lock = threading.Lock()
        self.last_seen = time.time()
        self.game: Optional[Hokm] = None
        self.human: Optional[EnhancedPlayer] = None

    # ---------- lifecycle ----------

    def touch(self) -> None:
        self.last_seen = time.time()

    def new_game(self) -> Dict[str, Any]:
        human = EnhancedPlayer(
            self.display_name, STATE_DIM, ACTION_DIM, is_human=True
        )
        human.learning_enabled = False
        ai_players = [_build_ai_seat(label) for label in _AI_LABELS]
        game = Hokm([human] + ai_players)
        self.game, self.human = game, human

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
        if body["game_over"]:
            self.game, self.human = None, None
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
            }
        events = self._run_ai_turns()
        body = self._payload(last_events=events, trick_before_ai=[])
        body["status"] = "success"
        body["phase"] = "ended" if body["game_over"] else "playing"
        if body["game_over"]:
            self.game, self.human = None, None
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

    def _game_over(self) -> bool:
        g = self.game
        return (
            g.scores[1] >= 7
            or g.scores[2] >= 7
            or all(len(pl.hand) == 0 for pl in g.players)
        )

    def _sort_human_hand(self) -> None:
        if self.human and self.human.hand:
            self.human.hand.sort(
                key=lambda c: (suits.index(c.suit), ranks.index(c.rank))
            )

    def _run_ai_turns(self) -> List[Dict[str, Any]]:
        g, human = self.game, self.human
        events: List[Dict[str, Any]] = []
        for _ in range(200):
            if self._game_over():
                return events
            if len(g.current_trick) == 4:
                winner, snapshot = g.resolve_trick_if_complete()
                if winner is not None:
                    events.append(
                        {"type": "trick", "winner": winner.name, "trick": snapshot}
                    )
                continue
            nxt = g.get_next_to_play()
            if nxt == human:
                return events
            card, _ = nxt.play_card(g.lead_suit)
            err = g.apply_play(nxt, card)
            if err:
                raise RuntimeError(f"AI play failed ({nxt.name}): {err}")
            events.append({"type": "play", "player": nxt.name, "card": str(card)})
        raise RuntimeError("AI turn loop exceeded safety limit")

    def _payload(
        self,
        *,
        last_events: Optional[List[Dict[str, Any]]] = None,
        trick_before_ai: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        g, human = self.game, self.human
        self._sort_human_hand()
        your_turn = (not self._game_over()) and g.get_next_to_play() == human
        legal = (
            [str(c) for c in g.legal_cards_for_player(human)] if your_turn else []
        )
        payload: Dict[str, Any] = {
            "scores": self._scores(),
            "seats": self._seats(),
            "hand": [str(c) for c in human.hand],
            "legal_cards": legal,
            "current_trick": [
                {"player": p.name, "card": str(c)} for p, c in g.current_trick
            ],
            "trump_suit": g.trump_suit,
            "hakem": g.hakem.name if g.hakem else None,
            "next_player": None if self._game_over() else g.get_next_to_play().name,
            "your_turn": your_turn,
            "game_over": self._game_over(),
            "last_events": last_events or [],
            "opponent_card_counts": {
                seat: len(p.hand)
                for seat, p in zip(
                    ["south", "east", "north", "west"], g.players
                )
            },
        }
        if trick_before_ai is not None:
            payload["trick_before_ai"] = trick_before_ai
        if payload["game_over"]:
            t1, t2 = g.scores[1], g.scores[2]
            payload["result"] = (
                "Team 1 wins" if t1 > t2 else "Team 2 wins" if t2 > t1 else "Draw"
            )
            payload["you_won"] = t1 > t2  # human is always on Team 1
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

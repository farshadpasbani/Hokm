"""
Persistent recording of finished hands as RL/imitation training material.

Every completed hand in the Mini App is appended as one JSON line to
`{GAME_DATA_DIR}/hands_YYYYMMDD.jsonl`. Records store the *generative*
facts of the hand rather than derived features, so any future training
pipeline can reconstruct exactly what it needs:

  {
    "v": 1,                       # record schema version
    "ts": 1754600000,             # unix seconds at hand completion
    "user": "tg:12345",           # session owner (human always sits seat 0)
    "player_name": "Fara",
    "ai_kind": "pimc",            # opponent family for seats 1-3
    "hakem_seat": 2,
    "trump": "Hearts",
    "trump_chosen_by_human": false,
    "initial_hands": [[...13 cards...], x4],   # after the deal, before play
    "plays": [[seat, "Ace of Hearts"], ...],   # exact order of play
    "scores": {"team1": 7, "team2": 4},        # tricks
    "winner_team": 1,
    "kot": false,
    "match_score": {"team1": 3, "team2": 1},   # after this hand
    "hand_index": 4                            # 1-based within the match
  }

`replay_hand(record)` rebuilds the hand through the real engine step by
step and yields `(game, seat, card)` immediately BEFORE each play is
applied — at that instant `game`'s public state (current_trick, void_map,
cards_played_this_hand, play_log_this_hand) and every player's hand are
exactly as the mover saw them, so consumers can extract observations for
any seat (e.g. imitation-learning targets for the human's decisions).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from game_constants import Card

GAME_DATA_DIR = os.getenv("GAME_DATA_DIR", "game_data")
# JSONL files need a persistent disk. When `DATABASE_URL` is set (store.py)
# hands go to Postgres instead, and writing them to the container's ephemeral
# filesystem as well would only fill it up — so default local recording off.
# `GAME_RECORDING=1` forces both sinks on; `GAME_RECORDING=0` forces this one off.
RECORDING_ENABLED = (
    os.getenv("GAME_RECORDING", "0" if os.getenv("DATABASE_URL") else "1") == "1"
)

SCHEMA_VERSION = 1


def stamp(record: Dict[str, Any]) -> Dict[str, Any]:
    """Add the schema version and completion timestamp every sink shares."""
    return {"v": SCHEMA_VERSION, "ts": int(time.time()), **record}


class GameRecorder:
    """Thread-safe append-only JSONL writer, one file per UTC day."""

    def __init__(self, directory: str = GAME_DATA_DIR, enabled: bool = RECORDING_ENABLED):
        self.directory = directory
        self.enabled = enabled
        self._lock = threading.Lock()

    def _path_for_now(self) -> str:
        day = time.strftime("%Y%m%d", time.gmtime())
        return os.path.join(self.directory, f"hands_{day}.jsonl")

    def record_hand(self, record: Dict[str, Any]) -> Optional[str]:
        """Append one hand record. Returns the file path, or None when
        disabled or on write failure — recording must never break a game."""
        if not self.enabled:
            return None
        record = stamp(record)
        try:
            os.makedirs(self.directory, exist_ok=True)
            path = self._path_for_now()
            line = json.dumps(record, separators=(",", ":"))
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return path
        except OSError:
            return None

    def files(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        return sorted(
            os.path.join(self.directory, f)
            for f in os.listdir(self.directory)
            if f.startswith("hands_") and f.endswith(".jsonl")
        )

    def stats(self) -> Dict[str, Any]:
        files = self.files()
        total = 0
        for p in files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    total += sum(1 for _ in f)
            except OSError:
                continue
        return {
            "enabled": self.enabled,
            "directory": self.directory,
            "files": [os.path.basename(p) for p in files],
            "hands_recorded": total,
        }


def load_hands(directory: str = GAME_DATA_DIR) -> Iterator[Dict[str, Any]]:
    """Yield every recorded hand across all files, skipping corrupt lines."""
    rec = GameRecorder(directory, enabled=True)
    for path in rec.files():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Replay: reconstruct a recorded hand through the real engine.
# ---------------------------------------------------------------------------

class _ReplaySeat:
    """Minimal seat object satisfying the engine's step-API surface."""

    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []
        self.current_trick: list = []
        self.lead_suit = None
        self.trump_suit = None
        self.trump_state = [0] * 4
        self.team = None
        self.tricks_won: dict = {}
        self.team_strategy = None
        self.played_suit_counts = [0] * 4

    def update_trump_suit(self, trump_suit):
        from game_constants import suits as _suits

        self.trump_suit = trump_suit
        self.trump_state = [0] * 4
        if trump_suit:
            self.trump_state[_suits.index(trump_suit)] = 1

    def reset(self):
        self.hand = []
        self.current_trick = []
        self.lead_suit = None

    def _sync_seats(self, game):
        return None


def replay_hand(record: Dict[str, Any]) -> Iterator[Tuple[Any, int, Card]]:
    """Yield (game, seat, card) before each play of a recorded hand.

    The engine state at yield time is exactly the mover's decision point.
    Raises ValueError if the record replays inconsistently (illegal play),
    which doubles as an integrity check on stored data.
    """
    from hokm import Hokm

    seats = [_ReplaySeat(f"Seat {i}") for i in range(4)]
    game = Hokm(seats, minimal_logging=True)
    game.reset_players()
    for i, cards in enumerate(record["initial_hands"]):
        seats[i].hand = [Card.from_string(c) for c in cards]
    game.hakem = seats[int(record["hakem_seat"])]
    game.trump_suit = record["trump"]
    for s in seats:
        s.update_trump_suit(record["trump"])
    game.trick_starter_index = int(record["hakem_seat"])
    game.cards_played_this_hand = []
    game.play_log_this_hand = []
    game.void_map = {p: set() for p in seats}
    game._sync_player_trick_context()

    for seat_idx, card_str in record["plays"]:
        if len(game.current_trick) == 4:
            game.resolve_trick_if_complete()
        card = Card.from_string(card_str)
        mover = seats[int(seat_idx)]
        yield game, int(seat_idx), card
        err = game.apply_play(mover, card)
        if err:
            raise ValueError(
                f"recorded hand replays illegally at {card_str} by seat "
                f"{seat_idx}: {err}"
            )
    if len(game.current_trick) == 4:
        game.resolve_trick_if_complete()

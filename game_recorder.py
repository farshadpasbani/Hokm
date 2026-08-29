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
    "hand_index": 4,                           # 1-based within the match
    "hand_id": "9f3c…",                        # unique; links flag amendments
    "flagged_tricks": [6, 11]                  # optional; omitted when empty
  }

`flagged_tricks` is the tester's "AI is stupid" signal: 0-based indices of
tricks a human marked as bad play. The record is written the instant the hand
ends, yet the final trick can still be flagged while the end-of-hand screen
is up, so those late flags go into the same file as an amendment line:

  {"v": 1, "ts": …, "type": "flag", "hand_id": "9f3c…", "tricks": [12]}

`load_hands()` folds amendments into the hand record they name and never
yields them alone. Appending rather than rewriting the hand record in place
means a crash can never corrupt an already-written hand.

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
RECORDING_ENABLED = os.getenv("GAME_RECORDING", "1") == "1"

SCHEMA_VERSION = 1


class GameRecorder:
    """Thread-safe append-only JSONL writer, one file per UTC day."""

    def __init__(self, directory: str = GAME_DATA_DIR, enabled: bool = RECORDING_ENABLED):
        self.directory = directory
        self.enabled = enabled
        self._lock = threading.Lock()

    def _path_for_now(self) -> str:
        day = time.strftime("%Y%m%d", time.gmtime())
        return os.path.join(self.directory, f"hands_{day}.jsonl")

    def _append(self, record: Dict[str, Any]) -> Optional[str]:
        """Append one JSON line. Returns the file path, or None when disabled
        or on write failure — recording must never break a game."""
        if not self.enabled:
            return None
        record = {"v": SCHEMA_VERSION, "ts": int(time.time()), **record}
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

    def record_hand(self, record: Dict[str, Any]) -> Optional[str]:
        """Append one finished-hand record."""
        return self._append(record)

    def record_flags(
        self, hand_id: str, trick_indices: List[int]
    ) -> Optional[str]:
        """Append a flag amendment for an already-written hand record.

        Carries the hand's *complete* flag set so folding amendments is a
        union — replaying them in any order gives the same result.
        """
        return self._append({
            "type": "flag",
            "hand_id": hand_id,
            "tricks": sorted(set(trick_indices)),
        })

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
        # Flag amendments share the file with hand records, so count records
        # rather than lines — `hands_recorded` must stay a count of hands.
        total = sum(1 for r in _iter_lines(files) if r.get("type") != "flag")
        return {
            "enabled": self.enabled,
            "directory": self.directory,
            "files": [os.path.basename(p) for p in files],
            "hands_recorded": total,
        }


def _iter_lines(paths: List[str]) -> Iterator[Dict[str, Any]]:
    """Yield every parsable JSON object across `paths`, skipping corrupt lines."""
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except OSError:
            continue


def load_hands(directory: str = GAME_DATA_DIR) -> Iterator[Dict[str, Any]]:
    """Yield every recorded hand across all files, skipping corrupt lines.

    Flag amendments (see the module docstring) are merged into the hand
    record they name and never yielded on their own, so every yielded object
    is a replayable hand record.
    """
    paths = GameRecorder(directory, enabled=True).files()
    # Two passes: an amendment is always appended *after* its hand record, so
    # the flags have to be collected before the records can be yielded.
    amendments: Dict[str, set] = {}
    for obj in _iter_lines(paths):
        if obj.get("type") != "flag":
            continue
        hand_id, tricks = obj.get("hand_id"), obj.get("tricks")
        if not isinstance(hand_id, str) or not isinstance(tricks, list):
            continue  # malformed amendment — ignore rather than crash
        amendments.setdefault(hand_id, set()).update(
            t for t in tricks if isinstance(t, int)
        )
    for obj in _iter_lines(paths):
        if obj.get("type") == "flag":
            continue
        # Orphan amendments (no matching hand_id) simply never get read.
        extra = amendments.get(obj.get("hand_id"))
        if extra:
            obj["flagged_tricks"] = sorted(
                set(obj.get("flagged_tricks") or []) | extra
            )
        yield obj


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

"""
Multi-human Hokm tables for the Telegram Mini App backend.

A table is four seats sharing one `GameSession`. Two to four humans take
seats; every seat still empty when the table starts is played by AI. The
table owns identity and time — which user sits where, and when each of them
was last seen, and therefore which seats the AI must cover so play never
stalls on someone who put their phone down. The session owns the cards.

  * **Seat ownership.** A player acts through their `user_id`; the table maps
    that to their seat. Nobody can name a seat they do not own, and the
    session refuses an action whose seat is not the one to move, so a
    wrong-seat or wrong-turn request is rejected rather than merely hidden.

  * **Private hands.** Every payload is built for one seat: shared table facts
    plus that seat's own cards. No other player's hand is ever serialised.

  * **Polling, not streaming.** The service runs `gunicorn --workers 1
    --threads 8`, so a held-open stream would pin one of eight threads per
    connected player. Instead the table carries a monotonically increasing
    `version`, and `GET /api/table/state?since=<version>` returns at once when
    the version has moved and otherwise parks for at most
    `TABLE_POLL_TIMEOUT_SECONDS`.

  * **Concurrency.** Two players of one table land on two threads at once, so
    every table has its own lock and every mutation runs under it.
    `TableStore`'s lock is never held while a table lock is taken.

State is in-memory, deliberately: a redeploy ends in-flight tables, which is
why tables carry a TTL rather than a persistence layer.

Environment:
  BOT_USERNAME              Bot username used to build the join deep link.
  MINI_APP_SHORT_NAME       Mini App short name; enables a `startapp` link.
  TABLE_IDLE_SECONDS        Silence before the AI covers a seat (default 45).
  TABLE_POLL_TIMEOUT_SECONDS  Long-poll parking time (default 3).
  TABLE_TTL_SECONDS         Idle table eviction (default 2 h).
  MAX_TABLES                Concurrent table cap (default 200).
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from typing import Any, Dict, List, Optional

from game_service import (
    SEAT_COUNT,
    GameServiceError,
    GameSession,
    _AI_SEAT_LABELS,
)

TABLE_TTL_SECONDS = int(os.getenv("TABLE_TTL_SECONDS", str(2 * 60 * 60)))
MAX_TABLES = int(os.getenv("MAX_TABLES", "200"))
TABLE_IDLE_SECONDS = float(os.getenv("TABLE_IDLE_SECONDS", "45"))
POLL_TIMEOUT_SECONDS = float(os.getenv("TABLE_POLL_TIMEOUT_SECONDS", "3"))

MIN_HUMANS_TO_START = 2
CODE_LENGTH = 6
# No I, O, 0 or 1: a join code gets read off one screen and typed into another.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def clean_code(raw: str) -> str:
    """A join code as typed or pasted anywhere, or "" when it is not ours."""
    code = (raw or "").strip().upper()
    if len(code) != CODE_LENGTH or any(c not in _CODE_ALPHABET for c in code):
        return ""
    return code


def deep_link(code: str) -> str:
    """
    Telegram link that opens the Mini App on this table.

    Read from the environment at call time so tests and redeploys can change
    it without a restart. Returns "" when no bot username is configured; the
    join code still works typed by hand.
    """
    bot = os.getenv("BOT_USERNAME", "").strip().lstrip("@")
    if not bot:
        return ""
    short = os.getenv("MINI_APP_SHORT_NAME", "").strip()
    if short:
        return f"https://t.me/{bot}/{short}?startapp={code}"
    return f"https://t.me/{bot}?start={code}"


class Seat:
    """One human-occupied seat. An empty seat is `None` in `Table.seats`."""

    __slots__ = ("user_id", "name", "last_seen", "seated_at")

    def __init__(self, user_id: str, name: str):
        self.user_id = user_id
        self.name = name
        self.last_seen = time.time()
        # When this player joined the table, as opposed to `last_seen`, which
        # every request refreshes. Moving seat carries the whole Seat across,
        # so seniority survives a partner swap — see `_promote_host`.
        self.seated_at = self.last_seen


class Table:
    """
    Four seats, one shared match, one lock.

    Every public method takes the table lock, refreshes the caller's
    last-seen stamp, recomputes which seats the AI is covering, performs the
    action, and returns that caller's private view of the table.
    """

    def __init__(self, code: str, host_user_id: str, host_name: str):
        self.code = code
        self.host_user_id = host_user_id
        self.created_at = time.time()
        self.last_activity = self.created_at
        self.lock = threading.RLock()
        self._changed = threading.Condition(self.lock)
        self.version = 0
        self.seats: List[Optional[Seat]] = [None] * SEAT_COUNT
        self.seats[0] = Seat(host_user_id, host_name or "Host")
        self.session: Optional[GameSession] = None

    @property
    def host_seat(self) -> Optional[int]:
        """
        Where the host is sitting **now** — derived, never stored.

        The host starts on seat 0 but may move like anyone else (they are
        usually the one who shifts, to partner whoever turned up), so an
        index captured at creation goes stale the moment they do. Identity
        lives in `host_user_id`; the seat is only ever a lookup of it.

        `None` means the host is no longer seated, which leaves a table
        nobody can start — see the leave path in `TableStore`.
        """
        return self.seat_of(self.host_user_id)

    # ---------- seating ----------

    def join(self, user_id: str, name: str, seat: Optional[int] = None) -> int:
        """
        Take a free seat (lowest by default), or move to a named free one.

        Re-joining is a reclaim, not a second seat — a deep link opened twice
        must not shuffle anyone. Naming a seat you do not hold moves you to it
        while the table is still in the lobby: seats 0 and 2 are one team and
        1 and 3 the other, so choosing a seat is how two friends end up
        partners instead of opponents. Once cards are dealt a seat is fixed.
        """
        with self.lock:
            self.last_activity = time.time()
            mine = self.seat_of(user_id)
            if mine is not None:
                self.seats[mine].last_seen = time.time()
                if seat is None or seat == mine:
                    return mine
                if self.session is not None:
                    raise GameServiceError("That table has already started.")
                self._require_free_seat(seat)
                # Move the Seat object, so the mover keeps their name and
                # their last-seen stamp.
                self.seats[seat], self.seats[mine] = self.seats[mine], None
                self._bump()
                return seat
            if self.session is not None:
                raise GameServiceError("That table has already started.")
            if seat is None:
                seat = next(
                    (i for i, s in enumerate(self.seats) if s is None), None
                )
                if seat is None:
                    raise GameServiceError("That table is full.")
            else:
                self._require_free_seat(seat)
            self.seats[seat] = Seat(user_id, self._unique_name(name))
            self._bump()
            return seat

    def leave(self, user_id: str) -> bool:
        """
        Give up a seat. Returns True when the seat was actually freed.

        Before the match starts the seat opens up again. Once cards are out it
        cannot be — the hand would be unplayable — so the seat is marked idle
        instead and the AI covers it. Touching any table endpoint reclaims it.

        This is the one place a seat is ever vacated, so it is also where the
        table changes hands when the vacating player was the host.
        """
        with self.lock:
            seat = self._require_seat(user_id)
            self.last_activity = time.time()
            if self.session is None:
                self.seats[seat] = None
                self._promote_host()
                self._bump()
                return True
            self.seats[seat].last_seen = 0.0
            self._apply_idle_cover()
            self._bump()
            return False

    def seat_of(self, user_id: str) -> Optional[int]:
        for i, s in enumerate(self.seats):
            if s is not None and s.user_id == user_id:
                return i
        return None

    def human_count(self) -> int:
        return sum(1 for s in self.seats if s is not None)

    # ---------- play ----------

    def start(self, user_id: str) -> Dict[str, Any]:
        with self.lock:
            seat = self._require_seat(user_id)
            self._touch(user_id)
            if self.session is not None:
                raise GameServiceError("That table has already started.")
            if user_id != self.host_user_id:
                raise GameServiceError(
                    "Only the player who created the table can start it."
                )
            humans = {i: s.name for i, s in enumerate(self.seats) if s is not None}
            if len(humans) < MIN_HUMANS_TO_START:
                raise GameServiceError(
                    f"A table needs at least {MIN_HUMANS_TO_START} players to start."
                )
            before = self._fingerprint()
            # `user` identifies the match in the recorded training data; the
            # per-seat names travel with it via `human_seats`. The name comes
            # from the caller's own seat — only the host reaches this line,
            # and `_require_seat` has already proved that seat is occupied,
            # whereas seat 0 may be empty or hold somebody else entirely.
            self.session = GameSession(
                f"table:{self.code}",
                self.seats[seat].name,
                human_seats=humans,
            )
            self._apply_idle_cover()
            body = self.session.new_game(seat)
            self._bump_if_changed(before)
            return self._decorate(body, seat)

    def set_trump(self, user_id: str, trump_suit: str) -> Dict[str, Any]:
        return self._act(user_id, lambda s, seat: s.set_trump(trump_suit, seat))

    def play_card(self, user_id: str, card: str) -> Dict[str, Any]:
        return self._act(user_id, lambda s, seat: s.play_card(card, seat))

    def next_hand(self, user_id: str) -> Dict[str, Any]:
        return self._act(user_id, lambda s, seat: s.next_hand(seat))

    def flag_trick(self, user_id: str, trick_index: Any) -> Dict[str, Any]:
        """
        Flag one trick of the hand on this table as bad AI play.

        Deliberately not routed through `_act`. A flag is legal on the
        end-of-match sheet — which is exactly where the last trick of a match
        gets judged — and `_act` refuses every action once the match is over.

        The flag set belongs to the hand, not to the presser, so a second
        player flagging the same trick is told it is already flagged: one
        training label per trick, whoever pressed. The table's version is
        left alone — a flag changes no card and no seat, so waking every
        long-poll for it would be noise.
        """
        with self.lock:
            self._require_seat(user_id)
            self._touch(user_id)
            if self.session is None:
                raise GameServiceError("That table has not started yet.")
            return self.session.flag_trick(trick_index)

    def view(self, user_id: str) -> Dict[str, Any]:
        """
        This player's private view; also the table's clock tick.

        A poll advances any AI seat that owes a move — including seats the AI
        is covering for idle humans — which is what keeps a table with one
        attentive player moving.
        """
        with self.lock:
            seat = self._require_seat(user_id)
            self._touch(user_id)
            before = self._fingerprint()
            if self.session is None:
                body: Dict[str, Any] = {"status": "success", "phase": "lobby"}
            else:
                body = self.session.state(seat)
            self._bump_if_changed(before)
            return self._decorate(body, seat)

    def wait_for_change(self, since: int, timeout: float) -> None:
        """Park until `version` passes `since`, or `timeout` elapses."""
        with self.lock:
            if self.version > since:
                return
            self._changed.wait_for(lambda: self.version > since, timeout=timeout)

    # ---------- internals ----------

    def _act(self, user_id: str, action) -> Dict[str, Any]:
        with self.lock:
            seat = self._require_seat(user_id)
            self._touch(user_id)
            if self.session is None:
                raise GameServiceError("That table has not started yet.")
            if self.session.match_over:
                # The session would answer "start a new match", which is a
                # solo action a table has no way to perform: a table's seats
                # are fixed at the deal. Say what a player here can actually
                # do instead.
                raise GameServiceError(
                    "This match is over — leave the table to start a new one."
                )
            before = self._fingerprint()
            body = action(self.session, seat)
            self._bump_if_changed(before)
            return self._decorate(body, seat)

    def _require_seat(self, user_id: str) -> int:
        seat = self.seat_of(user_id)
        if seat is None:
            raise GameServiceError("You are not seated at that table.")
        return seat

    def _promote_host(self) -> None:
        """
        Hand the table to its longest-seated player once the host has gone.

        `start` is gated on `host_user_id`, so a lobby whose host left is one
        nobody can ever begin — every remaining player is told to wait for
        someone who is not coming.

        Seniority decides it, not seat order: seats change freely in the
        lobby, so ordering by index would quietly move the hostship every
        time two friends swapped to sit together. Ties fall to the lower
        seat, since `seats` is scanned in order.

        Lobby only, by construction — this runs on the branch of `leave` that
        vacates a seat, and once cards are dealt no seat is ever vacated. A
        table with nobody left keeps its departed host and is disposed of by
        `TableStore`; it is not resurrected here.
        """
        if self.seat_of(self.host_user_id) is not None:
            return
        remaining = [s for s in self.seats if s is not None]
        if not remaining:
            return
        self.host_user_id = min(remaining, key=lambda s: s.seated_at).user_id

    def _require_free_seat(self, seat: Any) -> None:
        if not isinstance(seat, int) or not 0 <= seat < SEAT_COUNT:
            raise GameServiceError("Seat must be 0, 1, 2 or 3.")
        if self.seats[seat] is not None:
            raise GameServiceError("That seat is taken.")

    def _touch(self, user_id: str) -> None:
        """Mark the caller present, then recompute who the AI is covering."""
        now = time.time()
        self.last_activity = now
        seat = self.seat_of(user_id)
        if seat is not None:
            self.seats[seat].last_seen = now
        # Same clock reading as the stamp above, so the caller can never be
        # judged idle by the very call that marks them present.
        self._apply_idle_cover(now)

    def _apply_idle_cover(self, now: Optional[float] = None) -> None:
        if self.session is None:
            return
        now = time.time() if now is None else now
        self.session.auto_seats = {
            i
            for i, s in enumerate(self.seats)
            if s is not None and now - s.last_seen > TABLE_IDLE_SECONDS
        }

    def _unique_name(self, name: str) -> str:
        """Keep seat names distinct — two "Ali"s at one table read as one."""
        base = (name or "Player").strip()[:32] or "Player"
        taken = {s.name for s in self.seats if s is not None}
        if base not in taken:
            return base
        for n in range(2, SEAT_COUNT + 2):
            candidate = f"{base} {n}"
            if candidate not in taken:
                return candidate
        return base

    def _fingerprint(self):
        """Cheap "has anything visible changed" probe for the version counter."""
        return (
            self.session.revision if self.session is not None else -1,
            tuple(s.user_id if s is not None else None for s in self.seats),
        )

    def _bump_if_changed(self, before) -> None:
        if self._fingerprint() != before:
            self._bump()

    def _bump(self) -> None:
        self.version += 1
        self._changed.notify_all()

    def _decorate(self, body: Dict[str, Any], seat: int) -> Dict[str, Any]:
        body["your_seat"] = seat
        body["table"] = self._table_block()
        return body

    def _table_block(self) -> Dict[str, Any]:
        # An empty seat is still "open" in the lobby; once the table starts it
        # is an AI seat, because starting is what fills the empty seats.
        now = time.time()
        players = [
            {
                "seat": i,
                "name": s.name if s else (_AI_SEAT_LABELS[i] if self.session else None),
                "kind": "human" if s else ("ai" if self.session else "open"),
                "idle": bool(s) and now - s.last_seen > TABLE_IDLE_SECONDS,
            }
            for i, s in enumerate(self.seats)
        ]
        return {
            "code": self.code,
            "version": self.version,
            "started": self.session is not None,
            "host_seat": self.host_seat,
            "join_url": deep_link(self.code),
            "players": players,
        }


class TableStore:
    """Thread-safe code → `Table` map, plus one table per user, with TTL."""

    def __init__(self):
        self._tables: Dict[str, Table] = {}
        self._by_user: Dict[str, str] = {}
        self._lock = threading.Lock()

    def create(self, user_id: str, display_name: str) -> Table:
        self._detach(user_id)
        with self._lock:
            self._evict_locked()
            if len(self._tables) >= MAX_TABLES:
                raise GameServiceError(
                    "Too many tables right now — please try again in a few minutes."
                )
            code = self._new_code_locked()
            table = Table(code, user_id, display_name)
            self._tables[code] = table
            self._by_user[user_id] = code
            return table

    def join(
        self,
        user_id: str,
        display_name: str,
        code: str,
        seat: Optional[int] = None,
    ) -> Table:
        code = (code or "").strip().upper()
        with self._lock:
            self._evict_locked()
            table = self._tables.get(code)
        if table is None:
            raise GameServiceError("No table with that code — check it and retry.")
        if table.seat_of(user_id) is None:
            self._detach(user_id)
        table.join(user_id, display_name, seat)
        with self._lock:
            self._by_user[user_id] = code
        return table

    def for_user(self, user_id: str) -> Table:
        table = self._table_for(user_id)
        if table is None:
            raise GameServiceError("You are not at a table.")
        return table

    def leave(self, user_id: str) -> Table:
        table = self.for_user(user_id)
        if table.leave(user_id):
            with self._lock:
                self._by_user.pop(user_id, None)
                if table.human_count() == 0:
                    self._tables.pop(table.code, None)
        return table

    def count(self) -> int:
        with self._lock:
            return len(self._tables)

    # ---------- internals ----------

    def _table_for(self, user_id: str) -> Optional[Table]:
        with self._lock:
            code = self._by_user.get(user_id)
            return self._tables.get(code) if code else None

    def _detach(self, user_id: str) -> None:
        """
        Drop a stale seat before this user sits somewhere else.

        Taken outside the store lock: `Table.leave` takes the table lock, and
        the two locks are never held at once.
        """
        table = self._table_for(user_id)
        if table is None:
            return
        try:
            table.leave(user_id)
        except GameServiceError:
            pass
        with self._lock:
            self._by_user.pop(user_id, None)
            if table.human_count() == 0:
                self._tables.pop(table.code, None)

    def _new_code_locked(self) -> str:
        for _ in range(50):
            code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if code not in self._tables:
                return code
        raise GameServiceError("Could not allocate a join code — try again.")

    def _evict_locked(self) -> None:
        cutoff = time.time() - TABLE_TTL_SECONDS
        stale = [c for c, t in self._tables.items() if t.last_activity < cutoff]
        for code in stale:
            del self._tables[code]
        if stale:
            gone = set(stale)
            for user_id in [u for u, c in self._by_user.items() if c in gone]:
                del self._by_user[user_id]

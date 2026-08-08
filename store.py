"""
Off-box persistence for finished hands and the player registry (Postgres).

Why this exists
---------------
`game_recorder.GameRecorder` appends hands to local JSONL files. That needs a
persistent disk, and the container filesystem is wiped on every deploy. This
module is the disk-free alternative: point `DATABASE_URL` at a managed
Postgres (Neon, Supabase, Render Postgres, …) and every finished hand plus
every player who launches the Mini App is written there instead.

Two tables:

  players  one row per user_id ("tg:<id>" or "guest:<id>"), carrying the
           Telegram handle when the user has one, plus first/last seen,
           launch count, and hands played. This is the "who has played"
           log — query it for totals or to reach people later.

  hands    one row per finished hand. `record` is the exact JSON document
           `GameRecorder` would have written to JSONL, so any training
           pipeline reads it unchanged (`SELECT record FROM hands`).

Design constraints
------------------
* **Never block a game.** Postgres writes go through a bounded queue drained
  by one background thread. A cold serverless database (Neon scales to zero
  and can take seconds to wake) delays the write, never the player's move.
  A full queue drops the newest record rather than applying backpressure —
  training data is nice to have, a stalled hand is not.
* **Never break a game.** Every failure path is swallowed and counted.
* **Disabled by default.** No `DATABASE_URL` → `enabled` is False and every
  entry point is a no-op, so local dev and tests need no database.
* **Testable without Postgres.** `Store` takes a `connect` factory; tests
  inject a fake connection and assert on the SQL that would have run.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from game_recorder import stamp

logger = logging.getLogger("hokm.store")

DATABASE_URL = os.getenv("DATABASE_URL", "")
# Bounded so a wedged database cannot grow memory without limit.
QUEUE_MAXSIZE = int(os.getenv("STORE_QUEUE_MAXSIZE", "2000"))
# A player row is refreshed at most this often, so ordinary API polling does
# not turn into one UPDATE per request.
PLAYER_TOUCH_INTERVAL = int(os.getenv("STORE_PLAYER_TOUCH_SECONDS", "300"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    user_id       TEXT PRIMARY KEY,
    telegram_id   BIGINT,
    username      TEXT,
    first_name    TEXT,
    last_name     TEXT,
    language_code TEXT,
    is_premium    BOOLEAN,
    first_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    launches      INTEGER     NOT NULL DEFAULT 1,
    hands_played  INTEGER     NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS hands (
    id      BIGSERIAL   PRIMARY KEY,
    ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id TEXT        NOT NULL,
    record  JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS hands_user_idx ON hands (user_id);
CREATE INDEX IF NOT EXISTS hands_ts_idx   ON hands (ts);
CREATE INDEX IF NOT EXISTS players_username_idx ON players (username);
"""

# A returning player keeps their original first_seen; everything else is
# refreshed, because handles and names change. COALESCE keeps a previously
# known handle when a later launch omits it.
UPSERT_PLAYER = """
INSERT INTO players (user_id, telegram_id, username, first_name, last_name,
                     language_code, is_premium)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (user_id) DO UPDATE SET
    telegram_id   = COALESCE(EXCLUDED.telegram_id, players.telegram_id),
    username      = COALESCE(EXCLUDED.username, players.username),
    first_name    = COALESCE(EXCLUDED.first_name, players.first_name),
    last_name     = COALESCE(EXCLUDED.last_name, players.last_name),
    language_code = COALESCE(EXCLUDED.language_code, players.language_code),
    is_premium    = COALESCE(EXCLUDED.is_premium, players.is_premium),
    last_seen     = now(),
    launches      = players.launches + 1
"""

INSERT_HAND = "INSERT INTO hands (user_id, record) VALUES (%s, %s)"

# A hand can arrive for a user whose row was never written (e.g. the player
# upsert was dropped by a full queue), so create the row if missing.
BUMP_HANDS = """
INSERT INTO players (user_id, hands_played) VALUES (%s, 1)
ON CONFLICT (user_id) DO UPDATE SET
    hands_played = players.hands_played + 1,
    last_seen    = now()
"""


def _default_connect(dsn: str):
    """Open a psycopg connection. Imported lazily so psycopg stays optional."""
    import psycopg  # noqa: PLC0415 — optional dependency, only needed here

    return psycopg.connect(dsn, autocommit=True)


class Store:
    """Queue-backed Postgres writer for hands and the player registry."""

    def __init__(
        self,
        dsn: str = DATABASE_URL,
        *,
        connect: Optional[Callable[[str], Any]] = None,
        queue_maxsize: int = QUEUE_MAXSIZE,
        touch_interval: int = PLAYER_TOUCH_INTERVAL,
    ):
        self.dsn = dsn
        self.enabled = bool(dsn)
        self._connect = connect or _default_connect
        self._touch_interval = touch_interval
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(queue_maxsize)
        self._conn = None
        self._schema_ready = False
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()
        self._touched: Dict[str, float] = {}
        self._touch_lock = threading.Lock()
        self.dropped = 0
        self.written = 0
        self.failed = 0
        self.retried = 0

    # ---------- public API (all no-ops when disabled) ----------

    def note_player(self, profile: Dict[str, Any], *, force: bool = False) -> bool:
        """
        Record that `profile["user_id"]` launched the app.

        Throttled per user (`PLAYER_TOUCH_INTERVAL`) so request polling does
        not become one write per request. Returns True when the write was
        enqueued. `force=True` skips the throttle.
        """
        if not self.enabled:
            return False
        user_id = profile.get("user_id")
        if not user_id:
            return False
        if not force and not self._should_touch(user_id):
            return False
        return self._enqueue(("player", profile))

    def record_hand(self, record: Dict[str, Any]) -> bool:
        """Persist one finished hand. Mirrors `GameRecorder.record_hand`."""
        if not self.enabled:
            return False
        return self._enqueue(("hand", stamp(record)))

    def stats(self) -> Dict[str, Any]:
        """Counters plus live totals. Never raises."""
        info: Dict[str, Any] = {
            "enabled": self.enabled,
            "queued": self._queue.qsize(),
            "written": self.written,
            "failed": self.failed,
            "retried": self.retried,
            "dropped": self.dropped,
        }
        if not self.enabled:
            return info
        rows = self._query(
            "SELECT (SELECT count(*) FROM players), (SELECT count(*) FROM hands)"
        )
        if rows:
            info["players_total"], info["hands_total"] = rows[0][0], rows[0][1]
        return info

    def players(self, limit: int = 500) -> List[Dict[str, Any]]:
        """The player log, most recently active first."""
        if not self.enabled:
            return []
        rows = self._query(
            "SELECT user_id, telegram_id, username, first_name, last_name,"
            " language_code, is_premium, first_seen, last_seen, launches,"
            " hands_played FROM players ORDER BY last_seen DESC LIMIT %s",
            (limit,),
        )
        cols = [
            "user_id", "telegram_id", "username", "first_name", "last_name",
            "language_code", "is_premium", "first_seen", "last_seen",
            "launches", "hands_played",
        ]
        return [
            {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in zip(cols, row)
            }
            for row in rows
        ]

    def iter_hands(self, batch: int = 500):
        """Yield every stored hand record (oldest first) for export."""
        if not self.enabled:
            return
        offset = 0
        while True:
            rows = self._query(
                "SELECT record FROM hands ORDER BY id LIMIT %s OFFSET %s",
                (batch, offset),
            )
            if not rows:
                return
            for (record,) in rows:
                yield record
            if len(rows) < batch:
                return
            offset += len(rows)

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the queue drains. For tests and shutdown."""
        if not self.enabled:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    # ---------- internals ----------

    def _should_touch(self, user_id: str) -> bool:
        now = time.time()
        with self._touch_lock:
            last = self._touched.get(user_id, 0.0)
            if now - last < self._touch_interval:
                return False
            self._touched[user_id] = now
            # The map is unbounded otherwise; entries are cheap but a long
            # uptime with many users would still grow it forever.
            if len(self._touched) > 10000:
                cutoff = now - self._touch_interval
                self._touched = {
                    k: v for k, v in self._touched.items() if v >= cutoff
                }
            return True

    def _enqueue(self, item: tuple) -> bool:
        self._ensure_worker()
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _ensure_worker(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="hokm-store", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                try:
                    self._dispatch(item)
                except Exception:
                    # Nearly always a connection the server closed while it
                    # was idle — serverless Postgres sleeps, and the failure
                    # only surfaces on the next write. Reconnect and retry
                    # once, so waking up does not cost the first hand after it.
                    self._close()
                    self._dispatch(item)
                    self.retried += 1
                self.written += 1
            except Exception:
                self.failed += 1
                self._close()
                logger.warning("store write failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _dispatch(self, item: tuple) -> None:
        kind, payload = item
        if kind == "player":
            self._write_player(payload)
        elif kind == "hand":
            self._write_hand(payload)

    def _write_player(self, profile: Dict[str, Any]) -> None:
        with self._connection().cursor() as cur:
            cur.execute(
                UPSERT_PLAYER,
                (
                    profile.get("user_id"),
                    profile.get("telegram_id"),
                    profile.get("username"),
                    profile.get("first_name"),
                    profile.get("last_name"),
                    profile.get("language_code"),
                    profile.get("is_premium"),
                ),
            )

    def _write_hand(self, record: Dict[str, Any]) -> None:
        import json  # noqa: PLC0415 — only needed on the write path

        user_id = record.get("user") or "unknown"
        with self._connection().cursor() as cur:
            cur.execute(INSERT_HAND, (user_id, json.dumps(record)))
            cur.execute(BUMP_HANDS, (user_id,))

    def _connection(self):
        if self._conn is None:
            self._conn = self._connect(self.dsn)
            self._schema_ready = False
        if not self._schema_ready:
            with self._conn.cursor() as cur:
                cur.execute(SCHEMA)
            self._schema_ready = True
        return self._conn

    def _close(self) -> None:
        conn, self._conn = self._conn, None
        self._schema_ready = False
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _query(self, sql: str, params: tuple = ()) -> List[tuple]:
        """
        Run a read on a short-lived connection of its own.

        Reads come from Flask request threads; the writer thread owns
        `self._conn` exclusively, and psycopg connections are not safe to
        share across threads. Admin reads are rare, so a fresh connection is
        the simplest correct answer.
        """
        try:
            conn = self._connect(self.dsn)
        except Exception:
            logger.warning("store read connect failed", exc_info=True)
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(SCHEMA)
                cur.execute(sql, params)
                return list(cur.fetchall())
        except Exception:
            logger.warning("store read failed: %s", sql, exc_info=True)
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass


# Module-level singleton used by the service; tests swap it out.
STORE = Store()

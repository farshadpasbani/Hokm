"""
Tests for the Postgres sink (store.py) — hands and the player registry.

No real database is involved: `Store` takes a `connect` factory, so these
tests inject a fake connection and assert on the SQL that would have run.
"""

import json
import threading

import pytest

import store
from store import BUMP_HANDS, INSERT_HAND, UPSERT_PLAYER, Store


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        if self._conn.fail_on and self._conn.fail_on in sql:
            raise RuntimeError("boom")
        self._conn.executed.append((sql, params))

    def fetchall(self):
        return list(self._conn.rows)


class FakeConn:
    """Records every statement; optionally raises on a chosen SQL fragment."""

    def __init__(self, rows=(), fail_on=None):
        self.executed = []
        self.rows = rows
        self.fail_on = fail_on
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


@pytest.fixture()
def conn():
    return FakeConn()


@pytest.fixture()
def st(conn):
    s = Store("postgres://fake", connect=lambda dsn: conn, touch_interval=300)
    yield s


def _statements(conn):
    return [sql for sql, _ in conn.executed]


def _params_for(conn, needle):
    return [p for sql, p in conn.executed if needle in sql]


class TestDisabledWithoutDatabaseUrl:
    """No DATABASE_URL must mean a total no-op — local dev and tests never
    need a database, and nothing may raise."""

    def test_every_entry_point_is_inert(self):
        s = Store("", connect=lambda dsn: pytest.fail("must not connect"))
        assert s.enabled is False
        assert s.note_player({"user_id": "tg:1"}) is False
        assert s.record_hand({"user": "tg:1"}) is False
        assert s.players() == []
        assert list(s.iter_hands()) == []
        assert s.stats() == {
            "enabled": False, "queued": 0, "written": 0,
            "failed": 0, "retried": 0, "dropped": 0,
        }


class TestHandWrites:
    def test_hand_is_inserted_and_player_counter_bumped(self, st, conn):
        st.record_hand({"user": "tg:77", "trump": "Hearts"})
        assert st.flush()
        assert INSERT_HAND in _statements(conn)
        assert BUMP_HANDS in _statements(conn)
        assert _params_for(conn, "INSERT INTO hands")[0][0] == "tg:77"
        assert _params_for(conn, "INSERT INTO players (user_id, hands_played)")[0] == ("tg:77",)

    def test_record_carries_schema_version_and_timestamp(self, st, conn):
        st.record_hand({"user": "tg:77", "trump": "Hearts"})
        assert st.flush()
        payload = json.loads(_params_for(conn, "INSERT INTO hands")[0][1])
        assert payload["v"] == 1
        assert isinstance(payload["ts"], int)
        assert payload["trump"] == "Hearts"

    def test_schema_is_created_once_not_per_write(self, st, conn):
        for _ in range(3):
            st.record_hand({"user": "tg:1"})
        assert st.flush()
        assert _statements(conn).count(store.SCHEMA) == 1

    def test_missing_user_does_not_lose_the_hand(self, st, conn):
        st.record_hand({"trump": "Spades"})
        assert st.flush()
        assert _params_for(conn, "INSERT INTO hands")[0][0] == "unknown"


class TestPlayerRegistry:
    def test_telegram_handle_is_stored(self, st, conn):
        st.note_player({
            "user_id": "tg:5", "telegram_id": 5, "username": "farshad",
            "first_name": "Farshad", "last_name": "P",
            "language_code": "en", "is_premium": True,
        })
        assert st.flush()
        params = _params_for(conn, "INSERT INTO players (user_id, telegram_id")[0]
        assert params == ("tg:5", 5, "farshad", "Farshad", "P", "en", True)
        assert UPSERT_PLAYER in _statements(conn)

    def test_repeat_launches_are_throttled(self, st, conn):
        assert st.note_player({"user_id": "tg:5"}) is True
        assert st.note_player({"user_id": "tg:5"}) is False
        assert st.note_player({"user_id": "tg:6"}) is True
        assert st.flush()
        assert len(_params_for(conn, "INSERT INTO players (user_id, telegram_id")) == 2

    def test_force_bypasses_the_throttle(self, st):
        assert st.note_player({"user_id": "tg:5"}) is True
        assert st.note_player({"user_id": "tg:5"}, force=True) is True

    def test_throttle_expires(self, conn):
        s = Store("postgres://fake", connect=lambda d: conn, touch_interval=0)
        assert s.note_player({"user_id": "tg:5"}) is True
        assert s.note_player({"user_id": "tg:5"}) is True

    def test_profile_without_user_id_is_ignored(self, st):
        assert st.note_player({"username": "nobody"}) is False


class TestNeverBreaksAGame:
    """The write path is best-effort by contract: a broken database must
    cost training data, never a player's move."""

    def test_connection_killed_while_idle_is_retried_not_lost(self):
        """Serverless Postgres closes idle connections; the failure only shows
        up on the next write. That hand must survive, not vanish."""
        stale, fresh = FakeConn(fail_on="INSERT INTO hands"), FakeConn()
        pending = [stale, fresh]
        s = Store("postgres://fake", connect=lambda d: pending.pop(0))
        s.record_hand({"user": "tg:1"})
        assert s.flush()
        assert s.written == 1 and s.failed == 0 and s.retried == 1
        assert _params_for(fresh, "INSERT INTO hands")[0][0] == "tg:1"
        assert stale.closed is True

    def test_write_failure_is_swallowed_and_counted(self):
        conn = FakeConn(fail_on="INSERT INTO hands")
        s = Store("postgres://fake", connect=lambda d: conn)
        s.record_hand({"user": "tg:1"})
        assert s.flush()
        assert s.failed == 1
        assert conn.closed is True  # dropped so the next write reconnects

    def test_worker_survives_a_failure_and_writes_the_next_record(self):
        """Two failures exhaust the retry and the record is lost — but the
        worker must not die with it."""
        broken = FakeConn(fail_on="INSERT INTO hands")
        healthy = FakeConn()
        pending = [broken, broken, healthy]
        s = Store("postgres://fake", connect=lambda d: pending.pop(0))
        s.record_hand({"user": "tg:1"})
        assert s.flush()
        s.record_hand({"user": "tg:2"})
        assert s.flush()
        assert s.written == 1 and s.failed == 1
        assert _params_for(healthy, "INSERT INTO hands")[0][0] == "tg:2"

    def test_connect_failure_does_not_raise(self):
        def explode(dsn):
            raise RuntimeError("no route to host")

        s = Store("postgres://fake", connect=explode)
        s.record_hand({"user": "tg:1"})
        assert s.flush()
        assert s.failed == 1
        assert s.players() == []          # read path degrades to empty
        assert s.stats()["enabled"] is True

    def test_full_queue_drops_instead_of_blocking(self):
        gate = threading.Event()

        class BlockingConn(FakeConn):
            def cursor(self):
                gate.wait(timeout=5)
                return FakeCursor(self)

        s = Store(
            "postgres://fake", connect=lambda d: BlockingConn(), queue_maxsize=2
        )
        accepted = [s.record_hand({"user": "tg:%d" % i}) for i in range(20)]
        gate.set()
        assert accepted.count(False) > 0     # some were dropped
        assert s.dropped == accepted.count(False)


class TestReads:
    def test_players_maps_rows_to_dicts(self):
        row = (
            "tg:5", 5, "farshad", "Farshad", "P", "en", True,
            "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00", 3, 12,
        )
        s = Store("postgres://fake", connect=lambda d: FakeConn(rows=[row]))
        players = s.players()
        assert players[0]["username"] == "farshad"
        assert players[0]["hands_played"] == 12
        assert players[0]["launches"] == 3

    def test_iter_hands_stops_on_a_short_page(self):
        conn = FakeConn(rows=[({"user": "tg:1"},)])
        s = Store("postgres://fake", connect=lambda d: conn)
        assert list(s.iter_hands(batch=500)) == [{"user": "tg:1"}]

    def test_stats_reports_totals(self):
        s = Store("postgres://fake", connect=lambda d: FakeConn(rows=[(7, 42)]))
        stats = s.stats()
        assert stats["players_total"] == 7
        assert stats["hands_total"] == 42

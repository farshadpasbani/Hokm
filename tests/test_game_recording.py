"""
Tests for finished-hand recording (game_recorder.py + game_service hooks
+ the admin export endpoints).
"""

import json
import os

import pytest

import game_service
import store as store_module
from game_recorder import GameRecorder, load_hands, replay_hand


@pytest.fixture(autouse=True)
def cheap_seats(monkeypatch):
    monkeypatch.setattr(game_service, "AI_KIND", "heuristic")
    monkeypatch.setattr(game_service, "MODEL_PATH", "")


@pytest.fixture()
def recorder(tmp_path, monkeypatch):
    rec = GameRecorder(str(tmp_path / "games"), enabled=True)
    monkeypatch.setattr(game_service, "RECORDER", rec)
    return rec


def _play_full_hand(sess):
    d = sess.new_game()
    if d["phase"] == "choose_trump":
        d = sess.set_trump(d["trump_options"][0])
    for _ in range(30):
        if d.get("game_over"):
            return d
        d = sess.play_card(d["legal_cards"][0])
    raise AssertionError("hand did not finish")


class TestDatabaseSink:
    """A finished hand must reach the Postgres sink as well as the JSONL one —
    on Render that sink is the only one that survives a redeploy."""

    def test_finished_hand_reaches_the_store(self, monkeypatch, recorder):
        from tests.test_store import FakeConn
        from store import Store

        conn = FakeConn()
        monkeypatch.setattr(
            store_module, "STORE", Store("postgres://fake", connect=lambda d: conn)
        )
        sess = game_service.GameSession("guest:db1", "Rec")
        final = _play_full_hand(sess)
        assert store_module.STORE.flush()

        inserts = [p for sql, p in conn.executed if "INSERT INTO hands" in sql]
        assert len(inserts) == 1
        assert inserts[0][0] == "guest:db1"
        record = json.loads(inserts[0][1])
        assert record["v"] == 1
        assert record["scores"]["team1"] == final["scores"]["Team 1"]
        assert len(record["initial_hands"]) == 4
        # The player's hands_played counter is bumped in the same write.
        assert any("hands_played" in sql for sql, _ in conn.executed)

    def test_store_failure_does_not_break_the_game(self, monkeypatch, recorder):
        from store import Store

        def explode(dsn):
            raise RuntimeError("database is asleep")

        monkeypatch.setattr(
            store_module, "STORE", Store("postgres://fake", connect=explode)
        )
        sess = game_service.GameSession("guest:db2", "Rec")
        final = _play_full_hand(sess)          # must complete normally
        assert final["game_over"] is True
        assert len(list(load_hands(recorder.directory))) == 1


class TestRecording:
    def test_hand_recorded_with_complete_facts(self, recorder):
        sess = game_service.GameSession("guest:rec1", "Rec")
        final = _play_full_hand(sess)
        hands = list(load_hands(recorder.directory))
        assert len(hands) == 1
        r = hands[0]
        assert r["v"] == 1
        assert r["user"] == "guest:rec1"
        assert r["hand_index"] == 1
        # Deal integrity: 4 hands x 13 distinct cards = the whole deck.
        cards = [c for h in r["initial_hands"] for c in h]
        assert len(cards) == 52 and len(set(cards)) == 52
        assert all(len(h) == 13 for h in r["initial_hands"])
        # Outcome matches what the session reported.
        assert r["scores"]["team1"] == final["scores"]["Team 1"]
        assert r["scores"]["team2"] == final["scores"]["Team 2"]
        assert r["winner_team"] in (1, 2)
        assert r["trump"] in ("Hearts", "Diamonds", "Clubs", "Spades")
        # Every play references a real card; the hakem leads the first one.
        assert len(r["plays"]) >= 7 * 4 - 3
        assert r["plays"][0][0] == r["hakem_seat"]

    def test_recorded_hand_replays_through_engine(self, recorder):
        sess = game_service.GameSession("guest:rec2", "Rec")
        _play_full_hand(sess)
        r = next(iter(load_hands(recorder.directory)))
        decisions = 0
        game = None
        for game, seat, card in replay_hand(r):
            # At each yield the mover must actually hold the card and the
            # card must be legal — replay doubles as an integrity check.
            mover = game.players[seat]
            assert card in mover.hand
            assert card in game.legal_cards_for_player(mover)
            decisions += 1
        assert decisions == len(r["plays"])
        assert game.scores[1] == r["scores"]["team1"]
        assert game.scores[2] == r["scores"]["team2"]

    def test_second_hand_appends_with_index(self, recorder):
        sess = game_service.GameSession("guest:rec3", "Rec")
        _play_full_hand(sess)
        d = sess.next_hand()
        if d["phase"] == "choose_trump":
            d = sess.set_trump(d["trump_options"][0])
        for _ in range(30):
            if d.get("game_over"):
                break
            d = sess.play_card(d["legal_cards"][0])
        hands = list(load_hands(recorder.directory))
        assert [h["hand_index"] for h in hands] == [1, 2]

    def test_unwritable_directory_never_breaks_play(self, monkeypatch, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.write_text("i am a file, not a directory")
        rec = GameRecorder(str(blocked / "sub"), enabled=True)
        monkeypatch.setattr(game_service, "RECORDER", rec)
        sess = game_service.GameSession("guest:rec4", "Rec")
        final = _play_full_hand(sess)   # must not raise
        assert final.get("game_over") is True

    def test_disabled_recorder_writes_nothing(self, monkeypatch, tmp_path):
        rec = GameRecorder(str(tmp_path / "off"), enabled=False)
        monkeypatch.setattr(game_service, "RECORDER", rec)
        sess = game_service.GameSession("guest:rec5", "Rec")
        _play_full_hand(sess)
        assert rec.files() == []


class TestAdminEndpoints:
    @pytest.fixture()
    def client(self, monkeypatch, recorder):
        import server

        monkeypatch.setattr(server, "ADMIN_TOKEN", "sekret")
        server.app.config["TESTING"] = True
        with server.app.test_client() as c:
            yield c

    def test_export_requires_token(self, client):
        assert client.get("/api/admin/export").status_code == 404
        assert client.get("/api/admin/export?token=wrong").status_code == 404
        assert client.get("/api/admin/stats").status_code == 404

    def test_export_streams_recorded_hands(self, client, recorder):
        sess = game_service.GameSession("guest:rec6", "Rec")
        _play_full_hand(sess)
        resp = client.get("/api/admin/export?token=sekret")
        assert resp.status_code == 200
        lines = [l for l in resp.data.decode().splitlines() if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["user"] == "guest:rec6"
        stats = client.get("/api/admin/stats?token=sekret").get_json()
        assert stats["files"]["hands_recorded"] == 1

    def test_endpoints_disabled_without_token(self, monkeypatch, recorder):
        import server

        monkeypatch.setattr(server, "ADMIN_TOKEN", "")
        server.app.config["TESTING"] = True
        with server.app.test_client() as c:
            assert c.get("/api/admin/export?token=").status_code == 404

"""
Tests for finished-hand recording (game_recorder.py + game_service hooks
+ the admin export endpoints).
"""

import json
import os
import random

import pytest

import game_service
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


def _play_full_hand(sess, d=None):
    if d is None:
        d = sess.new_game()
    if d["phase"] == "choose_trump":
        d = sess.set_trump(d["trump_options"][0])
    for _ in range(30):
        if d.get("game_over"):
            return d
        d = sess.play_card(d["legal_cards"][0])
    raise AssertionError("hand did not finish")


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


def _hand_line(hand_id, index, **extra):
    """The smallest object `load_hands` treats as a hand record."""
    return json.dumps(
        {"v": 1, "ts": index, "hand_id": hand_id, "hand_index": index, **extra}
    )


def _flag_line(hand_id, tricks):
    return json.dumps({"v": 1, "ts": 9, "type": "flag", "hand_id": hand_id,
                       "tricks": tricks})


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCorruptionRecovery:
    """Damaged input must cost only the damaged line, never the rest.

    Recording is append-only from a live game, so a half-written line, a
    stray blank, or a file that cannot be opened at all are all reachable.
    Every skip in `_iter_lines` / `load_hands` is a `continue`; a `break`
    in any of them silently truncates the training set, and a test that
    writes only well-formed files cannot tell the difference.
    """

    @pytest.fixture()
    def damaged(self, tmp_path):
        d = tmp_path / "games"
        d.mkdir()
        # Sorts first, so an OSError that stopped the walk would cost every
        # later file. A directory raises IsADirectoryError from open().
        (d / "hands_00000000.jsonl").mkdir()
        _write_lines(d / "hands_20260101.jsonl", [
            _hand_line("a", 1),
            '{"v": 1, "hand_id": "trunc"',   # half-written line
            "",                              # blank
            "   ",                           # whitespace only
            _hand_line("b", 2),
            "[1, 2, 3]",                     # valid JSON, not a record
            _hand_line("c", 3),
        ])
        _write_lines(d / "hands_20260102.jsonl", [_hand_line("d", 4)])
        return d

    def test_every_undamaged_record_still_loads(self, damaged):
        got = [h["hand_id"] for h in load_hands(str(damaged))]
        assert got == ["a", "b", "c", "d"]

    def test_stats_counts_past_the_damage(self, damaged):
        stats = GameRecorder(str(damaged), enabled=True).stats()
        assert stats["hands_recorded"] == 4

    def test_amendment_flags_union_with_flags_already_on_the_record(
        self, tmp_path
    ):
        """A partial amendment must add to the flags already stored.

        The service writes amendments carrying the hand's complete flag
        set, which makes "union" and "replace" agree; hand-written or
        older files need not, so this uses partial, overlapping sets.
        """
        d = tmp_path / "games"
        d.mkdir()
        _write_lines(d / "hands_20260101.jsonl", [
            _hand_line("a", 1, flagged_tricks=[2, 6]),
            _flag_line("a", [6, 11]),   # overlaps, and adds one
            _flag_line("a", [0]),       # a second amendment folds in too
        ])
        record, = list(load_hands(str(d)))
        assert record["flagged_tricks"] == [0, 2, 6, 11]

    def test_malformed_amendments_are_skipped_not_fatal(self, tmp_path):
        d = tmp_path / "games"
        d.mkdir()
        _write_lines(d / "hands_20260101.jsonl", [
            _hand_line("a", 1),
            json.dumps({"type": "flag"}),                        # no fields
            json.dumps({"type": "flag", "hand_id": 7, "tricks": [5]}),
            json.dumps({"type": "flag", "hand_id": "a", "tricks": "1"}),
            _flag_line("a", [1, 3]),        # the good one, written last
        ])
        record, = list(load_hands(str(d)))
        assert record["flagged_tricks"] == [1, 3]

    def test_amendments_are_never_yielded_alone(self, tmp_path):
        d = tmp_path / "games"
        d.mkdir()
        _write_lines(d / "hands_20260101.jsonl", [
            _flag_line("orphan", [1]),      # no hand record anywhere
            _hand_line("a", 1),
        ])
        got = list(load_hands(str(d)))
        assert [h["hand_id"] for h in got] == ["a"]
        assert "flagged_tricks" not in got[0]

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert list(load_hands(str(tmp_path / "nope"))) == []


class TestTrumpProvenance:
    """`trump_chosen_by_human` labels who fixed trump: the human Hakem
    picking a suit, or the engine's `choose_trump_suit()` heuristic.

    It is a training-data label, so a wrong constant mislabels every hand
    without changing anything a player can see.
    """

    HUMAN_HAKEM_SEED = 6  # this seeded deal makes the human (seat 0) Hakem
    AI_HAKEM_SEED = 1

    def test_human_hakem_is_recorded_as_choosing_trump(self, recorder):
        sess = game_service.GameSession(
            "guest:tc1", "Rec", rng=random.Random(self.HUMAN_HAKEM_SEED)
        )
        d = sess.new_game()
        assert d["phase"] == "choose_trump", "seed must deal the human Hakem"
        chosen = d["trump_options"][0]
        _play_full_hand(sess, sess.set_trump(chosen))

        r = next(iter(load_hands(recorder.directory)))
        assert r["trump_chosen_by_human"] is True
        assert r["trump"] == chosen
        assert r["hakem_seat"] == 0  # the human always sits seat 0

    def test_ai_hakem_is_not_recorded_as_a_human_choice(self, recorder):
        sess = game_service.GameSession(
            "guest:tc2", "Rec", rng=random.Random(self.AI_HAKEM_SEED)
        )
        d = sess.new_game()
        assert d["phase"] == "playing", "seed must deal an AI Hakem"
        _play_full_hand(sess, d)

        r = next(iter(load_hands(recorder.directory)))
        assert r["trump_chosen_by_human"] is False
        assert r["hakem_seat"] != 0


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
        assert stats["hands_recorded"] == 1

    def test_endpoints_disabled_without_token(self, monkeypatch, recorder):
        import server

        monkeypatch.setattr(server, "ADMIN_TOKEN", "")
        server.app.config["TESTING"] = True
        with server.app.test_client() as c:
            assert c.get("/api/admin/export?token=").status_code == 404

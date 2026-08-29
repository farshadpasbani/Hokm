"""
Tests for the "AI is stupid" trick-flag signal.

End to end: POST /api/flag_trick → GameSession → the finished hand's JSONL
record (or an append-only amendment line for flags pressed while the
end-of-hand screen is up) → back out through game_recorder.load_hands.
"""

import json
import random

import pytest

import game_service
import server
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


@pytest.fixture()
def client(monkeypatch):
    """Guest-auth test client — the same ALLOW_GUESTS path the field test uses."""
    monkeypatch.setattr(server, "BOT_TOKEN", "")
    monkeypatch.setattr(server, "ALLOW_GUESTS", True)
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def _post(client, path, body, guest):
    return client.post(path, json=body, headers={"X-Guest-Id": guest})


def _ok(resp):
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def _resolved(data):
    """Tricks completed in the hand so far (both teams' trick counts)."""
    s = data.get("scores") or {}
    return (s.get("Team 1") or 0) + (s.get("Team 2") or 0)


def _start_hand(client, guest):
    d = _ok(_post(client, "/api/new_game", {}, guest))
    if d["phase"] == "choose_trump":
        d = _ok(
            _post(client, "/api/set_trump",
                  {"trump_suit": d["trump_options"][0]}, guest)
        )
    return d


def _play(client, guest, d, tricks=None):
    """Play the first legal card until the hand ends, or for `tricks` plays."""
    for i in range(30):
        if d.get("game_over"):
            return d
        d = _ok(_post(client, "/api/play_card", {"card": d["legal_cards"][0]}, guest))
        if tricks is not None and i + 1 >= tricks:
            return d
    raise AssertionError("hand did not finish")


def test_flag_mid_hand_reaches_the_loaded_training_record(client, recorder):
    guest = "flagtracer"
    d = _play(client, guest, _start_hand(client, guest), tricks=3)
    assert not d.get("game_over")

    displayed = _resolved(d)  # the trick now on the table
    _ok(_post(client, "/api/flag_trick", {"trick_index": displayed}, guest))

    _play(client, guest, d)

    hands = list(load_hands(recorder.directory))
    assert len(hands) == 1
    assert hands[0]["flagged_tricks"] == [displayed]


def test_flag_from_end_screen_is_amended_onto_the_finished_hand(client, recorder):
    """The last trick can only be flagged after the hand — and its record —
    has already ended. That flag must still reach the training data."""
    guest = "flagendscreen"
    d = _play(client, guest, _start_hand(client, guest))
    assert d["game_over"] is True
    last_trick = _resolved(d) - 1

    body = _ok(_post(client, "/api/flag_trick", {"trick_index": last_trick}, guest))
    assert body["already_flagged"] is False
    # Once the hand is over there is no trick on the table, so the index one
    # past the last completed trick must not be accepted.
    assert _post(
        client, "/api/flag_trick", {"trick_index": last_trick + 1}, guest
    ).status_code == 400

    # The hand record was written when the hand ended, so the late flag has to
    # be a separate append-only amendment line.
    lines = [
        json.loads(l)
        for l in open(recorder.files()[0], encoding="utf-8").read().splitlines()
        if l.strip()
    ]
    assert [l.get("type") for l in lines] == [None, "flag"]

    hands = list(load_hands(recorder.directory))
    assert len(hands) == 1  # the amendment is folded in, never yielded alone
    assert hands[0]["flagged_tricks"] == [last_trick]


def test_mid_hand_and_end_screen_flags_both_land_on_one_hand(client, recorder):
    """Mirrors the field test: one flag while playing, one from the end sheet."""
    guest = "flagboth"
    d = _play(client, guest, _start_hand(client, guest), tricks=2)
    mid = _resolved(d)
    _ok(_post(client, "/api/flag_trick", {"trick_index": mid}, guest))

    d = _play(client, guest, d)
    last = _resolved(d) - 1
    _ok(_post(client, "/api/flag_trick", {"trick_index": last}, guest))

    record = next(iter(load_hands(recorder.directory)))
    assert record["flagged_tricks"] == sorted({mid, last})
    # A flagged record is still exactly as replayable as an unflagged one.
    assert sum(1 for _ in replay_hand(record)) == len(record["plays"])


def test_repeated_presses_on_the_same_trick_dedupe(client, recorder):
    guest = "flagdupe"
    d = _play(client, guest, _start_hand(client, guest), tricks=2)
    mid = _resolved(d)
    first = _ok(_post(client, "/api/flag_trick", {"trick_index": mid}, guest))
    again = _ok(_post(client, "/api/flag_trick", {"trick_index": mid}, guest))
    assert first["already_flagged"] is False
    assert again["already_flagged"] is True
    assert again["flagged_tricks"] == [mid]

    d = _play(client, guest, d)
    # Duplicate presses after the record is on disk must not amend it either.
    _ok(_post(client, "/api/flag_trick", {"trick_index": mid}, guest))
    _ok(_post(client, "/api/flag_trick", {"trick_index": mid}, guest))
    assert next(iter(load_hands(recorder.directory)))["flagged_tricks"] == [mid]


@pytest.mark.parametrize("payload", [
    {},                              # missing field
    {"trick_index": None},
    {"trick_index": "nonsense"},
    {"trick_index": -1},
    {"trick_index": 99},             # beyond any hand
])
def test_bad_trick_index_is_rejected(client, recorder, payload):
    guest = "flagbad"
    _play(client, guest, _start_hand(client, guest), tricks=2)
    assert _post(client, "/api/flag_trick", payload, guest).status_code == 400


def test_trick_still_being_played_is_flaggable_but_the_next_one_is_not(client, recorder):
    """Pins the upper bound: the trick on the table counts, the unplayed
    trick after it does not."""
    guest = "flagbound"
    d = _play(client, guest, _start_hand(client, guest), tricks=2)
    on_table = _resolved(d)
    assert _post(
        client, "/api/flag_trick", {"trick_index": on_table}, guest
    ).status_code == 200
    assert _post(
        client, "/api/flag_trick", {"trick_index": on_table + 1}, guest
    ).status_code == 400


def test_flag_without_a_hand_in_progress_is_rejected():
    """No game at all, and a dealt hand whose trump is not chosen yet."""
    sess = game_service.GameSession("guest:noflag", "Tester")
    with pytest.raises(game_service.GameServiceError):
        sess.flag_trick(0)

    # Seeded so the deal reliably makes the human Hakem (trump picker up,
    # no trick on the table yet) rather than relying on a 1-in-4 shuffle.
    for seed in range(60):
        random.seed(seed)
        sess = game_service.GameSession("guest:noflag", "Tester")
        if sess.new_game()["phase"] == "choose_trump":
            break
    else:
        raise AssertionError("no seed produced a human-Hakem deal")
    with pytest.raises(game_service.GameServiceError):
        sess.flag_trick(0)


def test_next_hand_closes_the_amendment_window(client, recorder):
    """Once the next hand is dealt, a flag belongs to that hand — it must not
    be back-attached to the hand already on disk."""
    guest = "flagwindow"
    d = _play(client, guest, _start_hand(client, guest), tricks=2)
    first_mid = _resolved(d)
    _ok(_post(client, "/api/flag_trick", {"trick_index": first_mid}, guest))
    _play(client, guest, d)

    d = _ok(_post(client, "/api/next_hand", {}, guest))
    if d["phase"] == "choose_trump":
        # No trick exists yet in the new hand, so nothing can be flagged.
        assert _post(
            client, "/api/flag_trick", {"trick_index": 0}, guest
        ).status_code == 400
        d = _ok(_post(client, "/api/set_trump",
                      {"trump_suit": d["trump_options"][0]}, guest))
    _ok(_post(client, "/api/flag_trick", {"trick_index": 0}, guest))
    _play(client, guest, d)

    hands = list(load_hands(recorder.directory))
    assert [h["hand_index"] for h in hands] == [1, 2]
    assert hands[0]["flagged_tricks"] == [first_mid]  # untouched by hand 2
    assert hands[1]["flagged_tricks"] == [0]
    assert hands[0]["hand_id"] != hands[1]["hand_id"]


def test_orphan_and_corrupt_amendments_are_skipped(client, recorder):
    guest = "flagorphan"
    d = _play(client, guest, _start_hand(client, guest), tricks=2)
    mid = _resolved(d)
    _ok(_post(client, "/api/flag_trick", {"trick_index": mid}, guest))
    _play(client, guest, d)

    with open(recorder.files()[0], "a", encoding="utf-8") as f:
        f.write('{"v":1,"type":"flag","hand_id":"nosuchhand","tricks":[3]}\n')
        f.write('{"v":1,"type":"flag","hand_id":42,"tricks":"nope"}\n')
        f.write("{ this is not json\n")
        f.write("\n")

    hands = list(load_hands(recorder.directory))
    assert len(hands) == 1
    assert hands[0]["flagged_tricks"] == [mid]
    # Amendments are not hands: the admin stats count must not drift.
    assert recorder.stats()["hands_recorded"] == 1


def test_flagging_survives_a_broken_recorder(monkeypatch, tmp_path, client):
    """Recording failures must never break play — nor flagging."""
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    rec = GameRecorder(str(blocked / "sub"), enabled=True)
    monkeypatch.setattr(game_service, "RECORDER", rec)

    guest = "flagbroken"
    d = _play(client, guest, _start_hand(client, guest), tricks=2)
    _ok(_post(client, "/api/flag_trick", {"trick_index": _resolved(d)}, guest))
    d = _play(client, guest, d)
    assert d["game_over"] is True
    _ok(_post(client, "/api/flag_trick", {"trick_index": _resolved(d) - 1}, guest))

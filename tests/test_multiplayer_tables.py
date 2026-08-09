"""
Multi-human Hokm tables: `table_service.py` plus the `/api/table/*` surface.

A table seats 2-4 humans; AI fills every remaining seat. Seat ownership and
turn order are enforced server-side, each player sees only their own hand,
and an idle seat is covered by AI so the table never stalls.

Seats map to the engine's fixed order (south, east, north, west): seats 0
and 2 are Team 1, seats 1 and 3 are Team 2.

`AI_KIND=heuristic` keeps the AI seats cheap: these tests play out whole
hands and must stay fast next to a training run.
"""

import json
import os
import threading
import time

import pytest

# `game_service` reads AI_KIND once, at import; set it before importing in
# case this module is imported first. The autouse fixture below is what
# actually guarantees it, since another test module may import first.
os.environ.setdefault("AI_KIND", "heuristic")

import game_service  # noqa: E402
import table_service  # noqa: E402
from game_recorder import GameRecorder, load_hands  # noqa: E402
from game_service import GameServiceError  # noqa: E402
from table_service import TableStore  # noqa: E402
from telegram_auth import sign_init_data  # noqa: E402

TWO = (("u0", "Ann"), ("u1", "Ben"))


ALICE = {"X-Guest-Id": "alice1", "X-Guest-Name": "Alice"}
BOB = {"X-Guest-Id": "bob2", "X-Guest-Name": "Bob"}

# Invites need a verified Telegram identity: a guest has no @handle, and the
# handle → user-id directory is only ever filled from signed initData.
TEST_BOT_TOKEN = "123456:TESTTOKEN"


def _tma(uid, first_name, username=None):
    """Headers carrying HMAC-signed initData for one Telegram user."""
    user = {"id": uid, "first_name": first_name}
    if username:
        user["username"] = username
    signed = sign_init_data(
        {"auth_date": str(int(time.time())), "user": json.dumps(user)},
        TEST_BOT_TOKEN,
    )
    return {"Authorization": "tma " + signed}


@pytest.fixture(autouse=True)
def _cheap_seats(monkeypatch):
    """Rule-based AI seats regardless of the ambient env."""
    monkeypatch.setenv("AI_KIND", "heuristic")
    monkeypatch.setattr(game_service, "AI_KIND", "heuristic")
    monkeypatch.setattr(game_service, "MODEL_PATH", "")


@pytest.fixture()
def client(monkeypatch):
    """Guest-auth Flask client on a fresh, empty table store."""
    import server

    monkeypatch.setattr(server, "BOT_TOKEN", "")
    monkeypatch.setattr(server, "ALLOW_GUESTS", True)
    monkeypatch.setattr(server, "tables", table_service.TableStore())
    monkeypatch.setenv("BOT_USERNAME", "hokm_test_bot")
    monkeypatch.setenv("MINI_APP_SHORT_NAME", "play")
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


@pytest.fixture()
def tg_client(monkeypatch):
    """
    Telegram-authenticated client with every Bot API call captured.

    The bot call is faked, not reached: delivery must be reported from what
    the Bot API actually answered, and a test has no live bot to answer it.
    """
    import server

    monkeypatch.setattr(server, "BOT_TOKEN", TEST_BOT_TOKEN)
    monkeypatch.setattr(server, "ALLOW_GUESTS", False)
    monkeypatch.setattr(server, "WEBAPP_URL", "https://hokm.example.com")
    monkeypatch.setattr(server, "WEBHOOK_SECRET", "s3cret")
    monkeypatch.setattr(server, "tables", table_service.TableStore())
    monkeypatch.setattr(server, "directory", table_service.UserDirectory())
    monkeypatch.setenv("BOT_USERNAME", "hokm_test_bot")
    # No Mini App short name: the deep link is then the plain-bot `?start=`
    # shape, which is the one that goes through the webhook.
    monkeypatch.delenv("MINI_APP_SHORT_NAME", raising=False)
    sent = []
    monkeypatch.setattr(
        server,
        "_bot_api",
        lambda method, payload: (sent.append((method, payload)), {"ok": True})[1],
    )
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        c.sent = sent
        yield c


def _post(client, path, who, body=None):
    resp = client.post(path, json=body or {}, headers=who)
    return resp.status_code, resp.get_json()


def _get(client, path, who):
    resp = client.get(path, headers=who)
    return resp.status_code, resp.get_json()


def _settle_trump(client, seated):
    """Choose trump if the Hakem is one of `seated`; AI Hakems already have."""
    for who in seated:
        status, view = _get(client, "/api/table/state", who)
        assert status == 200, view
        if view["phase"] != "choose_trump":
            return
        if view["trump_options"]:
            status, view = _post(
                client,
                "/api/table/set_trump",
                who,
                {"trump_suit": view["trump_options"][0]},
            )
            assert status == 200, view
            return
    raise AssertionError("choose_trump phase but no seated player is Hakem")


def _started_table(humans=TWO):
    """A dealt table: `humans[0]` hosts at seat 0, the rest take free seats."""
    store = TableStore()
    (host_id, host_name), *rest = humans
    table = store.create(host_id, host_name)
    for user_id, name in rest:
        store.join(user_id, name, table.code)
    table.start(host_id)
    return store, table


def _one_action(table, users):
    """Perform the single legal action available to a seated human, if any."""
    for user_id in users:
        view = table.view(user_id)
        if view["phase"] == "choose_trump" and view["trump_options"]:
            table.set_trump(user_id, view["trump_options"][0])
            return True
        if view.get("your_turn"):
            table.play_card(user_id, view["legal_cards"][0])
            return True
    return False


def _settle_table_trump(table, users):
    """Make sure a trump is on the table: a human Hakem picks its first suit."""
    view = table.view(users[0])
    if view["phase"] != "choose_trump":
        return  # an AI Hakem has already chosen
    hakem = next(u for u in users if table.view(u)["trump_options"])
    table.set_trump(hakem, table.view(hakem)["trump_options"][0])


def _play_hand(table, users):
    """Play the current hand out; humans always take `legal_cards[0]`."""
    for _ in range(120):
        view = table.view(users[0])
        if view.get("game_over"):
            return view
        if not _one_action(table, users):
            raise AssertionError(f"table stalled in phase {view['phase']}")
    raise AssertionError("hand did not finish within bound")


def _to_move(client, seated):
    """(headers, view) of the seated player whose turn it is, else (None, ...)."""
    for who in seated:
        status, view = _get(client, "/api/table/state", who)
        assert status == 200, view
        if view.get("your_turn"):
            return who, view
    return None, view


class TestTracer:
    """End-to-end: auth -> table store -> seat ownership -> engine -> view."""

    def test_two_humans_play_one_shared_table(self, client):
        status, created = _post(client, "/api/table/create", ALICE)
        assert status == 200, created
        code = created["code"]
        assert len(code) == 6
        assert created["your_seat"] == 0
        assert created["join_url"] == (
            f"https://t.me/hokm_test_bot/play?startapp={code}"
        )

        status, joined = _post(client, "/api/table/join", BOB, {"code": code})
        assert status == 200, joined
        assert joined["your_seat"] == 1

        status, started = _post(client, "/api/table/start", ALICE)
        assert status == 200, started

        _settle_trump(client, [ALICE, BOB])

        _, view_a = _get(client, "/api/table/state", ALICE)
        _, view_b = _get(client, "/api/table/state", BOB)

        # Same shared table for both players...
        assert view_a["table"]["code"] == view_b["table"]["code"] == code
        assert view_a["seats"] == view_b["seats"]
        assert view_a["trump_suit"] == view_b["trump_suit"]
        assert [p["kind"] for p in view_a["table"]["players"]] == [
            "human",
            "human",
            "ai",
            "ai",
        ]
        # ...but each sees only their own cards.
        assert len(view_a["hand"]) in (12, 13)
        assert len(view_b["hand"]) in (12, 13)
        assert set(view_a["hand"]).isdisjoint(view_b["hand"])

        # The player who is NOT to move is rejected: wrong turn.
        mover, view = _to_move(client, [ALICE, BOB])
        assert mover is not None, "neither human is to move after the deal"
        waiting = BOB if mover is ALICE else ALICE
        _, waiting_view = _get(client, "/api/table/state", waiting)
        status, refused = _post(
            client,
            "/api/table/play_card",
            waiting,
            {"card": waiting_view["hand"][0]},
        )
        assert status == 400
        assert refused["status"] == "error"
        assert "turn" in refused["message"].lower()

        # The player whose turn it is plays a legal card.
        before = view["table"]["version"]
        status, played = _post(
            client, "/api/table/play_card", mover, {"card": view["legal_cards"][0]}
        )
        assert status == 200, played
        assert played["table"]["version"] > before

        # Both observers see the same advanced shared state.
        _, view_a = _get(client, "/api/table/state", ALICE)
        _, view_b = _get(client, "/api/table/state", BOB)
        assert view_a["current_trick"] == view_b["current_trick"]
        assert view_a["scores"] == view_b["scores"]

        # Drive on until a trick resolves (the AI seats fill in behind us).
        for _ in range(12):
            _, view_a = _get(client, "/api/table/state", ALICE)
            if sum(view_a["scores"].values()) >= 1:
                break
            mover, view = _to_move(client, [ALICE, BOB])
            assert mover is not None, "table stalled with no human to move"
            status, played = _post(
                client,
                "/api/table/play_card",
                mover,
                {"card": view["legal_cards"][0]},
            )
            assert status == 200, played
        else:
            raise AssertionError("no trick resolved within bound")

        _, view_b = _get(client, "/api/table/state", BOB)
        assert view_a["scores"] == view_b["scores"]
        assert sum(view_b["scores"].values()) >= 1


class TestInviteTracer:
    """End-to-end: initData -> handle directory -> invite -> DM -> deep link
    -> webhook -> join on a chosen seat -> shared table."""

    def test_an_invited_handle_ends_up_seated_at_the_table(self, tg_client):
        alice = _tma(101, "Alice", "alice_h")
        ben = _tma(202, "Ben", "ben_h")

        # Ben has opened the app before, so the server knows his handle.
        assert tg_client.get("/api/state", headers=ben).status_code == 200

        created = tg_client.post(
            "/api/table/create", json={}, headers=alice
        ).get_json()
        code = created["code"]
        assert code in created["join_url"]

        invited = tg_client.post(
            "/api/table/invite", json={"handle": "@ben_h"}, headers=alice
        )
        assert invited.status_code == 200, invited.get_json()
        body = invited.get_json()
        assert body["known"] is True and body["delivered"] is True
        method, payload = tg_client.sent[-1]
        assert method == "sendMessage" and payload["chat_id"] == 202
        dm_button = payload["reply_markup"]["inline_keyboard"][0][0]
        assert code in dm_button["web_app"]["url"]

        # Ben taps the share link instead, which sends "/start <code>".
        hook = tg_client.post(
            "/telegram/webhook/s3cret",
            json={"message": {"chat": {"id": 202}, "text": f"/start {code}"}},
        )
        assert hook.status_code == 200
        method, payload = tg_client.sent[-1]
        start_button = payload["reply_markup"]["inline_keyboard"][0][0]
        assert code in start_button["web_app"]["url"]

        # The Mini App opens on that code and seats him — on Alice's team.
        joined = tg_client.post(
            "/api/table/join", json={"code": code, "seat": 2}, headers=ben
        ).get_json()
        assert joined["your_seat"] == 2

        assert tg_client.post(
            "/api/table/start", json={}, headers=alice
        ).status_code == 200
        view = tg_client.get("/api/table/state", headers=ben).get_json()
        assert [p["kind"] for p in view["table"]["players"]] == [
            "human",
            "ai",
            "human",
            "ai",
        ]


class TestSeating:
    """Who may sit where, and what the table looks like once it starts."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_humans_take_seats_and_ai_fills_the_rest(self, n):
        _, table = _started_table([(f"u{i}", f"P{i}") for i in range(n)])
        kinds = [p["kind"] for p in table.view("u0")["table"]["players"]]
        assert kinds == ["human"] * n + ["ai"] * (4 - n)
        assert len(table.session.human_seats) == n

    def test_seat_choice_clashes_and_capacity(self):
        store = TableStore()
        table = store.create("u0", "Ann")
        assert store.join("u2", "Ben", table.code, seat=2).seat_of("u2") == 2
        with pytest.raises(GameServiceError, match="taken"):
            store.join("u9", "Cy", table.code, seat=2)
        with pytest.raises(GameServiceError, match="Seat must be"):
            store.join("u9", "Cy", table.code, seat=9)
        store.join("u1", "Ann", table.code)  # lowest free seat, clashing name
        store.join("u3", "Di", table.code)
        with pytest.raises(GameServiceError, match="full"):
            store.join("u9", "Ed", table.code)
        # Distinct seat names: two "Ann"s at one table would read as one player.
        assert [s.name for s in table.seats] == ["Ann", "Ann 2", "Ben", "Di"]
        table.start("u0")
        assert table.session.human_seats == {0: "Ann", 1: "Ann 2", 2: "Ben", 3: "Di"}

    def test_start_needs_two_humans_and_the_host(self):
        store = TableStore()
        table = store.create("u0", "Ann")
        with pytest.raises(GameServiceError, match="at least 2"):
            table.start("u0")
        store.join("u1", "Ben", table.code)
        with pytest.raises(GameServiceError, match="created the table"):
            table.start("u1")
        table.start("u0")
        with pytest.raises(GameServiceError, match="already started"):
            table.start("u0")
        with pytest.raises(GameServiceError, match="already started"):
            store.join("u2", "Cy", table.code)

    def test_actions_before_the_table_starts_are_refused(self):
        store = TableStore()
        table = store.create("u0", "Ann")
        store.join("u1", "Ben", table.code)
        assert table.view("u0")["phase"] == "lobby"
        for call in (
            lambda: table.play_card("u0", "Ace of Spades"),
            lambda: table.set_trump("u0", "Hearts"),
            lambda: table.next_hand("u0"),
        ):
            with pytest.raises(GameServiceError, match="not started"):
                call()

    def test_leaving_frees_a_seat_before_the_deal_and_is_cover_after(self):
        store = TableStore()
        table = store.create("u0", "Ann")
        store.join("u1", "Ben", table.code)
        store.leave("u1")
        assert table.seats[1] is None
        store.join("u1", "Ben", table.code)
        table.start("u0")
        store.leave("u1")
        # Mid-hand the seat cannot open up — the AI takes it over instead.
        assert table.seats[1] is not None
        assert 1 in table.session.auto_seats

    def test_unknown_and_empty_join_codes_are_refused(self):
        store = TableStore()
        for code in ("", "   ", "ZZZZZZ", None):
            with pytest.raises(GameServiceError, match="No table with that code"):
                store.join("u9", "Zed", code)

    def test_a_stranger_can_neither_look_nor_act(self):
        store, table = _started_table()
        with pytest.raises(GameServiceError, match="not seated"):
            table.view("u404")
        with pytest.raises(GameServiceError, match="not at a table"):
            store.for_user("u404")


class TestTurnAndSeatEnforcement:
    def test_a_seat_you_do_not_own_is_refused(self):
        _, table = _started_table()
        sess = table.session
        with pytest.raises(GameServiceError, match="not yours"):
            sess.play_card("Ace of Spades", seat=2)  # an AI seat
        with pytest.raises(GameServiceError, match="not yours"):
            sess.set_trump("Hearts", seat=3)

    def test_playing_out_of_turn_is_refused(self):
        _, table = _started_table()
        _settle_table_trump(table, ("u0", "u1"))
        waiting = "u1" if table.view("u0")["your_turn"] else "u0"
        with pytest.raises(GameServiceError, match="Not your turn"):
            table.play_card(waiting, table.view(waiting)["hand"][0])

    def test_only_the_hakem_chooses_trump(self):
        for _ in range(40):
            _, table = _started_table()
            sess = table.session
            if sess.game.trump_suit:
                continue  # AI Hakem; it has already chosen
            other = 1 - sess._hakem_seat()
            with pytest.raises(GameServiceError, match="Only the Hakem"):
                sess.set_trump("Hearts", other)
            return
        raise AssertionError("no deal in 40 gave a human Hakem")


class TestPrivateHands:
    def test_no_view_ever_carries_another_seats_unplayed_cards(self):
        _, table = _started_table()
        game = table.session.game
        for _ in range(120):
            views = {u: table.view(u) for u in ("u0", "u1")}
            for user_id, seat in (("u0", 0), ("u1", 1)):
                theirs = {str(c) for c in game.players[1 - seat].hand}
                body = json.dumps(views[user_id])
                assert not [c for c in theirs if c in body], (
                    f"{user_id} was shown seat {1 - seat}'s cards"
                )
            if views["u0"].get("game_over"):
                return
            assert _one_action(table, ("u0", "u1"))
        raise AssertionError("hand did not finish within bound")


class TestTrumpAndMatchRules:
    def test_trump_is_chosen_by_the_hakem_human_or_ai(self):
        seen = set()
        for _ in range(40):
            _, table = _started_table()
            view = table.view("u0")
            if view["phase"] == "playing":
                seen.add("ai")
                assert view["trump_suit"] in game_service.suits
            else:
                seen.add("human")
                hakem = "u0" if view["trump_options"] else "u1"
                options = table.view(hakem)["trump_options"]
                assert options, "the Hakem's seat must be offered its suits"
                assert table.set_trump(hakem, options[0])["trump_suit"] == options[0]
            if seen == {"ai", "human"}:
                return
        raise AssertionError(f"only ever saw a {seen} Hakem in 40 deals")

    def test_match_target_and_kot_match_solo(self, monkeypatch):
        monkeypatch.setenv("MATCH_TARGET", "2")
        _, table = _started_table()
        _settle_table_trump(table, ("u0", "u1"))
        sess = table.session
        assert sess.match_target == 2
        game = sess.game
        game.scores[1], game.scores[2] = 7, 0  # a Kot for Team 1 (seats 0 and 2)
        for p in game.players:
            p.hand = []
            game.tricks_won[p] = 0
        game.tricks_won[game.team1[0]] = 7
        sess.hand_result = None
        sess._finish_hand_if_over()
        assert sess.match_score == {"Team 1": 2, "Team 2": 0}  # Kot counts double
        assert sess.match_over is True and sess.match_winner == "Team 1"
        # `you_won` follows the asking seat's team, not a fixed side.
        assert table.view("u0")["you_won"] is True
        assert table.view("u1")["you_won"] is False

    def test_a_finished_table_hand_is_recorded(self, tmp_path, monkeypatch):
        recorder = GameRecorder(str(tmp_path / "games"), enabled=True)
        monkeypatch.setattr(game_service, "RECORDER", recorder)
        _, table = _started_table()
        _play_hand(table, ("u0", "u1"))
        hands = list(load_hands(recorder.directory))
        assert len(hands) == 1
        assert hands[0]["user"] == f"table:{table.code}"
        assert hands[0]["human_seats"] == {"0": "Ann", "1": "Ben"}
        assert hands[0]["winner_team"] in (1, 2)
        assert len(hands[0]["plays"]) >= 7 * 4 - 3

    def test_a_table_plays_a_second_hand(self):
        _, table = _started_table()
        first = _play_hand(table, ("u0", "u1"))
        assert sum(first["match_score"].values()) in (1, 2)
        nxt = table.next_hand("u0")
        assert nxt["scores"] == {"Team 1": 0, "Team 2": 0}
        assert nxt["phase"] in ("choose_trump", "playing")


class TestIdleCover:
    def test_ai_plays_an_idle_seat_and_the_human_reclaims_it(self, monkeypatch):
        monkeypatch.setattr(table_service, "TABLE_IDLE_SECONDS", 30.0)
        _, table = _started_table()
        table.seats[1].last_seen = time.time() - 120  # Ben put his phone down
        for _ in range(120):
            view = table.view("u0")
            assert 1 in table.session.auto_seats
            if view.get("game_over"):
                break
            assert _one_action(table, ("u0",)), "table stalled on the idle seat"
        else:
            raise AssertionError("the covered table never finished the hand")
        assert view["game_over"] is True
        assert table.view("u0")["table"]["players"][1]["idle"] is True
        back = table.view("u1")  # Ben comes back
        assert 1 not in table.session.auto_seats
        assert back["table"]["players"][1]["idle"] is False

    def test_ai_chooses_trump_for_an_idle_hakem(self, monkeypatch):
        monkeypatch.setattr(table_service, "TABLE_IDLE_SECONDS", 30.0)
        for _ in range(40):
            _, table = _started_table()
            view = table.view("u0")
            if view["phase"] != "choose_trump":
                continue  # AI Hakem: nothing to go idle on
            hakem_seat = 0 if view["trump_options"] else 1
            table.seats[hakem_seat].last_seen = time.time() - 120
            after = table.view(f"u{1 - hakem_seat}")
            assert after["phase"] == "playing"
            assert after["trump_suit"] in game_service.suits
            return
        raise AssertionError("no deal in 40 gave a human Hakem")


class TestVersionAndPolling:
    def test_version_moves_only_when_the_table_does(self):
        _, table = _started_table()
        settled = table.view("u0")["table"]["version"]
        assert table.view("u0")["table"]["version"] == settled
        assert table.view("u1")["table"]["version"] == settled
        assert _one_action(table, ("u0", "u1"))
        assert table.view("u0")["table"]["version"] > settled

    def test_waiting_wakes_on_another_players_move(self):
        _, table = _started_table()
        settled = table.view("u0")["table"]["version"]
        threading.Timer(0.05, _one_action, (table, ("u0", "u1"))).start()
        began = time.time()
        table.wait_for_change(settled, timeout=5.0)
        assert table.version > settled
        assert time.time() - began < 4.0


class TestConcurrency:
    def test_parallel_requests_do_not_corrupt_one_table(self, monkeypatch):
        """Two players on eight gunicorn threads share one table object."""
        monkeypatch.setattr(table_service, "TABLE_IDLE_SECONDS", 0.0)
        _, table = _started_table()
        errors = []

        def hammer(user_id):
            try:
                for _ in range(60):
                    table.view(user_id)
            except Exception as exc:  # noqa: BLE001 — the assertion is "none"
                errors.append(exc)

        threads = [
            threading.Thread(target=hammer, args=(u,))
            for u in ("u0", "u1", "u0", "u1", "u0", "u1")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert errors == []
        game = table.session.game
        dealt = sum(len(p.hand) for p in game.players)
        resolved = 4 * (game.scores[1] + game.scores[2])
        assert dealt + len(game.current_trick) + resolved == 52

    def test_racing_joiners_never_share_a_seat(self):
        """Four friends tap Join at once; nobody may land on the same seat."""
        store = TableStore()
        table = store.create("u0", "Ann")
        seated, refused = {}, []

        def race(user_id):
            try:
                seated[user_id] = store.join(user_id, user_id, table.code).seat_of(
                    user_id
                )
            except GameServiceError as exc:
                refused.append(str(exc))

        threads = [threading.Thread(target=race, args=(f"r{i}",)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(seated) == 3 and sorted(seated.values()) == [1, 2, 3]
        assert len(refused) == 3 and all("full" in m for m in refused)


class TestStoreLifecycle:
    def test_capacity_cap_and_ttl_eviction(self, monkeypatch):
        monkeypatch.setattr(table_service, "MAX_TABLES", 1)
        store = TableStore()
        store.create("u0", "Ann")
        with pytest.raises(GameServiceError, match="Too many tables"):
            store.create("u9", "Zed")
        monkeypatch.setattr(table_service, "TABLE_TTL_SECONDS", 0)
        store.create("u9", "Zed")  # the stale table is evicted to make room
        assert store.count() == 1
        with pytest.raises(GameServiceError, match="not at a table"):
            store.for_user("u0")

    def test_creating_a_second_table_gives_up_the_first_seat(self):
        store = TableStore()
        first = store.create("u0", "Ann")
        store.join("u1", "Ben", first.code)
        store.create("u1", "Ben")
        assert first.seats[1] is None
        assert store.for_user("u1") is not first


class TestDeepLink:
    def test_join_url_shapes(self, monkeypatch):
        monkeypatch.delenv("BOT_USERNAME", raising=False)
        monkeypatch.delenv("MINI_APP_SHORT_NAME", raising=False)
        assert table_service.deep_link("ABC234") == ""
        monkeypatch.setenv("BOT_USERNAME", "@hokm_bot")
        assert table_service.deep_link("ABC234") == "https://t.me/hokm_bot?start=ABC234"
        monkeypatch.setenv("MINI_APP_SHORT_NAME", "play")
        assert (
            table_service.deep_link("ABC234")
            == "https://t.me/hokm_bot/play?startapp=ABC234"
        )


class TestUserDirectory:
    """The handle → user-id map that makes an invite by @handle possible."""

    def test_a_handle_is_learned_from_initdata_and_matched_loosely(self, tg_client):
        import server

        assert server.directory.lookup("zara") is None
        tg_client.get("/api/state", headers=_tma(303, "Zara", "Zara_H"))
        # Telegram handles are case-insensitive and get typed with or without
        # the "@", so every spelling of one handle must find the same person.
        for typed in ("zara_h", "Zara_H", "@ZARA_H", "  @zara_h  "):
            assert server.directory.lookup(typed) == "tg:303"

    def test_a_guest_never_enters_the_directory(self, client, monkeypatch):
        import server

        directory = table_service.UserDirectory()
        monkeypatch.setattr(server, "directory", directory)
        _post(client, "/api/table/create", ALICE)
        assert directory.count() == 0

    def test_entries_expire_and_the_cap_drops_the_stalest(self, monkeypatch):
        directory = table_service.UserDirectory()
        directory.remember("@Ann", "tg:1")
        assert directory.lookup("ann") == "tg:1"
        monkeypatch.setattr(table_service, "DIRECTORY_TTL_SECONDS", 0.0)
        assert directory.lookup("ann") is None

        monkeypatch.setattr(table_service, "DIRECTORY_TTL_SECONDS", 3600.0)
        monkeypatch.setattr(table_service, "MAX_DIRECTORY_ENTRIES", 2)
        for i, handle in enumerate(("a", "b", "c")):
            directory.remember(handle, f"tg:{i}")
        assert directory.count() == 2
        assert directory.lookup("a") is None and directory.lookup("c") == "tg:2"

    def test_junk_handles_are_ignored_rather_than_stored(self):
        directory = table_service.UserDirectory()
        for junk in ("", "   ", "@", None):
            directory.remember(junk, "tg:1")
            assert directory.lookup(junk) is None
        assert directory.count() == 0


class TestInvite:
    """An invite must never fail silently, and never overclaim delivery."""

    def _table(self, tg_client, host):
        return tg_client.post(
            "/api/table/create", json={}, headers=host
        ).get_json()["code"]

    def test_an_unknown_handle_returns_the_share_link_and_says_so(self, tg_client):
        code = self._table(tg_client, _tma(101, "Alice", "alice_h"))
        body = tg_client.post(
            "/api/table/invite",
            json={"handle": "@never_seen"},
            headers=_tma(101, "Alice", "alice_h"),
        ).get_json()
        assert body["known"] is False and body["delivered"] is False
        assert code in body["join_url"]
        assert "has not opened Hokm" in body["message"]
        assert tg_client.sent == []  # nothing was sent, nothing was claimed

    @pytest.mark.parametrize("answer", [None, {"ok": False, "error_code": 403}])
    def test_a_failed_dm_is_reported_as_not_delivered(
        self, tg_client, monkeypatch, answer
    ):
        """No BOT_TOKEN (None) and "the invitee never started the bot" (ok
        false) must both read as *not sent*, with the link as the fallback."""
        import server

        alice = _tma(101, "Alice", "alice_h")
        tg_client.get("/api/state", headers=_tma(202, "Ben", "ben_h"))
        code = self._table(tg_client, alice)
        monkeypatch.setattr(server, "_bot_api", lambda method, payload: answer)
        body = tg_client.post(
            "/api/table/invite", json={"handle": "ben_h"}, headers=alice
        ).get_json()
        assert body["known"] is True and body["delivered"] is False
        assert "Could not message" in body["message"]
        assert code in body["join_url"]

    def test_with_no_bot_username_the_fallback_is_the_code_not_a_blank_link(
        self, tg_client, monkeypatch
    ):
        """`deep_link` returns "" when BOT_USERNAME is unset, so telling the
        inviter to "send them this link" would be telling them to send
        nothing."""
        monkeypatch.delenv("BOT_USERNAME", raising=False)
        alice = _tma(101, "Alice", "alice_h")
        code = self._table(tg_client, alice)
        body = tg_client.post(
            "/api/table/invite", json={"handle": "@never_seen"}, headers=alice
        ).get_json()
        assert body["join_url"] == ""
        assert f"give them the code {code}" in body["message"]
        assert "this link" not in body["message"]

    def test_an_empty_handle_and_a_tableless_inviter_are_refused(self, tg_client):
        alice = _tma(101, "Alice", "alice_h")
        refused = tg_client.post(
            "/api/table/invite", json={"handle": "@x"}, headers=alice
        )
        assert refused.status_code == 400
        assert "not at a table" in refused.get_json()["message"]
        self._table(tg_client, alice)
        for body in ({}, {"handle": ""}, {"handle": "  @  "}):
            answer = tg_client.post("/api/table/invite", json=body, headers=alice)
            assert answer.status_code == 400
            assert "@handle" in answer.get_json()["message"]


class TestDeepLinkEntry:
    """Opening a share link must land the tapper in that table, once."""

    def test_start_with_a_code_offers_a_button_onto_that_table(self, tg_client):
        code = tg_client.post(
            "/api/table/create", json={}, headers=_tma(101, "Alice", "alice_h")
        ).get_json()["code"]
        tg_client.post(
            "/telegram/webhook/s3cret",
            json={"message": {"chat": {"id": 202}, "text": f"/start {code}"}},
        )
        _, payload = tg_client.sent[-1]
        button = payload["reply_markup"]["inline_keyboard"][0][0]
        assert button["web_app"]["url"] == f"https://hokm.example.com?table={code}"
        assert code in payload["text"]

    @pytest.mark.parametrize("text", ["/start", "/start not-a-code", "/start ZZZZZZ1"])
    def test_start_without_a_usable_code_still_answers_the_plain_welcome(
        self, tg_client, text
    ):
        tg_client.post(
            "/telegram/webhook/s3cret",
            json={"message": {"chat": {"id": 202}, "text": text}},
        )
        _, payload = tg_client.sent[-1]
        assert "Welcome to Hokm" in payload["text"]
        assert payload["reply_markup"]["inline_keyboard"][0][0]["web_app"] == {
            "url": "https://hokm.example.com"
        }

    def test_a_webhook_without_a_public_url_sends_text_and_does_not_crash(
        self, tg_client, monkeypatch
    ):
        import server

        monkeypatch.setattr(server, "WEBAPP_URL", "")
        answer = tg_client.post(
            "/telegram/webhook/s3cret",
            json={"message": {"chat": {"id": 202}, "text": "/start ABC234"}},
        )
        assert answer.status_code == 200
        _, payload = tg_client.sent[-1]
        assert "reply_markup" not in payload

    def test_following_the_same_link_twice_keeps_one_seat(self, client):
        code = _post(client, "/api/table/create", ALICE)[1]["code"]
        first = _post(client, "/api/table/join", BOB, {"code": code})[1]
        second = _post(client, "/api/table/join", BOB, {"code": code})[1]
        assert first["your_seat"] == second["your_seat"] == 1
        assert [p["kind"] for p in second["table"]["players"]] == [
            "human",
            "human",
            "open",
            "open",
        ]


class TestSeatChoice:
    """Seats 0 and 2 are one team, 1 and 3 the other, so two friends who
    arrive in sequence must be able to move onto the same side."""

    def test_a_player_can_move_to_a_free_seat_until_the_deal(self):
        store = TableStore()
        table = store.create("u0", "Ann")
        assert store.join("u1", "Ben", table.code).seat_of("u1") == 1
        assert store.join("u1", "Ben", table.code, seat=2).seat_of("u1") == 2
        assert table.seats[1] is None
        assert table.seats[2].name == "Ben"  # the mover keeps their identity
        with pytest.raises(GameServiceError, match="taken"):
            store.join("u1", "Ben", table.code, seat=0)
        table.start("u0")
        with pytest.raises(GameServiceError, match="already started"):
            store.join("u1", "Ben", table.code, seat=3)
        assert table.session.human_seats == {0: "Ann", 2: "Ben"}

    def test_racing_seat_moves_never_duplicate_a_seat(self):
        store = TableStore()
        table = store.create("u0", "Ann")
        store.join("u1", "Ben", table.code)
        errors = []

        def grab(user_id):
            for _ in range(20):
                try:
                    store.join(user_id, user_id, table.code, seat=3)
                    store.join(user_id, user_id, table.code, seat=2)
                except GameServiceError:
                    pass
                except Exception as exc:  # noqa: BLE001 — the assertion is "none"
                    errors.append(exc)

        threads = [threading.Thread(target=grab, args=(u,)) for u in ("u0", "u1")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert errors == []
        occupied = [s.user_id for s in table.seats if s is not None]
        assert sorted(occupied) == ["u0", "u1"]


class TestFinishedTable:
    def test_a_finished_table_never_tells_you_to_start_a_new_match(
        self, monkeypatch
    ):
        """A table cannot start a new match — its seats are fixed at the deal —
        so the inherited solo advice would be an impossible instruction."""
        monkeypatch.setenv("MATCH_TARGET", "1")
        _, table = _started_table()
        _play_hand(table, ("u0", "u1"))
        assert table.session.match_over is True
        for call in (
            lambda: table.next_hand("u0"),
            lambda: table.play_card("u0", "Ace of Spades"),
            lambda: table.set_trump("u0", "Hearts"),
        ):
            with pytest.raises(GameServiceError) as caught:
                call()
            assert "new match" not in str(caught.value)
            assert "leave the table" in str(caught.value)
        # The final score stays readable, so the end screen can show it.
        assert table.view("u0")["match_over"] is True


class TestApiSurface:
    def test_bad_table_requests_are_400_not_500(self, client):
        assert _get(client, "/api/table/state", ALICE)[0] == 400
        assert _post(client, "/api/table/join", ALICE, {"code": ""})[0] == 400
        assert _post(client, "/api/table/join", ALICE, {"code": "ZZZZZZ"})[0] == 400
        code = _post(client, "/api/table/create", ALICE)[1]["code"]
        assert _post(client, "/api/table/start", ALICE)[0] == 400  # needs two
        assert _post(client, "/api/table/play_card", ALICE, {"card": "x"})[0] == 400
        _post(client, "/api/table/join", BOB, {"code": code})
        assert _post(client, "/api/table/start", BOB)[0] == 400  # host only
        assert _post(client, "/api/table/start", ALICE)[0] == 200
        bad_suit = {"trump_suit": "X"}
        assert _post(client, "/api/table/set_trump", ALICE, bad_suit)[0] == 400
        assert _post(client, "/api/table/play_card", ALICE, {"card": ""})[0] == 400

    def test_unauthenticated_table_requests_are_401(self, client, monkeypatch):
        import server

        monkeypatch.setattr(server, "ALLOW_GUESTS", False)
        assert client.post("/api/table/create", json={}).status_code == 401
        assert client.get("/api/table/state").status_code == 401

    def test_since_long_polls_and_tolerates_junk(self, client, monkeypatch):
        monkeypatch.setattr(table_service, "POLL_TIMEOUT_SECONDS", 0.2)
        code = _post(client, "/api/table/create", ALICE)[1]["code"]
        _post(client, "/api/table/join", BOB, {"code": code})
        _post(client, "/api/table/start", ALICE)
        version = _get(client, "/api/table/state", ALICE)[1]["table"]["version"]

        began = time.time()
        status, body = _get(client, f"/api/table/state?since={version}", ALICE)
        assert status == 200 and body["table"]["version"] == version
        assert 0.15 < time.time() - began < 3.0

        status, body = _get(client, "/api/table/state?since=nonsense", ALICE)
        assert status == 200 and body["table"]["version"] == version

    def test_leaving_over_http(self, client):
        code = _post(client, "/api/table/create", ALICE)[1]["code"]
        _post(client, "/api/table/join", BOB, {"code": code})
        assert _post(client, "/api/table/leave", BOB)[0] == 200
        assert _get(client, "/api/table/state", BOB)[0] == 400

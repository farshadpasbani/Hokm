"""
Tests for the Telegram Mini App backend: initData verification,
multi-session game service, and the Flask API surface.
"""

import json
import time

import pytest

import telegram_auth
from telegram_auth import InitDataError, sign_init_data, verify_init_data


BOT_TOKEN = "1234567:TEST_TOKEN_abcDEF"


def _fresh_fields(user_id=42, name="Fara"):
    return {
        "auth_date": str(int(time.time())),
        "query_id": "AAA111",
        "user": json.dumps({"id": user_id, "first_name": name}),
    }


class TestInitDataVerification:
    def test_roundtrip(self):
        init_data = sign_init_data(_fresh_fields(), BOT_TOKEN)
        fields = verify_init_data(init_data, BOT_TOKEN)
        assert fields["user"]["id"] == 42
        assert fields["user"]["first_name"] == "Fara"

    def test_tampered_payload_rejected(self):
        init_data = sign_init_data(_fresh_fields(user_id=42), BOT_TOKEN)
        tampered = init_data.replace("42", "43")
        with pytest.raises(InitDataError, match="hash mismatch"):
            verify_init_data(tampered, BOT_TOKEN)

    def test_wrong_token_rejected(self):
        init_data = sign_init_data(_fresh_fields(), BOT_TOKEN)
        with pytest.raises(InitDataError, match="hash mismatch"):
            verify_init_data(init_data, "other:token")

    def test_expired_rejected(self):
        fields = _fresh_fields()
        fields["auth_date"] = str(int(time.time()) - 100_000)
        init_data = sign_init_data(fields, BOT_TOKEN)
        with pytest.raises(InitDataError, match="expired"):
            verify_init_data(init_data, BOT_TOKEN)

    def test_missing_hash_rejected(self):
        with pytest.raises(InitDataError, match="no hash"):
            verify_init_data("auth_date=1&user=x", BOT_TOKEN)


def _play_full_game(session):
    """Drive a session through one complete hand, always playing the first
    legal card. Returns the final payload."""
    data = session.new_game()
    if data["phase"] == "choose_trump":
        data = session.set_trump(data["trump_options"][0])
    for _ in range(60):  # 13 tricks max; generous bound
        if data.get("game_over"):
            return data
        assert data["your_turn"], f"expected human turn, got: {data['phase']}"
        assert data["legal_cards"], "human turn but no legal cards"
        data = session.play_card(data["legal_cards"][0])
    raise AssertionError("game did not finish within bound")


class TestGameSession:
    def test_full_game_completes_with_valid_scores(self):
        from game_service import GameSession

        final = _play_full_game(GameSession("guest:t1", "Tester"))
        t1, t2 = final["scores"]["Team 1"], final["scores"]["Team 2"]
        assert t1 + t2 == 13 or max(t1, t2) >= 7
        assert final["result"] in ("Team 1 wins", "Team 2 wins", "Draw")
        assert final["phase"] == "ended"

    def test_illegal_moves_rejected(self):
        from game_service import GameServiceError, GameSession

        sess = GameSession("guest:t2", "Tester")
        data = sess.new_game()
        if data["phase"] == "choose_trump":
            with pytest.raises(GameServiceError):
                sess.play_card("Ace of Spades")  # must pick trump first
            data = sess.set_trump(data["trump_options"][0])
        illegal = [c for c in data["hand"] if c not in data["legal_cards"]]
        if illegal:
            with pytest.raises(GameServiceError):
                sess.play_card(illegal[0])
        with pytest.raises(GameServiceError):
            sess.play_card("13 of Nothing")

    def test_sessions_are_isolated(self):
        from game_service import SessionStore

        store = SessionStore()
        a = store.get_or_create("tg:1", "A")
        b = store.get_or_create("tg:2", "B")
        a.new_game()
        assert a.game is not None
        assert b.game is None
        assert store.get_or_create("tg:1", "A") is a


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", BOT_TOKEN)
    # Force a clean import so server picks up the token + guest policy.
    import importlib
    import server

    monkeypatch.setattr(server, "BOT_TOKEN", BOT_TOKEN)
    monkeypatch.setattr(server, "ALLOW_GUESTS", False)
    importlib.reload(telegram_auth)  # no-op safety; keeps module state clean
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


class TestApi:
    def _auth_header(self):
        return {
            "Authorization": "tma " + sign_init_data(_fresh_fields(), BOT_TOKEN)
        }

    def test_unauthenticated_rejected(self, client):
        resp = client.post("/api/new_game", json={})
        assert resp.status_code == 401

    def test_guest_rejected_when_disabled(self, client):
        resp = client.post(
            "/api/new_game", json={}, headers={"X-Guest-Id": "abc123"}
        )
        assert resp.status_code == 401

    def test_authenticated_game_flow(self, client):
        headers = self._auth_header()
        resp = client.post("/api/new_game", json={}, headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["phase"] in ("choose_trump", "playing")

        if data["phase"] == "choose_trump":
            resp = client.post(
                "/api/set_trump",
                json={"trump_suit": data["trump_options"][0]},
                headers=headers,
            )
            assert resp.status_code == 200
            data = resp.get_json()
        assert data["phase"] == "playing"
        assert data["your_turn"] is True

        resp = client.post(
            "/api/play_card",
            json={"card": data["legal_cards"][0]},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "success"

        resp = client.get("/api/state", headers=headers)
        assert resp.status_code == 200

    def test_bad_card_is_400(self, client):
        headers = self._auth_header()
        data = client.post("/api/new_game", json={}, headers=headers).get_json()
        if data["phase"] == "choose_trump":
            data = client.post(
                "/api/set_trump",
                json={"trump_suit": data["trump_options"][0]},
                headers=headers,
            ).get_json()
        resp = client.post(
            "/api/play_card", json={"card": "nonsense"}, headers=headers
        )
        assert resp.status_code == 400

    def test_telegram_handle_reaches_the_player_log(self, client, monkeypatch):
        """The @handle only exists inside initData; the whole point of the
        player log is that it survives the request."""
        import server

        noted = []
        monkeypatch.setattr(
            server.STORE, "note_player", lambda p, **kw: noted.append(p)
        )
        fields = _fresh_fields()
        fields["user"] = json.dumps({
            "id": 42, "first_name": "Fara", "username": "farshad",
            "last_name": "P", "language_code": "fa", "is_premium": True,
        })
        client.post(
            "/api/new_game",
            json={},
            headers={"Authorization": "tma " + sign_init_data(fields, BOT_TOKEN)},
        )
        assert noted and noted[0]["username"] == "farshad"
        assert noted[0]["user_id"] == "tg:42"
        assert noted[0]["telegram_id"] == 42
        assert noted[0]["language_code"] == "fa"
        assert noted[0]["is_premium"] is True

    def test_player_without_a_handle_is_still_logged(self, client, monkeypatch):
        import server

        noted = []
        monkeypatch.setattr(
            server.STORE, "note_player", lambda p, **kw: noted.append(p)
        )
        client.post("/api/new_game", json={}, headers=self._auth_header())
        assert noted and noted[0]["username"] is None
        assert noted[0]["display_name"] == "Fara"

    def test_admin_players_requires_token(self, client):
        assert client.get("/api/admin/players").status_code == 404

    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_webhook_requires_secret(self, client):
        resp = client.post("/telegram/webhook/wrong", json={})
        assert resp.status_code == 403

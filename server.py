"""
Production entrypoint: Telegram Mini App backend for Hokm.

    gunicorn --workers 1 --threads 8 --bind 0.0.0.0:$PORT server:app

Serves:
  * `/`                      — the Mini App UI (templates/miniapp.html)
  * `/api/*`                 — per-user match API (Telegram initData auth):
                               new_game (new match), set_trump, play_card,
                               next_hand, state
  * `/telegram/webhook/<s>`  — bot webhook (answers /start with a Play button)
  * `/healthz`               — liveness probe

Environment:
  BOT_TOKEN        Telegram bot token (required in production; enables auth
                   and the webhook).
  WEBAPP_URL       Public HTTPS URL of this service (used in the /start
                   button and menu-button helper).
  WEBHOOK_SECRET   Path secret for the webhook route; also sent by Telegram
                   in X-Telegram-Bot-Api-Secret-Token when set via
                   scripts/set_webhook.py.
  ALLOW_GUESTS     "1" to accept unauthenticated browser sessions (guest ids
                   from the client). On by default when BOT_TOKEN is unset so
                   local dev works; set to "0" to hard-require Telegram auth.
  MODEL_PATH       Optional NFSP checkpoint for AI seats (see game_service).
  MATCH_TARGET     Hands a team must win to take the match (default 7).
  DATABASE_URL     Postgres connection string (Neon/Supabase/Render). When
                   set, every finished hand and every player who launches the
                   app is stored there — no persistent disk needed. See
                   store.py.
  GAME_DATA_DIR    Where finished hands are appended as JSONL training data
                   (default "game_data"; needs a persistent disk mount — the
                   container filesystem is ephemeral). Off by default when
                   DATABASE_URL is set.
  ADMIN_TOKEN      Enables /api/admin/stats, /api/admin/players and
                   /api/admin/export (download the recorded training data).
                   Unset = endpoints disabled.
  GAME_RECORDING   Forces local JSONL recording on ("1") or off ("0").

Session state is in-memory, so run exactly one gunicorn worker (use threads
for concurrency). Scale-out needs a shared store (e.g. Redis) — see
DEPLOYMENT.md.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

from game_service import GameServiceError, SessionStore
from store import STORE
from telegram_auth import InitDataError, verify_init_data

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger("hokm.server")

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBAPP_URL = os.getenv("WEBAPP_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
ALLOW_GUESTS = os.getenv("ALLOW_GUESTS", "" if BOT_TOKEN else "1") == "1"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

app = Flask(__name__)
sessions = SessionStore()

# Fail loudly (in logs) if the training-data directory is not writable —
# otherwise recording silently drops every hand (e.g. a mis-mounted disk).
def _probe_recorder() -> None:
    from game_service import RECORDER

    if not RECORDER.enabled:
        logger.info("Local JSONL recording disabled (GAME_RECORDING=0).")
        return
    try:
        os.makedirs(RECORDER.directory, exist_ok=True)
        probe = os.path.join(RECORDER.directory, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        logger.info("Game recording active: %s", RECORDER.directory)
    except OSError as e:
        logger.error(
            "Game recording DIRECTORY NOT WRITABLE (%s): %s — finished hands "
            "will NOT be saved. Check the disk mount / permissions.",
            RECORDER.directory, e,
        )


def _probe_store() -> None:
    """Connect once at boot so a bad DATABASE_URL shows up in the deploy log
    rather than silently swallowing every hand."""
    if not STORE.enabled:
        logger.warning(
            "DATABASE_URL not set — hands and players are NOT persisted off-box."
        )
        return
    stats = STORE.stats()
    if "players_total" in stats:
        logger.info(
            "Postgres store ready: %s players, %s hands recorded.",
            stats["players_total"], stats["hands_total"],
        )
    else:
        logger.error(
            "DATABASE_URL is set but the database is unreachable — finished "
            "hands will NOT be saved. Check the connection string."
        )


_probe_recorder()
_probe_store()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def _resolve_identity() -> Dict[str, Any]:
    """
    Returns the caller's profile: `user_id` and `display_name` always, plus
    the Telegram fields (`telegram_id`, `username`, `first_name`, `last_name`,
    `language_code`, `is_premium`) when the request is Telegram-authenticated.
    `username` is the @handle and is absent for users who have not set one.

    Preferred: `Authorization: tma <initData>` header, HMAC-verified against
    BOT_TOKEN. Fallback (dev / guests, only when ALLOW_GUESTS): a client-
    chosen `X-Guest-Id` header, namespaced so it can never collide with a
    real Telegram id.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("tma "):
        if not BOT_TOKEN:
            raise InitDataError(
                "Server is missing BOT_TOKEN; cannot verify Telegram auth."
            )
        fields = verify_init_data(auth[4:].strip(), BOT_TOKEN)
        user = fields.get("user") or {}
        user_id = user.get("id")
        if user_id is None:
            raise InitDataError("initData has no user id")
        username = (user.get("username") or "").strip() or None
        name = (user.get("first_name") or "").strip() or username or "You"
        return {
            "user_id": f"tg:{user_id}",
            "display_name": name,
            "telegram_id": user_id,
            "username": username,
            "first_name": (user.get("first_name") or "").strip() or None,
            "last_name": (user.get("last_name") or "").strip() or None,
            "language_code": user.get("language_code") or None,
            "is_premium": bool(user.get("is_premium")),
        }

    if ALLOW_GUESTS:
        guest_id = (request.headers.get("X-Guest-Id") or "").strip()
        if guest_id and len(guest_id) <= 64 and guest_id.isalnum():
            name = (request.headers.get("X-Guest-Name") or "You").strip()[:32]
            return {
                "user_id": f"guest:{guest_id}",
                "display_name": name or "You",
                "first_name": name or None,
            }

    raise InitDataError("Missing or invalid Telegram authorization.")


def _session():
    profile = _resolve_identity()
    # Throttled inside the store, so this is cheap on every request and still
    # keeps `last_seen` fresh for players who never finish a hand.
    STORE.note_player(profile)
    return sessions.get_or_create(profile["user_id"], profile["display_name"])


def _api_error(message: str, code: int):
    return jsonify({"status": "error", "message": message}), code


@app.errorhandler(GameServiceError)
def _handle_game_error(e: GameServiceError):
    return _api_error(str(e), 400)


@app.errorhandler(InitDataError)
def _handle_auth_error(e: InitDataError):
    return _api_error(str(e), 401)


# --------------------------------------------------------------------------
# Mini App UI + health
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("miniapp.html")


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "sessions": sessions.count()})


# --------------------------------------------------------------------------
# Game API
# --------------------------------------------------------------------------

@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    sess = _session()
    with sess.lock:
        return jsonify(sess.new_game())


@app.route("/api/next_hand", methods=["POST"])
def api_next_hand():
    """Deal the next hand of the current match (same seats, rotated Hakem)."""
    sess = _session()
    with sess.lock:
        return jsonify(sess.next_hand())


@app.route("/api/set_trump", methods=["POST"])
def api_set_trump():
    sess = _session()
    data = request.get_json(silent=True) or {}
    with sess.lock:
        return jsonify(sess.set_trump(data.get("trump_suit", "")))


@app.route("/api/play_card", methods=["POST"])
def api_play_card():
    sess = _session()
    data = request.get_json(silent=True) or {}
    with sess.lock:
        return jsonify(sess.play_card(data.get("card", "")))


@app.route("/api/state", methods=["GET"])
def api_state():
    sess = _session()
    with sess.lock:
        return jsonify(sess.state())


# --------------------------------------------------------------------------
# Training-data admin (token-gated; disabled entirely when ADMIN_TOKEN unset)
# --------------------------------------------------------------------------

def _admin_authorized() -> bool:
    if not ADMIN_TOKEN:
        return False
    supplied = request.args.get("token", "") or request.headers.get(
        "X-Admin-Token", ""
    )
    import hmac as _hmac

    return _hmac.compare_digest(supplied, ADMIN_TOKEN)


@app.route("/api/admin/stats")
def admin_stats():
    if not _admin_authorized():
        return "not found", 404
    from game_service import RECORDER

    return jsonify({"files": RECORDER.stats(), "database": STORE.stats()})


@app.route("/api/admin/players")
def admin_players():
    """The player log: who has launched the app, their @handle, and how much
    they have played. Most recently active first."""
    if not _admin_authorized():
        return "not found", 404
    try:
        limit = max(1, min(int(request.args.get("limit", "500")), 5000))
    except ValueError:
        limit = 500
    players = STORE.players(limit=limit)
    return jsonify({"count": len(players), "players": players})


@app.route("/api/admin/export")
def admin_export():
    """Stream every recorded hand as one concatenated JSONL download.

    Reads from Postgres when configured, otherwise from the local JSONL files.
    """
    if not _admin_authorized():
        return "not found", 404
    from game_service import RECORDER

    def generate():
        if STORE.enabled:
            for record in STORE.iter_hands():
                yield json.dumps(record, separators=(",", ":")) + "\n"
            return
        for path in RECORDER.files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield line
            except OSError:
                continue

    from flask import Response

    return Response(
        generate(),
        mimetype="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=hokm_hands.jsonl"},
    )


# --------------------------------------------------------------------------
# Telegram webhook (stdlib HTTP client; no extra dependency, no polling
# process — a single web service handles bot chat and the Mini App).
# --------------------------------------------------------------------------

def _bot_api(method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not BOT_TOKEN:
        return None
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/{method}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        logger.exception("Telegram Bot API call failed: %s", method)
        return None


@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if not WEBHOOK_SECRET or secret != WEBHOOK_SECRET:
        return "forbidden", 403
    header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if header_secret and header_secret != WEBHOOK_SECRET:
        return "forbidden", 403

    update = request.get_json(silent=True) or {}
    message = update.get("message") or {}
    text = (message.get("text") or "").strip()
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id and text.startswith("/start"):
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": (
                "🎴 Welcome to Hokm!\n\n"
                "Tap the button below to play the classic Persian card game "
                "against three AI opponents. You and your AI partner (North) "
                "are Team 1 — 7 tricks wins the hand, 7 hands wins the match."
            ),
        }
        if WEBAPP_URL:
            payload["reply_markup"] = {
                "inline_keyboard": [
                    [{"text": "▶️ Play Hokm", "web_app": {"url": WEBAPP_URL}}]
                ]
            }
        _bot_api("sendMessage", payload)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)

"""
Production entrypoint: Telegram Mini App backend for Hokm.

    gunicorn --workers 1 --threads 8 --bind 0.0.0.0:$PORT server:app

Serves:
  * `/`                      — the Mini App UI (templates/miniapp.html)
  * `/api/*`                 — per-user game API (Telegram initData auth)
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

Session state is in-memory, so run exactly one gunicorn worker (use threads
for concurrency). Scale-out needs a shared store (e.g. Redis) — see
DEPLOYMENT.md.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, render_template, request

from game_service import GameServiceError, SessionStore
from telegram_auth import InitDataError, verify_init_data

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger("hokm.server")

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
WEBAPP_URL = os.getenv("WEBAPP_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
ALLOW_GUESTS = os.getenv("ALLOW_GUESTS", "" if BOT_TOKEN else "1") == "1"

app = Flask(__name__)
sessions = SessionStore()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def _resolve_identity() -> Tuple[str, str]:
    """
    Returns (user_id, display_name).

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
        name = (user.get("first_name") or "").strip() or (
            user.get("username") or "You"
        )
        return f"tg:{user_id}", name

    if ALLOW_GUESTS:
        guest_id = (request.headers.get("X-Guest-Id") or "").strip()
        if guest_id and len(guest_id) <= 64 and guest_id.isalnum():
            name = (request.headers.get("X-Guest-Name") or "You").strip()[:32]
            return f"guest:{guest_id}", name or "You"

    raise InitDataError("Missing or invalid Telegram authorization.")


def _session():
    user_id, name = _resolve_identity()
    return sessions.get_or_create(user_id, name)


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
                "are Team 1 — first team to 7 tricks wins the hand."
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

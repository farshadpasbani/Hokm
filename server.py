"""
Production entrypoint: Telegram Mini App backend for Hokm.

    gunicorn --workers 1 --threads 8 --bind 0.0.0.0:$PORT server:app

Serves:
  * `/`                      — the Mini App UI (templates/miniapp.html)
  * `/api/*`                 — per-user match API (Telegram initData auth):
                               new_game (new match), set_trump, play_card,
                               next_hand, state, flag_trick
  * `/api/table/*`           — shared tables for 2-4 humans plus AI seats:
                               create, join, start, set_trump, play_card,
                               next_hand, leave, invite, state (polled)
  * `/telegram/webhook/<s>`  — bot webhook (answers /start with a Play
                               button, and `/start <code>` with a button
                               that opens the Mini App on that table)
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
  GAME_DATA_DIR    Where finished hands are appended as JSONL training data
                   (default "game_data"; point at a persistent disk mount in
                   production — the container filesystem is ephemeral).
  ADMIN_TOKEN      Enables /api/admin/stats and /api/admin/export (download
                   the recorded training data). Unset = endpoints disabled.
  GAME_RECORDING   "0" disables hand recording (default on).
  BOT_USERNAME     Bot username used to build a table's join deep link.
  MINI_APP_SHORT_NAME  Mini App short name; enables a `startapp` deep link.
  TABLE_IDLE_SECONDS   Silence before AI covers a table seat (default 45).
  DIRECTORY_TTL_SECONDS / MAX_DIRECTORY_ENTRIES
                   Lifetime and cap of the @handle → user-id directory that
                   makes invites by handle possible (see table_service).

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

import table_service
from game_service import GameServiceError, SessionStore
from table_service import TableStore, UserDirectory
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
tables = TableStore()
directory = UserDirectory()

# Fail loudly (in logs) if the training-data directory is not writable —
# otherwise recording silently drops every hand (e.g. a mis-mounted disk).
def _probe_recorder() -> None:
    from game_service import RECORDER

    if not RECORDER.enabled:
        logger.info("Game recording disabled (GAME_RECORDING=0).")
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


_probe_recorder()


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
        # Every authenticated request is the one place a *verified* @handle is
        # visible, so this is where the invite directory is filled. Telegram
        # offers no username lookup of its own (see table_service).
        directory.remember(user.get("username") or "", f"tg:{user_id}")
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


@app.route("/api/flag_trick", methods=["POST"])
def api_flag_trick():
    """Tester feedback: flag the trick the CLIENT is displaying as bad AI play.

    The index comes from the client because the server runs ahead of the
    animation — see GameSession.flag_trick.
    """
    sess = _session()
    data = request.get_json(silent=True) or {}
    with sess.lock:
        return jsonify(sess.flag_trick(data.get("trick_index")))


@app.route("/api/state", methods=["GET"])
def api_state():
    sess = _session()
    with sess.lock:
        return jsonify(sess.state())


# --------------------------------------------------------------------------
# Shared tables (2-4 humans + AI seats). The solo `/api/*` surface above is
# untouched; everything multi-human lives under `/api/table/*`.
#
# Locking lives inside `table_service` — one lock per table, because two
# players of the same table land on two of gunicorn's threads at once.
# --------------------------------------------------------------------------

@app.route("/api/table/create", methods=["POST"])
def api_table_create():
    user_id, name = _resolve_identity()
    table = tables.create(user_id, name)
    body = table.view(user_id)
    body["code"] = table.code
    body["join_url"] = body["table"]["join_url"]
    return jsonify(body)


@app.route("/api/table/join", methods=["POST"])
def api_table_join():
    user_id, name = _resolve_identity()
    data = request.get_json(silent=True) or {}
    table = tables.join(user_id, name, data.get("code", ""), data.get("seat"))
    body = table.view(user_id)
    body["code"] = table.code
    body["join_url"] = body["table"]["join_url"]
    return jsonify(body)


@app.route("/api/table/start", methods=["POST"])
def api_table_start():
    user_id, _ = _resolve_identity()
    return jsonify(tables.for_user(user_id).start(user_id))


@app.route("/api/table/set_trump", methods=["POST"])
def api_table_set_trump():
    user_id, _ = _resolve_identity()
    data = request.get_json(silent=True) or {}
    return jsonify(
        tables.for_user(user_id).set_trump(user_id, data.get("trump_suit", ""))
    )


@app.route("/api/table/play_card", methods=["POST"])
def api_table_play_card():
    user_id, _ = _resolve_identity()
    data = request.get_json(silent=True) or {}
    return jsonify(tables.for_user(user_id).play_card(user_id, data.get("card", "")))


@app.route("/api/table/next_hand", methods=["POST"])
def api_table_next_hand():
    user_id, _ = _resolve_identity()
    return jsonify(tables.for_user(user_id).next_hand(user_id))


@app.route("/api/table/invite", methods=["POST"])
def api_table_invite():
    """
    Invite an `@handle` to the caller's table.

    Telegram has no username → user-id lookup, so a handle resolves only
    against the directory this service fills from verified initData. Both
    outcomes ship: a known handle gets a bot DM carrying a join button, an
    unknown one does not, and *either way* the answer carries the share link
    and says plainly what happened. `delivered` is true only when the Bot API
    confirmed the send — a missing BOT_TOKEN, a bot the invitee has never
    started, or any API failure is reported as not delivered rather than
    silently swallowed.

    Any seated player may invite, not only the host: everyone at the table
    already holds the join code and could paste it anyway, so restricting it
    would add a rule that protects nothing.
    """
    user_id, _ = _resolve_identity()
    table = tables.for_user(user_id)
    handle = table_service.normalize_handle(
        (request.get_json(silent=True) or {}).get("handle", "")
    )
    if not handle:
        raise GameServiceError("Type your friend's @handle to invite them.")
    invitee = directory.lookup(handle)
    delivered = bool(invitee) and _dm_invite(invitee, table)
    join_url = table_service.deep_link(table.code)
    # Without BOT_USERNAME there is no link to pass on, so the fallback has to
    # be the join code itself rather than an instruction to send nothing.
    fallback = "send them this link" if join_url else f"give them the code {table.code}"
    if not invitee:
        message = f"@{handle} has not opened Hokm yet — {fallback}."
    elif delivered:
        message = f"Invite sent to @{handle}."
    else:
        message = (
            f"Could not message @{handle} — they may never have opened a chat "
            f"with the bot. Instead, {fallback}."
        )
    return jsonify({
        "status": "success",
        "handle": handle,
        "known": bool(invitee),
        "delivered": delivered,
        "code": table.code,
        "join_url": join_url,
        "message": message,
    })


@app.route("/api/table/leave", methods=["POST"])
def api_table_leave():
    user_id, _ = _resolve_identity()
    tables.leave(user_id)
    return jsonify({"status": "success", "left": True})


@app.route("/api/table/state", methods=["GET"])
def api_table_state():
    """
    This player's view of the table.

    `?since=<version>` long-polls: the request parks until the table's
    version passes `since` or `TABLE_POLL_TIMEOUT_SECONDS` elapses, then
    answers either way. Polling rather than streaming is deliberate — the
    service runs one worker with eight threads, so a held-open stream per
    player would exhaust the pool (see table_service's module docstring).
    A malformed `since` is treated as absent.
    """
    user_id, _ = _resolve_identity()
    table = tables.for_user(user_id)
    try:
        since = int(request.args.get("since", ""))
    except ValueError:
        since = None
    body = table.view(user_id)
    if since is not None and body["table"]["version"] <= since:
        table.wait_for_change(since, table_service.POLL_TIMEOUT_SECONDS)
        body = table.view(user_id)
    return jsonify(body)


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

    return jsonify(RECORDER.stats())


@app.route("/api/admin/export")
def admin_export():
    """Stream every recorded hand as one concatenated JSONL download."""
    if not _admin_authorized():
        return "not found", 404
    from game_service import RECORDER

    def generate():
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


def _table_url(code: str) -> str:
    """
    Mini App URL that opens straight onto one table.

    A `web_app` button carries a URL, not a Telegram `start_param`, so the
    code travels as an ordinary query parameter that `miniapp.html` reads on
    load (it also accepts `start_param`, which is how the `?startapp=` link
    shape arrives).
    """
    separator = "&" if "?" in WEBAPP_URL else "?"
    return f"{WEBAPP_URL}{separator}table={code}"


def _join_button(code: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [{"text": "▶️ Join the table", "web_app": {"url": _table_url(code)}}]
        ]
    }


def _dm_invite(invitee_id: str, table) -> bool:
    """
    DM one invitee a join button. True only when Telegram confirmed the send.

    A DM can only land in a chat the invitee already opened with the bot, and
    `_bot_api` answers None when BOT_TOKEN is missing or the call failed — so
    the return value is the only honest basis for telling the inviter that
    their friend was messaged.
    """
    if not invitee_id.startswith("tg:"):
        return False
    payload: Dict[str, Any] = {
        "chat_id": int(invitee_id[3:]),
        "text": (
            "🎴 You have been invited to a game of Hokm!\n\n"
            f"Join code: {table.code}"
        ),
    }
    if WEBAPP_URL:
        payload["reply_markup"] = _join_button(table.code)
    response = _bot_api("sendMessage", payload)
    return bool(response and response.get("ok"))


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
        # `t.me/<bot>?start=<code>` — the deep-link shape used when no Mini App
        # short name is configured — arrives here as "/start <code>". Anything
        # that is not one of our join codes falls back to the plain welcome.
        parts = text.split()
        code = table_service.clean_code(parts[1]) if len(parts) > 1 else ""
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": (
                f"🎴 You have been invited to a game of Hokm!\n\n"
                f"Join code: {code}"
                if code
                else "🎴 Welcome to Hokm!\n\n"
                "Tap the button below to play the classic Persian card game "
                "against three AI opponents. You and your AI partner (North) "
                "are Team 1 — 7 tricks wins the hand, 7 hands wins the match."
            ),
        }
        if WEBAPP_URL:
            payload["reply_markup"] = (
                _join_button(code)
                if code
                else {
                    "inline_keyboard": [
                        [{"text": "▶️ Play Hokm", "web_app": {"url": WEBAPP_URL}}]
                    ]
                }
            )
        _bot_api("sendMessage", payload)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)

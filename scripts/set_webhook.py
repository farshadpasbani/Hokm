#!/usr/bin/env python3
"""
One-shot Telegram bot configuration for the Hokm Mini App.

Run after deploying the service (locally or from any machine):

    BOT_TOKEN=... WEBAPP_URL=https://your-app.example.com \
    WEBHOOK_SECRET=some-long-random-string \
    python scripts/set_webhook.py

It configures:
  1. The bot webhook  -> {WEBAPP_URL}/telegram/webhook/{WEBHOOK_SECRET}
  2. The chat menu button -> opens the Mini App (this is what makes the bot
     "playable" from the attachment/menu button, alongside the /start button)
  3. The bot's command list (/start)

Uses only the standard library.
"""

import json
import os
import sys
import urllib.request


def bot_api(token: str, method: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    token = os.getenv("BOT_TOKEN", "")
    webapp_url = os.getenv("WEBAPP_URL", "").rstrip("/")
    secret = os.getenv("WEBHOOK_SECRET", "")
    if not token or not webapp_url or not secret:
        print("Set BOT_TOKEN, WEBAPP_URL and WEBHOOK_SECRET first.", file=sys.stderr)
        return 1
    if not webapp_url.startswith("https://"):
        print("WEBAPP_URL must be HTTPS (Telegram requirement).", file=sys.stderr)
        return 1

    steps = [
        (
            "setWebhook",
            {
                "url": f"{webapp_url}/telegram/webhook/{secret}",
                "secret_token": secret,
                "allowed_updates": ["message"],
            },
        ),
        (
            "setChatMenuButton",
            {
                "menu_button": {
                    "type": "web_app",
                    "text": "Play Hokm",
                    "web_app": {"url": webapp_url},
                }
            },
        ),
        (
            "setMyCommands",
            {"commands": [{"command": "start", "description": "Play Hokm"}]},
        ),
    ]
    ok = True
    for method, payload in steps:
        result = bot_api(token, method, payload)
        print(f"{method}: {result}")
        ok = ok and bool(result.get("ok"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

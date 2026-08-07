"""
Telegram Mini App initData verification.

Implements the HMAC scheme documented at
https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app:

    secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token)
    hash       = hex(HMAC_SHA256(key=secret_key, msg=data_check_string))

where data_check_string is all received key=value pairs except `hash`,
sorted by key and joined with "\n".

`verify_init_data` returns the parsed fields (with `user` JSON-decoded)
on success and raises `InitDataError` on any failure, including stale
`auth_date` (default max age 24 h — Telegram re-issues initData on every
Mini App launch, so a long-expired blob is a replay).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict
from urllib.parse import parse_qsl


DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60


class InitDataError(ValueError):
    """Raised when Telegram initData fails validation."""


def _secret_key(bot_token: str) -> bytes:
    return hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()


def verify_init_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: float | None = None,
) -> Dict[str, Any]:
    if not init_data:
        raise InitDataError("empty initData")
    if not bot_token:
        raise InitDataError("server has no BOT_TOKEN configured")

    try:
        pairs = parse_qsl(init_data, strict_parsing=True, keep_blank_values=True)
    except ValueError as e:
        raise InitDataError(f"initData is not a query string: {e}") from e

    fields = dict(pairs)
    received_hash = fields.pop("hash", None)
    if not received_hash:
        raise InitDataError("initData has no hash field")

    data_check_string = "\n".join(
        f"{k}={v}" for k, v in sorted(fields.items(), key=lambda kv: kv[0])
    )
    expected_hash = hmac.new(
        _secret_key(bot_token), data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, received_hash):
        raise InitDataError("initData hash mismatch")

    try:
        auth_date = int(fields.get("auth_date", "0"))
    except ValueError as e:
        raise InitDataError("initData auth_date is not an integer") from e
    current = time.time() if now is None else now
    if auth_date <= 0 or current - auth_date > max_age_seconds:
        raise InitDataError("initData is expired")

    if "user" in fields:
        try:
            fields["user"] = json.loads(fields["user"])
        except json.JSONDecodeError as e:
            raise InitDataError("initData user field is not valid JSON") from e

    return fields


def sign_init_data(fields: Dict[str, str], bot_token: str) -> str:
    """
    Build a signed initData query string from raw string fields — the inverse
    of `verify_init_data`. Used by tests and local tooling only; real initData
    always comes from Telegram.
    """
    from urllib.parse import urlencode

    data_check_string = "\n".join(
        f"{k}={v}" for k, v in sorted(fields.items(), key=lambda kv: kv[0])
    )
    signature = hmac.new(
        _secret_key(bot_token), data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    return urlencode({**fields, "hash": signature})

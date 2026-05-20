"""Read live market quotes from Redis."""

from __future__ import annotations

import json
from typing import Any

from backend.app.core.paths import NIFTY_TOP_5, SIGNALS_LATEST_KEY, quote_key
from backend.app.core.redis_client import get_redis


def get_live_quotes() -> list[dict[str, Any]]:
    redis_client = get_redis()
    quotes: list[dict[str, Any]] = []
    for symbol in NIFTY_TOP_5:
        raw = redis_client.get(quote_key(symbol))
        if raw:
            quotes.append(json.loads(raw))
    return quotes


def get_latest_signal() -> dict[str, Any] | None:
    redis_client = get_redis()
    raw = redis_client.get(SIGNALS_LATEST_KEY)
    if not raw:
        return None
    return json.loads(raw)

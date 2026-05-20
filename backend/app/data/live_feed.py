"""
Zerodha KiteTicker live WebSocket feed.

Updates Redis with live quotes and runs RSI checks every 5 minutes.

Run from project root:
    python backend/app/data/live_feed.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

from backend.app.core.paths import (
    NIFTY_TOP_5,
    PROJECT_ROOT,
    SIGNALS_LATEST_KEY,
    quote_key,
)
from backend.app.core.redis_client import get_redis
from backend.app.strategies.rsi_strategy import RSIStrategy

ENV_PATH = PROJECT_ROOT / ".env"

RSI_CHECK_INTERVAL_SEC = 300
MIN_BARS_FOR_RSI = 20
RECONNECT_BASE_DELAY = 2
RECONNECT_MAX_DELAY = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_env() -> None:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env not found at {ENV_PATH}")
    load_dotenv(ENV_PATH, override=True)


def _get_env(key: str) -> str:
    value = os.getenv(key)
    if not value or not value.strip():
        raise ValueError(f"{key} is not set in {ENV_PATH}")
    return value.strip()


def _create_kite() -> KiteConnect:
    _load_env()
    kite = KiteConnect(api_key=_get_env("KITE_API_KEY"))
    kite.set_access_token(_get_env("KITE_ACCESS_TOKEN"))
    return kite


def _resolve_instrument_tokens(kite: KiteConnect) -> tuple[list[int], dict[int, str]]:
    instruments = kite.instruments("NSE")
    token_to_symbol: dict[int, str] = {}
    tokens: list[int] = []

    for symbol in NIFTY_TOP_5:
        found = False
        for inst in instruments:
            if (
                inst["tradingsymbol"] == symbol
                and inst["exchange"] == "NSE"
                and inst["instrument_type"] == "EQ"
            ):
                token = int(inst["instrument_token"])
                tokens.append(token)
                token_to_symbol[token] = symbol
                found = True
                break
        if not found:
            raise ValueError(f"NSE instrument not found for {symbol}")

    return tokens, token_to_symbol


class MinuteBarBuilder:
    """Build 1-minute OHLCV bars from live ticks."""

    def __init__(self) -> None:
        self._bars: dict[str, dict[datetime, dict[str, Any]]] = defaultdict(dict)
        self._last_volume: dict[str, int] = {}

    def add_tick(self, symbol: str, price: float, volume_traded: int, ts: datetime) -> None:
        minute_ts = ts.replace(second=0, microsecond=0)
        prev_vol = self._last_volume.get(symbol, volume_traded)
        vol_delta = max(volume_traded - prev_vol, 0) if symbol in self._last_volume else 0
        self._last_volume[symbol] = volume_traded

        symbol_bars = self._bars[symbol]
        if minute_ts not in symbol_bars:
            symbol_bars[minute_ts] = {
                "date": minute_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": float(vol_delta),
            }
        else:
            bar = symbol_bars[minute_ts]
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["volume"] += float(vol_delta)

        # Keep last 3 days of minute bars in memory
        cutoff = datetime.now() - timedelta(days=3)
        stale = [k for k in symbol_bars if k < cutoff]
        for key in stale:
            del symbol_bars[key]

    def to_dataframe(self, symbol: str) -> pd.DataFrame:
        symbol_bars = self._bars.get(symbol, {})
        if not symbol_bars:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        rows = sorted(symbol_bars.values(), key=lambda x: x["date"])
        return pd.DataFrame(rows)


class LiveFeed:
    """KiteTicker WebSocket client with Redis publishing and RSI checks."""

    def __init__(self) -> None:
        _load_env()
        self.api_key = _get_env("KITE_API_KEY")
        self.access_token = _get_env("KITE_ACCESS_TOKEN")
        self.redis = get_redis()
        self.kite = _create_kite()
        self.instrument_tokens, self.token_to_symbol = _resolve_instrument_tokens(
            self.kite
        )
        self.bar_builder = MinuteBarBuilder()
        self.rsi_strategy = RSIStrategy()
        self._kws: Optional[KiteTicker] = None
        self._connected = False
        self._running = True
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._rsi_thread: Optional[threading.Thread] = None
        self._last_signal_bar: dict[str, datetime] = {}

    def _tick_timestamp(self, tick: dict[str, Any]) -> datetime:
        ts = tick.get("exchange_timestamp") or tick.get("last_trade_time")
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None) if ts.tzinfo else ts
        return datetime.now()

    def _update_redis_quote(self, symbol: str, tick: dict[str, Any]) -> None:
        ts = self._tick_timestamp(tick)
        payload = {
            "symbol": symbol,
            "price": float(tick.get("last_price", 0)),
            "volume": int(tick.get("volume_traded", 0)),
            "timestamp": ts.isoformat(),
        }
        self.redis.set(quote_key(symbol), json.dumps(payload))

    def _on_ticks(self, _ws: KiteTicker, ticks: list[dict[str, Any]]) -> None:
        for tick in ticks:
            token = tick.get("instrument_token")
            symbol = self.token_to_symbol.get(token)
            if not symbol:
                continue
            price = float(tick.get("last_price", 0))
            volume = int(tick.get("volume_traded", 0))
            ts = self._tick_timestamp(tick)
            try:
                self._update_redis_quote(symbol, tick)
                self.bar_builder.add_tick(symbol, price, volume, ts)
            except Exception as exc:
                logger.error("Failed to process tick for %s: %s", symbol, exc)

    def _on_connect(self, ws: KiteTicker, _response: Any) -> None:
        self._connected = True
        self._reconnect_delay = RECONNECT_BASE_DELAY
        logger.info("KiteTicker connected. Subscribing to %s", ", ".join(NIFTY_TOP_5))
        ws.subscribe(self.instrument_tokens)
        ws.set_mode(ws.MODE_QUOTE, self.instrument_tokens)

    def _on_close(self, _ws: KiteTicker, code: int, reason: str) -> None:
        self._connected = False
        logger.warning("KiteTicker closed: code=%s reason=%s", code, reason)

    def _on_error(self, _ws: KiteTicker, code: int, reason: str) -> None:
        logger.error("KiteTicker error: code=%s reason=%s", code, reason)

    def _on_reconnect(self, _ws: KiteTicker, attempts_count: int) -> None:
        logger.info("KiteTicker reconnecting (attempt %s)...", attempts_count)

    def _on_noreconnect(self, _ws: KiteTicker) -> None:
        logger.warning("KiteTicker max reconnect attempts reached.")

    def _publish_signal(
        self, symbol: str, signal: int, price: float, ts: datetime
    ) -> None:
        payload = {
            "symbol": symbol,
            "signal": signal,
            "price": price,
            "time": ts.isoformat(),
        }
        self.redis.set(SIGNALS_LATEST_KEY, json.dumps(payload))
        label = "BUY" if signal == 1 else "SELL"
        logger.info("Signal generated: %s %s @ %.2f", symbol, label, price)

    def _run_rsi_check(self) -> None:
        while self._running:
            time.sleep(RSI_CHECK_INTERVAL_SEC)
            if not self._running:
                break
            try:
                self._check_rsi_signals()
            except Exception as exc:
                logger.error("RSI check failed: %s", exc)

    def _check_rsi_signals(self) -> None:
        logger.info("Running RSI strategy check on accumulated bars...")
        for symbol in NIFTY_TOP_5:
            df = self.bar_builder.to_dataframe(symbol)
            if len(df) < MIN_BARS_FOR_RSI:
                logger.debug(
                    "%s: only %s bars, need %s for RSI",
                    symbol,
                    len(df),
                    MIN_BARS_FOR_RSI,
                )
                continue

            df_signals = self.rsi_strategy.generate_signals(df)
            last_row = df_signals.iloc[-1]
            signal = int(last_row["signal"])
            if signal == 0:
                continue

            bar_time = pd.Timestamp(last_row["date"]).to_pydatetime()
            prev = self._last_signal_bar.get(symbol)
            if prev == bar_time:
                continue

            self._last_signal_bar[symbol] = bar_time
            self._publish_signal(
                symbol=symbol,
                signal=signal,
                price=float(last_row["close"]),
                ts=bar_time,
            )

    def _start_rsi_thread(self) -> None:
        self._rsi_thread = threading.Thread(
            target=self._run_rsi_check, name="rsi-checker", daemon=True
        )
        self._rsi_thread.start()

    def _create_ticker(self) -> KiteTicker:
        kws = KiteTicker(self.api_key, self.access_token)
        kws.on_ticks = self._on_ticks
        kws.on_connect = self._on_connect
        kws.on_close = self._on_close
        kws.on_error = self._on_error
        kws.on_reconnect = self._on_reconnect
        kws.on_noreconnect = self._on_noreconnect
        return kws

    def _stop_ticker(self) -> None:
        if self._kws is not None:
            try:
                self._kws.close()
            except Exception as exc:
                logger.debug("Ticker close: %s", exc)
            self._kws = None
        self._connected = False

    def run(self) -> None:
        """Run feed with automatic reconnect on disconnect."""
        self._start_rsi_thread()
        logger.info("Starting live feed for: %s", ", ".join(NIFTY_TOP_5))

        while self._running:
            try:
                self._kws = self._create_ticker()
                self._kws.connect(threaded=True)

                for _ in range(30):
                    if self._connected or not self._running:
                        break
                    time.sleep(1)

                if not self._connected and self._running:
                    raise ConnectionError("KiteTicker failed to connect within 30 seconds")

                while self._running and self._connected:
                    time.sleep(1)

                if not self._running:
                    break

                self._stop_ticker()
                logger.info("Reconnecting in %s seconds...", self._reconnect_delay)
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, RECONNECT_MAX_DELAY
                )
            except KeyboardInterrupt:
                self._running = False
                break
            except Exception as exc:
                logger.error("Live feed error: %s", exc)
                self._stop_ticker()
                if not self._running:
                    break
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, RECONNECT_MAX_DELAY
                )

        self._stop_ticker()
        logger.info("Live feed stopped.")

    def stop(self) -> None:
        self._running = False
        self._stop_ticker()


def main() -> None:
    try:
        feed = LiveFeed()
        feed.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

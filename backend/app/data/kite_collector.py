"""
Zerodha Kite Connect historical data collector for Indian equities.
Run: python backend/app/data/kite_collector.py
"""

import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
RATE_LIMIT_SLEEP = 0.4

# Max candles per historical_data request (Kite Connect limits)
_INTERVAL_MAX_DAYS = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}

_instruments_cache: Optional[list] = None


def _load_env() -> None:
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f".env file not found at {ENV_PATH}. "
            "Copy .env.example to .env and set KITE_API_KEY and KITE_API_SECRET."
        )
    load_dotenv(ENV_PATH, override=True)


def _get_env(key: str) -> str:
    value = os.getenv(key)
    if not value or value.strip() == "":
        raise ValueError(f"{key} is not set in {ENV_PATH}")
    return value.strip()


def _create_kite(require_access_token: bool = True) -> KiteConnect:
    _load_env()
    api_key = _get_env("KITE_API_KEY")
    kite = KiteConnect(api_key=api_key)
    if require_access_token:
        access_token = _get_env("KITE_ACCESS_TOKEN")
        kite.set_access_token(access_token)
    return kite


def _parse_date(value: Union[str, date, datetime]) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _get_nse_instruments(kite: KiteConnect) -> list:
    global _instruments_cache
    if _instruments_cache is None:
        _instruments_cache = kite.instruments("NSE")
        time.sleep(RATE_LIMIT_SLEEP)
    return _instruments_cache


def _get_instrument_token(kite: KiteConnect, symbol: str) -> int:
    instruments = _get_nse_instruments(kite)
    symbol_upper = symbol.upper()
    for inst in instruments:
        if (
            inst["tradingsymbol"] == symbol_upper
            and inst["exchange"] == "NSE"
            and inst["instrument_type"] == "EQ"
        ):
            return inst["instrument_token"]
    raise ValueError(f"NSE equity instrument not found for symbol: {symbol}")


def generate_login_url() -> str:
    """Print and return the Zerodha login URL for browser authentication."""
    try:
        kite = _create_kite(require_access_token=False)
        login_url = kite.login_url()
        print("Open this URL in your browser to log in to Zerodha Kite:")
        print(login_url)
        print(
            "\nAfter login, copy the request_token from the redirect URL "
            "and call get_access_token(request_token)."
        )
        return login_url
    except Exception as exc:
        print(f"Error generating login URL: {exc}", file=sys.stderr)
        raise


def get_access_token(request_token: str) -> str:
    """Exchange request_token for access_token and persist it in .env."""
    try:
        kite = _create_kite(require_access_token=False)
        api_secret = _get_env("KITE_API_SECRET")
        session = kite.generate_session(request_token.strip(), api_secret=api_secret)
        access_token = session["access_token"]
        set_key(str(ENV_PATH), "KITE_ACCESS_TOKEN", access_token)
        os.environ["KITE_ACCESS_TOKEN"] = access_token
        print(f"Access token obtained and saved to {ENV_PATH}")
        return access_token
    except KiteException as exc:
        print(f"Kite API error while generating session: {exc}", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"Error obtaining access token: {exc}", file=sys.stderr)
        raise


def fetch_historical_data(
    symbol: str,
    interval: str,
    from_date: Union[str, date, datetime],
    to_date: Union[str, date, datetime],
) -> pd.DataFrame:
    """
    Fetch OHLCV historical data for an NSE symbol.

    Returns a DataFrame with columns: date (datetime64), open, high, low, close, volume.
    """
    try:
        kite = _create_kite(require_access_token=True)
        instrument_token = _get_instrument_token(kite, symbol)
        start = _parse_date(from_date)
        end = _parse_date(to_date)
        if start > end:
            raise ValueError(f"from_date ({start}) must be on or before to_date ({end})")

        max_days = _INTERVAL_MAX_DAYS.get(interval)
        if max_days is None:
            raise ValueError(
                f"Unsupported interval: {interval}. "
                f"Supported: {', '.join(_INTERVAL_MAX_DAYS)}"
            )

        all_records: list[dict] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=max_days - 1), end)
            records = kite.historical_data(
                instrument_token,
                chunk_start,
                chunk_end,
                interval,
                continuous=False,
                oi=False,
            )
            time.sleep(RATE_LIMIT_SLEEP)
            if records:
                all_records.extend(records)
            chunk_start = chunk_end + timedelta(days=1)

        if not all_records:
            print(f"Warning: No historical data returned for {symbol}")
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(all_records)
        df = df.rename(
            columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except KiteException as exc:
        print(f"Kite API error fetching {symbol}: {exc}", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"Error fetching historical data for {symbol}: {exc}", file=sys.stderr)
        raise


def main() -> None:
    """Fetch 3 years of daily OHLCV for core symbols and save to data/raw/."""
    try:
        _load_env()
        if not os.getenv("KITE_ACCESS_TOKEN", "").strip():
            print(
                "KITE_ACCESS_TOKEN is missing. Authenticate first:\n"
                "  1. Call generate_login_url() and log in via browser\n"
                "  2. Call get_access_token('<request_token>')\n"
                "Then run this script again."
            )
            sys.exit(1)

        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        to_date = date.today()
        from_date = to_date - timedelta(days=3 * 365)
        print(
            f"Fetching daily data from {from_date} to {to_date} "
            f"for {len(SYMBOLS)} symbols..."
        )

        for symbol in SYMBOLS:
            try:
                print(f"Fetching {symbol}...")
                df = fetch_historical_data(
                    symbol=symbol,
                    interval="day",
                    from_date=from_date,
                    to_date=to_date,
                )
                if df.empty:
                    print(f"Skipping save for {symbol}: no data returned.")
                    continue
                output_path = DATA_RAW_DIR / f"{symbol}.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved {len(df)} rows to {output_path}")
            except Exception as exc:
                print(f"Failed to fetch/save {symbol}: {exc}", file=sys.stderr)
                continue

        print("Done.")
    except Exception as exc:
        print(f"Fatal error in main(): {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Download NIFTY50 and Nifty top-5 stock OHLCV data via yfinance.

Usage (from project root):
    python backend/app/data/download_nifty.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
DATA_DIR = "data/raw"

TICKERS = {
    "NIFTY50": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
}


def clean_dataframe(df: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        print(f"  ERROR: No data returned for {symbol}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower().strip() for c in df.columns]

    rename_map = {
        "adj close": "adj_close",
        "stock splits": "stock_splits",
        "capital gains": "capital_gains",
    }
    df = df.rename(columns=rename_map)

    df = df.reset_index()
    for col in ["index", "Date", "Datetime", "datetime"]:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            break

    keep = ["date", "open", "high", "low", "close", "volume"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing]

    if "close" not in df.columns:
        print(f"  ERROR: No close column found for {symbol}. Columns: {list(df.columns)}")
        return None

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype(int)

    return df


def download_ticker(symbol: str, ticker: str, data_dir: str) -> bool:
    print(f"\nDownloading {symbol} ({ticker})...")
    try:
        raw_df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
        )
        df = clean_dataframe(raw_df, symbol)
        if df is None:
            return False

        save_path = os.path.join(data_dir, f"{symbol}.csv")
        df.to_csv(save_path, index=False)

        print(f"  Rows downloaded : {len(df)}")
        print(f"  Date range      : {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"  Price range     : Rs.{df['close'].min():,.2f} to Rs.{df['close'].max():,.2f}")
        print(f"  Saved to        : {save_path}")
        return True
    except Exception as exc:
        print(f"  ERROR downloading {symbol}: {exc}")
        return False


def verify_existing_csv(symbol: str, data_dir: str) -> dict:
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        return {"exists": False}

    df = pd.read_csv(path)
    if "date" not in df.columns:
        return {"exists": True, "rows": len(df), "start": "N/A", "end": "N/A", "path": path}

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return {"exists": True, "rows": len(df), "start": "N/A", "end": "N/A", "path": path}

    return {
        "exists": True,
        "rows": len(df),
        "start": dates.iloc[0].strftime("%Y-%m-%d"),
        "end": dates.iloc[-1].strftime("%Y-%m-%d"),
        "path": path,
    }


def print_summary(results: dict[str, bool]) -> None:
    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"{'Symbol':<12} {'Status':<10} {'Rows':<8} {'From':<12} {'To':<12}")
    print("-" * 65)

    for symbol, success in results.items():
        status = "OK" if success else "FAILED"
        info = verify_existing_csv(symbol, DATA_DIR)
        rows = str(info.get("rows", "-")) if info.get("exists") else "-"
        start = info.get("start", "-") if info.get("exists") else "-"
        end = info.get("end", "-") if info.get("exists") else "-"
        print(f"{symbol:<12} {status:<10} {rows:<8} {start:<12} {end:<12}")

    print("-" * 65)
    success_count = sum(1 for ok in results.values() if ok)
    total = len(results)
    print(f"Downloaded: {success_count}/{total} symbols successfully")
    print(f"Data saved to: {DATA_DIR}/")
    print("=" * 65)
    print(
        """
NEXT STEP:
Run your backtest with the fresh data:
    python backend/app/backtesting/run_backtest.py
"""
    )


def _find_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    for _ in range(6):
        if os.path.isfile(os.path.join(project_root, "pyproject.toml")):
            return project_root
        project_root = os.path.dirname(project_root)
    return os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))


def main() -> None:
    project_root = _find_project_root()
    os.chdir(project_root)
    print(f"Working directory set to: {project_root}")

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 65)
    print("QUANTEDGE - Market Data Downloader")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Symbols: {list(TICKERS.keys())}")
    print("=" * 65)

    print("\nChecking existing files...")
    for symbol in TICKERS:
        info = verify_existing_csv(symbol, DATA_DIR)
        if info["exists"]:
            print(
                f"  {symbol}: EXISTS ({info['rows']} rows, "
                f"{info['start']} to {info['end']})"
            )
        else:
            print(f"  {symbol}: NOT FOUND - will download")

    print("\nOptions:")
    print("  1. Download ALL (overwrite existing)")
    print("  2. Download MISSING only (skip existing)")
    print("  3. Download NIFTY50 only")

    choice = input("\nEnter choice (1/2/3) or press Enter for option 2: ").strip()
    if not choice:
        choice = "2"

    results: dict[str, bool] = {}
    for symbol, ticker in TICKERS.items():
        should_download = False

        if choice == "1":
            should_download = True
        elif choice == "2":
            existing = verify_existing_csv(symbol, DATA_DIR)
            should_download = not existing["exists"]
        elif choice == "3":
            should_download = symbol == "NIFTY50"

        if should_download:
            results[symbol] = download_ticker(symbol, ticker, DATA_DIR)
        else:
            print(f"\nSkipping {symbol} (already exists)")
            results[symbol] = True

    print_summary(results)

    nifty_path = os.path.join(DATA_DIR, "NIFTY50.csv")
    if os.path.exists(nifty_path):
        df = pd.read_csv(nifty_path)
        print("\nNIFTY50 Quick Check:")
        print(f"  Total trading days : {len(df)}")
        df["return"] = df["close"].pct_change()
        print(f"  Best day           : +{df['return'].max() * 100:.2f}%")
        print(f"  Worst day          : {df['return'].min() * 100:.2f}%")
        print(f"  Avg daily return   : {df['return'].mean() * 100:.3f}%")
        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        total_return = (last_close - first_close) / first_close * 100
        print(f"  Total 4yr return   : {total_return:+.2f}%")
        print(f"  Start price        : Rs.{first_close:,.2f}")
        print(f"  End price          : Rs.{last_close:,.2f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        sys.exit(1)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)

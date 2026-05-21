"""
Run RSI vs EMA strategy comparison across Nifty top-5 symbols.

Usage (from project root):
    python backend/app/backtesting/run_backtest.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from backend.app.backtesting.engine import Backtester
from backend.app.strategies.ema_strategy import EMAStrategy
from backend.app.strategies.rsi_strategy import RSIStrategy

SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
DATA_DIR = "data/raw"
RESULTS_DIR = "data/processed"
INITIAL_CAPITAL = 100_000

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_PATH = os.path.join(PROJECT_ROOT, DATA_DIR)
RESULTS_PATH = os.path.join(PROJECT_ROOT, RESULTS_DIR)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"date", "datetime", "timestamp", "time"}:
            rename[col] = "date"
        elif key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key
        elif key.endswith("close") or key in {"close price", "adj close"}:
            rename[col] = "close"
        elif "open" in key:
            rename[col] = "open"
        elif "high" in key:
            rename[col] = "high"
        elif "low" in key:
            rename[col] = "low"
        elif "volume" in key or key == "vol":
            rename[col] = "volume"
    return df.rename(columns=rename)


def load_csv(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_PATH, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"[WARNING] {symbol}: file not found at {path}")
        return None

    df = pd.read_csv(path)
    df = _standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").reset_index(drop=True)

    start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    print(f"Loaded {symbol}: {len(df)} rows, {start_date} to {end_date}")
    return df


def load_nifty() -> pd.DataFrame | None:
    path = os.path.join(DATA_PATH, "NIFTY50.csv")
    if not os.path.exists(path):
        print("WARNING: NIFTY50.csv not found - regime filter disabled")
        return None

    df = pd.read_csv(path)
    df = _standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded NIFTY50 regime data: {len(df)} rows")
    return df


def _sharpe_indicator(sharpe: float) -> str:
    if sharpe >= 1.0:
        return "*"
    if sharpe >= 0.5:
        return "^"
    if sharpe >= 0.0:
        return "-"
    return "v"


def _verdict_label(avg_sharpe: float) -> str:
    if avg_sharpe >= 1.0:
        return "EXCELLENT *"
    if avg_sharpe >= 0.5:
        return "GOOD ^"
    if avg_sharpe >= 0.0:
        return "AVERAGE -"
    return "WEAK v"


def run_strategy_comparison(
    stock_data: dict[str, pd.DataFrame], nifty_df: pd.DataFrame | None
) -> None:
    """Run RSI and EMA strategies on all symbols; print side-by-side comparison."""
    strategies = [
        {
            "name": "RSI(14) 45/60",
            "short": "RSI",
            "build": lambda: RSIStrategy(
                rsi_period=14,
                oversold=45,
                overbought=60,
                nifty_df=nifty_df,
            ),
        },
        {
            "name": "EMA(9/21) + Vol",
            "short": "EMA_Vol",
            "build": lambda: EMAStrategy(
                fast_period=9,
                slow_period=21,
                volume_filter=True,
                nifty_df=nifty_df,
            ),
        },
        {
            "name": "EMA(9/21) NoVol",
            "short": "EMA_NoVol",
            "build": lambda: EMAStrategy(
                fast_period=9,
                slow_period=21,
                volume_filter=False,
                nifty_df=nifty_df,
            ),
        },
        {
            "name": "EMA(5/20) + Vol",
            "short": "EMA_Fast",
            "build": lambda: EMAStrategy(
                fast_period=5,
                slow_period=20,
                volume_filter=True,
                nifty_df=nifty_df,
            ),
        },
    ]

    bt_config = {
        "initial_capital": INITIAL_CAPITAL,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
        "max_holding_days": 15,
    }

    all_results: dict[str, dict] = {}

    for config in strategies:
        strat_results: dict[str, dict | None] = {}
        print(f"\n  Running {config['name']}...")

        for symbol, df in stock_data.items():
            try:
                strategy = config["build"]()
                df_signals = strategy.generate_signals(df.copy())
                bt = Backtester(**bt_config)
                bt.run(df_signals, symbol)
                r = bt.results()
                strat_results[symbol] = r
                print(
                    f"    {symbol}: Sharpe={r['sharpe_ratio']:+.2f} "
                    f"Trades={r['total_trades']}"
                )
            except Exception as exc:
                print(f"    {symbol}: ERROR - {exc}")
                strat_results[symbol] = None

        all_results[config["short"]] = {
            "name": config["name"],
            "results": strat_results,
        }

    print("\n")
    print("=" * 90)
    print("  STRATEGY COMPARISON - RSI vs EMA Crossover Variants")
    print(
        "  All settings: SL=5% | TP=10% | MaxHold=15d | "
        "Capital=Rs.1,00,000 | NIFTY50 Regime Filter ON"
    )
    print("=" * 90)

    winner_rows: list[tuple[str, float, float]] = []

    for config in strategies:
        short = config["short"]
        name = config["name"]
        data = all_results[short]["results"]

        pad = max(0, 60 - len(name))
        print(f"\n  +-- {name} {'-' * pad}+")
        print(
            f"  | {'Symbol':<12} {'CAGR':>7} {'Sharpe':>7} "
            f"{'MaxDD':>8} {'WinRate':>8} "
            f"{'Trades':>7} {'PF':>6} |"
        )
        print(f"  |{'-' * 65}|")

        sharpes: list[float] = []
        cagrs: list[float] = []

        for symbol in SYMBOLS:
            r = data.get(symbol)
            if r is None or r.get("total_trades", 0) == 0:
                print(f"  | {symbol:<12} {'NO TRADES':>42} |")
                continue

            sharpe = float(r["sharpe_ratio"])
            sharpes.append(sharpe)
            cagrs.append(float(r["cagr"]))
            ind = _sharpe_indicator(sharpe)
            pf = r["profit_factor"]
            pf_str = f"{pf:.2f}" if pf < 999 else "inf"

            print(
                f"  | {symbol:<12} "
                f"{r['cagr']:>+6.1f}% "
                f"{sharpe:>6.2f}{ind} "
                f"{r['max_drawdown_pct']:>7.1f}% "
                f"{r['win_rate']:>7.1f}% "
                f"{r['total_trades']:>7} "
                f"{pf_str:>6} |"
            )

        print(f"  |{'-' * 65}|")

        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
        avg_cagr = sum(cagrs) / len(cagrs) if cagrs else 0.0
        verdict = _verdict_label(avg_sharpe)

        print(
            f"  | {'AVERAGE':<12} "
            f"{avg_cagr:>+6.1f}% "
            f"{avg_sharpe:>6.2f}  "
            f"{'':>7} {'':>7} {'':>7} {'':>6} |"
        )
        print(f"  | VERDICT: {verdict:<54} |")
        print(f"  +{'-' * 65}+")

        winner_rows.append((name, avg_sharpe, avg_cagr))

    print("\n" + "=" * 65)
    print("WINNER BY AVERAGE SHARPE RATIO")
    print("=" * 65)
    print(
        f"{'Strategy':<20} {'Avg Sharpe':>12} {'Avg CAGR':>10} {'Verdict':>15}"
    )
    print("-" * 65)

    winner_rows.sort(key=lambda x: x[1], reverse=True)
    best_name, best_sharpe, best_cagr = winner_rows[0] if winner_rows else ("N/A", 0.0, 0.0)

    for i, (name, avg_sharpe, avg_cagr) in enumerate(winner_rows):
        tag = " <- BEST" if i == 0 else ""
        print(
            f"{name:<20} {avg_sharpe:>12.2f} "
            f"{avg_cagr:>+9.1f}% {_verdict_label(avg_sharpe):>15}{tag}"
        )

    print("=" * 65)

    if best_sharpe >= 0.5:
        print(f"\nRECOMMENDATION: Use {best_name}")
        print("ACTION: Build Streamlit dashboard + start paper trading")
        print("SIZING: Rs.10,000 per trade to start")
    elif best_sharpe >= 0.0:
        print(f"\nRECOMMENDATION: {best_name} shows some promise")
        print("ACTION: Test on hourly data before paper trading")
        print("TRY: EMA(9/21) on 1-hour OHLCV data")
    else:
        print("\nRECOMMENDATION: Neither strategy ready yet")
        print("ACTION: Try VWAP breakout on hourly data")
        print("NEXT PROMPT: Ask for VWAP intraday strategy")


def main() -> None:
    try:
        print("=" * 65)
        print("QUANTEDGE - Strategy Comparison Runner")
        print("RSI Mean Reversion vs EMA Crossover")
        print("=" * 65)

        print("\nLoading data...")
        nifty_df = load_nifty()
        stock_data: dict[str, pd.DataFrame] = {}
        for symbol in SYMBOLS:
            df = load_csv(symbol)
            if df is not None:
                stock_data[symbol] = df

        if len(stock_data) == 0:
            print("ERROR: No data found.")
            print("Run: python backend/app/data/download_nifty.py")
            sys.exit(1)

        print(f"Loaded: {list(stock_data.keys())}")
        regime_status = "NIFTY50 filter ON" if nifty_df is not None else "filter OFF"
        print(f"Regime: {regime_status}")

        print("\nRunning all strategies (this takes ~30 seconds)...")
        run_strategy_comparison(stock_data, nifty_df)

        os.makedirs(RESULTS_PATH, exist_ok=True)
        print("\nDone. Check results above.")
        print("Next step: Build Streamlit dashboard")
        print("  streamlit run frontend/dashboard.py")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

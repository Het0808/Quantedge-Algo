"""
Run RSI strategy backtest comparison on RELIANCE daily data.

Tests three strategy variants and prints a comparison table.

Usage (from project root):
    python run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "RELIANCE.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.backtesting.engine import Backtester
from backend.app.strategies.rsi_strategy import RSIStrategy


def generate_synthetic_reliance_data() -> pd.DataFrame:
    dates = pd.bdate_range(start="2022-01-01", end="2024-12-31")
    n = len(dates)
    rng = np.random.default_rng(42)
    close = 2400.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, size=n))

    rows = []
    for i, dt in enumerate(dates):
        c = close[i]
        o = c * (1 + rng.normal(0, 0.003))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.008)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.008)))
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": int(rng.integers(500_000, 5_000_000)),
            }
        )
    return pd.DataFrame(rows)


def load_ohlcv(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        print(f"CSV not found at {path}. Generating synthetic RELIANCE data...")
        path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_reliance_data()
        df.to_csv(path, index=False)
        df["date"] = pd.to_datetime(df["date"])

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def run_variant(
    name: str,
    strategy: RSIStrategy,
    df: pd.DataFrame,
    symbol: str,
    initial_capital: float = 100_000,
) -> dict:
    df_signals = strategy.generate_signals(df)
    backtester = Backtester(initial_capital=initial_capital)
    backtester.run(df_signals, symbol=symbol)
    metrics = backtester.results()
    metrics["version"] = name
    return metrics


def print_comparison(rows: list[dict]) -> None:
    header = (
        f"{'Version':<52} | {'Trades':>6} | {'CAGR':>8} | {'Sharpe':>7} "
        f"| {'MaxDD':>8} | {'WinRate':>8} | {'ProfitFactor':>12}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("RSI Strategy Comparison — RELIANCE")
    print(sep)
    print(header)
    print(sep)
    for m in rows:
        pf = m["profit_factor"]
        pf_str = f"{pf:.2f}" if pf < 999 else "inf"
        print(
            f"{m['version']:<52} | {m['total_trades']:>6} | "
            f"{m['cagr']:>7.2f}% | {m['sharpe_ratio']:>7.2f} | "
            f"{m['max_drawdown_pct']:>7.2f}% | {m['win_rate']:>7.2f}% | "
            f"{pf_str:>12}"
        )
    print(f"{sep}\n")


def main() -> None:
    try:
        df = load_ohlcv(DATA_PATH)
        symbol = "RELIANCE"

        variants = [
            (
                "A: Original (OS=30, OB=70, no trend)",
                RSIStrategy(oversold=30, overbought=70, trend_filter=False),
            ),
            (
                "B: Looser  (OS=40, OB=60, no trend)",
                RSIStrategy(oversold=40, overbought=60, trend_filter=False),
            ),
            (
                "C: Looser + Trend Filter (OS=40, OB=60, EMA50)",
                RSIStrategy(oversold=40, overbought=60, trend_filter=True),
            ),
        ]

        results = []
        for name, strategy in variants:
            results.append(run_variant(name, strategy, df, symbol))

        print_comparison(results)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Run RSI backtest on RELIANCE and print formatted results.

Usage (from project root):
    python -m backend.app.backtesting.run_backtest
    python backend/app/backtesting/run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "RELIANCE.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "backtest_results.csv"
TRADES_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "backtest_trades.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.backtesting.engine import Backtester
from backend.app.strategies.rsi_strategy import RSIStrategy


def generate_synthetic_reliance_data() -> pd.DataFrame:
    """Three years of realistic daily OHLCV starting near Rs. 2400."""
    dates = pd.bdate_range(start="2022-01-01", end="2024-12-31")
    n = len(dates)
    rng = np.random.default_rng(42)

    daily_returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    close = 2400.0 * np.cumprod(1 + daily_returns)

    rows = []
    for i, dt in enumerate(dates):
        c = close[i]
        intraday = abs(rng.normal(0, 0.008))
        o = c * (1 + rng.normal(0, 0.003))
        h = max(o, c) * (1 + intraday)
        l = min(o, c) * (1 - intraday)
        vol = int(rng.integers(500_000, 5_000_000))
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": vol,
            }
        )

    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        print(f"CSV not found at {DATA_PATH}. Generating synthetic RELIANCE data...")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_reliance_data()
        df.to_csv(DATA_PATH, index=False)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing columns: {missing}")

    return df.sort_values("date").reset_index(drop=True)


def format_period(df: pd.DataFrame) -> str:
    start = pd.Timestamp(df["date"].iloc[0]).strftime("%Y-%m-%d")
    end = pd.Timestamp(df["date"].iloc[-1]).strftime("%Y-%m-%d")
    return f"{start} to {end}"


def format_profit_factor(value: float) -> str:
    if value >= 999:
        return "inf"
    return f"{value:.2f}"


def print_results(
    df: pd.DataFrame, metrics: dict, capital: float, backtester: Backtester
) -> None:
    alpha = metrics["total_return_pct"] - metrics["benchmark_return_pct"]
    alpha_sign = "+" if alpha >= 0 else ""
    winning_trades = metrics["total_trades"] - metrics["losing_trades"]

    print()
    print("=" * 48)
    print("BACKTEST RESULTS — RELIANCE RSI Strategy")
    print("=" * 48)
    print(f"Period        : {format_period(df)}")
    print(f"Capital       : Rs. {capital:,.0f}")
    print(f"Stop Loss     : {backtester.stop_loss_pct * 100:.0f}%")
    print(f"Take Profit   : {backtester.take_profit_pct * 100:.0f}%")
    print("-" * 48)
    print(f"Total Return  : {metrics['total_return_pct']:.2f}%")
    print(f"CAGR          : {metrics['cagr']:.2f}%")
    print(f"Sharpe Ratio  : {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown  : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win Rate      : {metrics['win_rate']:.2f}%")
    print(f"Total Trades  : {metrics['total_trades']}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades : {metrics['losing_trades']}")
    print(f"Avg Win       : {metrics['avg_win_pct']:+.2f}%")
    print(f"Avg Loss      : {metrics['avg_loss_pct']:+.2f}%")
    print(f"Profit Factor : {format_profit_factor(metrics['profit_factor'])}")
    print(
        "Exit reasons  : "
        f"{metrics['signal_exits']} signal / "
        f"{metrics['stop_loss_exits']} stop loss / "
        f"{metrics['take_profit_exits']} take profit"
    )
    print("-" * 48)
    print(
        f"Benchmark     : {metrics['benchmark_return_pct']:.2f}% "
        f"(Buy & Hold RELIANCE)"
    )
    print(f"Alpha         : {alpha_sign}{alpha:.2f}%")
    print("=" * 48)
    print()


def save_outputs(df: pd.DataFrame, metrics: dict, backtester: Backtester) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d")
            if pd.api.types.is_datetime64_any_dtype(df["date"])
            else df["date"],
            "portfolio_value": metrics["equity_curve"],
        }
    )
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Equity curve saved to {OUTPUT_PATH}")

    trades_df = backtester.trades_table
    if not trades_df.empty:
        trades_out = trades_df.copy()
        for col in ("entry_date", "exit_date"):
            if col in trades_out.columns:
                trades_out[col] = pd.to_datetime(trades_out[col]).dt.strftime(
                    "%Y-%m-%d"
                )
        trades_out.to_csv(TRADES_OUTPUT_PATH, index=False)
        print(f"Trade log saved to {TRADES_OUTPUT_PATH}")


def main() -> None:
    capital = 100_000
    try:
        df = load_data()
        df = RSIStrategy().generate_signals(df)
        backtester = Backtester(
            initial_capital=capital,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        backtester.run(df, "RELIANCE")
        metrics = backtester.results()
        print_results(df, metrics, capital, backtester)
        save_outputs(df, metrics, backtester)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

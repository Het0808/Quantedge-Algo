"""CSV market data loading and strategy helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.backtesting.engine import Backtester
from backend.app.core.paths import DATA_DIRS, KNOWN_SYMBOLS, VALID_STRATEGIES
from backend.app.strategies.ema_strategy import EMAStrategy
from backend.app.strategies.rsi_strategy import RSIStrategy


def _find_csv_path(symbol: str) -> Path | None:
    symbol_upper = symbol.upper()
    for data_dir in DATA_DIRS:
        path = Path(data_dir) / f"{symbol_upper}.csv"
        if path.exists():
            return path
    return None


def list_available_symbols() -> list[str]:
    available = []
    for symbol in KNOWN_SYMBOLS:
        if _find_csv_path(symbol) is not None:
            available.append(symbol)
    return available


def load_symbol_data(symbol: str) -> pd.DataFrame:
    path = _find_csv_path(symbol)
    if path is None:
        raise FileNotFoundError(
            f"No CSV found for {symbol.upper()}. "
            "Run: python backend/app/data/kite_collector.py"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def get_latest_close(symbol: str) -> float:
    df = load_symbol_data(symbol)
    return float(df["close"].iloc[-1])


def get_historical(symbol: str, days: int) -> list[dict[str, Any]]:
    df = load_symbol_data(symbol)
    subset = df.tail(days).copy()
    subset["date"] = subset["date"].dt.strftime("%Y-%m-%d")
    return subset[["date", "open", "high", "low", "close", "volume"]].to_dict(
        orient="records"
    )


def get_strategy(strategy: str):
    key = strategy.lower().strip()
    if key not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: rsi, ema")
    if key == "rsi":
        return RSIStrategy()
    return EMAStrategy()


def filter_date_range(
    df: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (df["date"] >= start) & (df["date"] <= end)
    filtered = df.loc[mask].reset_index(drop=True)
    if len(filtered) < 2:
        raise ValueError("Date range must contain at least 2 trading days.")
    return filtered


def run_backtest(
    symbol: str,
    strategy: str,
    start_date: str,
    end_date: str,
    capital: float,
) -> dict[str, Any]:
    df = filter_date_range(load_symbol_data(symbol), start_date, end_date)
    df_signals = get_strategy(strategy).generate_signals(df)
    backtester = Backtester(initial_capital=capital)
    backtester.run(df_signals, symbol=symbol.upper())
    results = backtester.results()

    equity_curve = []
    dates = backtester.dates or df_signals["date"].tolist()
    for dt, value in zip(dates, results["equity_curve"]):
        equity_curve.append(
            {
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "value": round(float(value), 2),
            }
        )

    trades_table = backtester.trades_table
    trades: list[dict[str, Any]] = []
    if not trades_table.empty:
        for _, row in trades_table.iterrows():
            if "exit_date" in row:
                trade = {
                    "date": pd.Timestamp(row["exit_date"]).strftime("%Y-%m-%d")
                    if pd.notna(row["exit_date"])
                    else None,
                    "type": "SELL",
                    "entry_price": row["entry_price"],
                    "exit_price": row["exit_price"],
                    "shares": int(row["shares"]),
                    "pnl": row["pnl"] if pd.notna(row.get("pnl")) else None,
                    "pnl_pct": row.get("pnl_pct"),
                    "exit_reason": row.get("exit_reason"),
                }
            else:
                trade = {
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
                    if pd.notna(row.get("date"))
                    else None,
                    "type": row.get("type", "TRADE"),
                    "price": row.get("price"),
                    "shares": int(row["shares"]),
                    "fee": row.get("fee"),
                    "pnl": row["pnl"] if pd.notna(row.get("pnl")) else None,
                }
            trades.append(trade)

    final_value = results["equity_curve"][-1] if results["equity_curve"] else capital
    return {
        "symbol": symbol.upper(),
        "strategy": strategy.lower(),
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": capital,
        "final_portfolio_value": round(final_value, 2),
        "total_return_pct": results["total_return_pct"],
        "cagr": results["cagr"],
        "sharpe_ratio": results["sharpe_ratio"],
        "max_drawdown_pct": results["max_drawdown_pct"],
        "win_rate": results["win_rate"],
        "total_trades": results["total_trades"],
        "benchmark_return_pct": results.get("benchmark_return_pct", 0.0),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def get_signals(symbol: str, strategy: str, days: int = 30) -> list[dict[str, Any]]:
    df = get_strategy(strategy).generate_signals(load_symbol_data(symbol))
    subset = df.tail(days).copy()
    subset["date"] = subset["date"].dt.strftime("%Y-%m-%d")
    cols = ["date", "open", "high", "low", "close", "volume", "signal"]
    return subset[cols].to_dict(orient="records")

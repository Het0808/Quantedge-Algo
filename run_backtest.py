"""
Run RSI strategy backtest on RELIANCE daily data.

Usage (from project root):
    python run_backtest.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "RELIANCE.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.backtesting.engine import Backtester
from backend.app.strategies.rsi_strategy import RSIStrategy


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run the Kite collector first: python backend/app/data/kite_collector.py"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    try:
        df = load_ohlcv(DATA_PATH)
        symbol = "RELIANCE"

        strategy = RSIStrategy()
        df_signals = strategy.generate_signals(df)

        backtester = Backtester(initial_capital=100_000)
        backtester.run(df_signals, symbol=symbol)
        metrics = backtester.results()

        print(f"\n{'=' * 50}")
        print(f"Backtest Results — {metrics['symbol']}")
        print(f"{'=' * 50}")
        print(f"Initial Capital:        ₹{metrics['initial_capital']:,.2f}")
        print(f"Final Portfolio Value:  ₹{metrics['final_portfolio_value']:,.2f}")
        print(f"Total Return:           {metrics['total_return_pct']:.2f}%")
        print(f"CAGR:                   {metrics['cagr']:.2f}%")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
        print(f"Win Rate:               {metrics['win_rate']:.2f}%")
        print(f"Total Trades:           {metrics['total_trades']}")
        print(f"Equity Curve Points:    {len(metrics['equity_curve'])}")
        print(f"{'=' * 50}\n")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

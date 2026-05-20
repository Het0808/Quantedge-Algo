"""Bar-by-bar backtesting engine with next-bar execution (no lookahead)."""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
BROKERAGE_RATE = 0.0003
MIN_BROKERAGE = 20.0


class Backtester:
    """
    Simple long-only backtester.

    Signals on bar i execute at bar i+1 open. Portfolio is marked to market
    at each bar's close.
    """

    def __init__(self, initial_capital: float = 100_000) -> None:
        self.initial_capital = initial_capital
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self.cash = self.initial_capital
        self.position = 0
        self.symbol: Optional[str] = None
        self.equity_curve: list[float] = []
        self.pending_signal: Optional[int] = None
        self._closed_trades: list[dict[str, float]] = []
        self._entry_price: Optional[float] = None
        self._total_executions = 0
        self._dates: list[pd.Timestamp] = []

    @staticmethod
    def _brokerage(trade_value: float) -> float:
        return max(trade_value * BROKERAGE_RATE, MIN_BROKERAGE)

    @staticmethod
    def _max_affordable_shares(cash: float, price: float) -> int:
        if price <= 0:
            return 0
        max_shares = int(cash // price)
        for shares in range(max_shares, 0, -1):
            trade_value = shares * price
            if trade_value + Backtester._brokerage(trade_value) <= cash:
                return shares
        return 0

    def _execute_buy(self, price: float, bar_index: int) -> None:
        shares = self._max_affordable_shares(self.cash, price)
        if shares <= 0:
            return

        trade_value = shares * price
        fee = self._brokerage(trade_value)
        self.cash -= trade_value + fee
        self.position = shares
        self._entry_price = price
        self._total_executions += 1
        self._closed_trades.append(
            {"type": "buy", "price": price, "shares": shares, "fee": fee, "bar": bar_index}
        )

    def _execute_sell(self, price: float, bar_index: int) -> None:
        if self.position <= 0:
            return

        shares = self.position
        trade_value = shares * price
        fee = self._brokerage(trade_value)
        proceeds = trade_value - fee
        self.cash += proceeds

        entry = self._entry_price if self._entry_price is not None else price
        pnl = (price - entry) * shares - fee
        if len(self._closed_trades) > 0 and self._closed_trades[-1]["type"] == "buy":
            buy_fee = self._closed_trades[-1]["fee"]
            pnl -= buy_fee

        self._closed_trades.append(
            {
                "type": "sell",
                "price": price,
                "shares": shares,
                "fee": fee,
                "pnl": pnl,
                "bar": bar_index,
            }
        )

        self.position = 0
        self._entry_price = None
        self._total_executions += 1

    def run(self, df_with_signals: pd.DataFrame, symbol: str) -> "Backtester":
        """
        Run backtest bar by bar. Signals execute on the next bar's open.
        """
        self._reset_run_state()
        self.symbol = symbol

        required = {"open", "high", "low", "close", "volume", "signal"}
        missing = required - set(df_with_signals.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        df = df_with_signals.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        n = len(df)
        if n < 2:
            raise ValueError("Need at least 2 bars to run a backtest with next-bar execution.")

        for i in range(n):
            row = df.iloc[i]
            open_price = float(row["open"])
            close_price = float(row["close"])

            if self.pending_signal == 1 and self.position == 0:
                self._execute_buy(open_price, i)
            elif self.pending_signal == -1 and self.position > 0:
                self._execute_sell(open_price, i)
            self.pending_signal = None

            portfolio_value = self.cash + self.position * close_price
            self.equity_curve.append(portfolio_value)
            if "date" in df.columns:
                self._dates.append(row["date"])

            if i < n - 1:
                signal = int(row["signal"])
                if signal == 1 and self.position == 0:
                    self.pending_signal = 1
                elif signal == -1 and self.position > 0:
                    self.pending_signal = -1

        return self

    def _years_elapsed(self) -> float:
        if len(self._dates) >= 2:
            delta = self._dates[-1] - self._dates[0]
            return max(delta.days / 365.25, 1 / TRADING_DAYS_PER_YEAR)
        return max(len(self.equity_curve) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)

    def _round_trip_pnls(self) -> list[float]:
        pnls: list[float] = []
        for trade in self._closed_trades:
            if trade["type"] == "sell" and "pnl" in trade:
                pnls.append(trade["pnl"])
        return pnls

    def results(self) -> dict[str, Any]:
        """Return performance metrics and the daily equity curve."""
        if not self.equity_curve:
            raise RuntimeError("No backtest results. Call run() first.")

        initial = self.initial_capital
        final = self.equity_curve[-1]
        total_return_pct = ((final - initial) / initial) * 100

        years = self._years_elapsed()
        if final > 0 and initial > 0:
            cagr = ((final / initial) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0

        equity = pd.Series(self.equity_curve, dtype=float)
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        else:
            sharpe_ratio = 0.0

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0.0

        round_trip_pnls = self._round_trip_pnls()
        if round_trip_pnls:
            wins = sum(1 for p in round_trip_pnls if p > 0)
            win_rate = (wins / len(round_trip_pnls)) * 100
        else:
            win_rate = 0.0

        return {
            "symbol": self.symbol,
            "initial_capital": initial,
            "final_portfolio_value": round(final, 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr": round(cagr, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": self._total_executions,
            "equity_curve": self.equity_curve,
            "dates": self._dates,
            "trades_table": self._trades_table(),
        }

    def _trades_table(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for trade in self._closed_trades:
            bar = int(trade["bar"])
            trade_date = self._dates[bar] if bar < len(self._dates) else None
            row: dict[str, Any] = {
                "date": trade_date,
                "type": trade["type"].upper(),
                "price": round(trade["price"], 2),
                "shares": int(trade["shares"]),
                "fee": round(trade["fee"], 2),
            }
            if trade["type"] == "sell":
                row["pnl"] = round(trade.get("pnl", 0.0), 2)
            else:
                row["pnl"] = None
            rows.append(row)
        return pd.DataFrame(rows)

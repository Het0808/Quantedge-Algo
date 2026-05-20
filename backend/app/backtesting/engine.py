"""Bar-by-bar backtesting engine with next-bar execution (no lookahead)."""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.065
BROKERAGE_RATE = 0.0003
MIN_BROKERAGE = 20.0


class Backtester:
    """Long-only backtester with stop-loss, take-profit, and signal exits."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
    ) -> None:
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.symbol: str = "SYMBOL"
        self.cash: float = initial_capital
        self.shares_held: int = 0
        self.equity_curve: list[float] = []
        self._dates: list[Any] = []
        self._completed_trades: list[dict[str, Any]] = []
        self._benchmark_return_pct: float = 0.0
        self._df: Optional[pd.DataFrame] = None
        self._open_position: Optional[dict[str, Any]] = None

    @staticmethod
    def _brokerage(trade_value: float) -> float:
        return max(MIN_BROKERAGE, trade_value * BROKERAGE_RATE)

    @staticmethod
    def _max_affordable_shares(cash: float, price: float) -> int:
        if price <= 0:
            return 0
        for shares in range(int(cash // price), 0, -1):
            trade_value = shares * price
            if trade_value + Backtester._brokerage(trade_value) <= cash:
                return shares
        return 0

    def _row_date(self, row: pd.Series, idx: int) -> Any:
        if "date" in row.index and pd.notna(row["date"]):
            return row["date"]
        return idx

    def _execute_buy(self, fill_price: float, idx: int, row: pd.Series) -> None:
        shares = self._max_affordable_shares(self.cash, fill_price)
        if shares <= 0:
            return

        trade_value = shares * fill_price
        buy_fee = self._brokerage(trade_value)
        self.cash -= trade_value + buy_fee
        self.shares_held = shares
        self._open_position = {
            "entry_price": fill_price,
            "entry_date": self._row_date(row, idx),
            "shares": shares,
            "buy_fee": buy_fee,
            "entry_bar": idx,
        }

    def _execute_sell(
        self,
        fill_price: float,
        idx: int,
        row: pd.Series,
        exit_reason: str,
    ) -> None:
        if self.shares_held <= 0 or self._open_position is None:
            return

        pos = self._open_position
        shares = self.shares_held
        entry_price = float(pos["entry_price"])
        buy_fee = float(pos["buy_fee"])
        trade_value = shares * fill_price
        sell_fee = self._brokerage(trade_value)
        proceeds = trade_value - sell_fee

        cost_basis = entry_price * shares + buy_fee
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0

        effective_entry = entry_price + (buy_fee / shares)
        effective_exit = fill_price - (sell_fee / shares)
        is_win = effective_exit > effective_entry

        self._completed_trades.append(
            {
                "entry_date": pos["entry_date"],
                "exit_date": self._row_date(row, idx),
                "entry_price": round(entry_price, 2),
                "exit_price": round(fill_price, 2),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "exit_reason": exit_reason,
                "is_win": is_win,
            }
        )

        self.cash += proceeds
        self.shares_held = 0
        self._open_position = None

    def _determine_exit_reason(
        self, close_price: float, entry_price: float, signal: int
    ) -> Optional[str]:
        if close_price < entry_price * (1 - self.stop_loss_pct):
            return "STOP_LOSS"
        if close_price > entry_price * (1 + self.take_profit_pct):
            return "TAKE_PROFIT"
        if signal == -1:
            return "SIGNAL"
        return None

    def run(self, df: pd.DataFrame, symbol: str = "SYMBOL") -> "Backtester":
        required = {"open", "high", "low", "close", "signal"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        self.symbol = symbol
        self.cash = self.initial_capital
        self.shares_held = 0
        self.equity_curve = []
        self._dates = []
        self._completed_trades = []
        self._open_position = None

        data = df.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date").reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)

        self._df = data
        data["next_open"] = data["open"].shift(-1)

        first_close = float(data["close"].iloc[0])
        last_close = float(data["close"].iloc[-1])
        self._benchmark_return_pct = ((last_close - first_close) / first_close) * 100

        for idx, row in data.iterrows():
            close_price = float(row["close"])
            portfolio_value = self.cash + self.shares_held * close_price
            self.equity_curve.append(portfolio_value)
            if "date" in data.columns:
                self._dates.append(row["date"])

            next_open = row["next_open"]
            if pd.isna(next_open):
                continue

            fill_price = float(next_open)
            signal = int(row["signal"])

            if self.shares_held > 0 and self._open_position is not None:
                entry_price = float(self._open_position["entry_price"])
                exit_reason = self._determine_exit_reason(
                    close_price, entry_price, signal
                )
                if exit_reason:
                    self._execute_sell(fill_price, idx, row, exit_reason)
                    continue

            if signal == 1 and self.shares_held == 0:
                self._execute_buy(fill_price, idx, row)

        return self

    def _years_elapsed(self) -> float:
        if len(self._dates) >= 2:
            delta = self._dates[-1] - self._dates[0]
            return max(delta.days / 365.25, 1 / TRADING_DAYS_PER_YEAR)
        return max(len(self.equity_curve) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)

    def _trade_stats(self) -> dict[str, Any]:
        trades = self._completed_trades
        if not trades:
            return {
                "win_rate": 0.0,
                "losing_trades": 0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "profit_factor": 0.0,
                "stop_loss_exits": 0,
                "take_profit_exits": 0,
                "signal_exits": 0,
            }

        wins = [t for t in trades if t["is_win"]]
        losses = [t for t in trades if not t["is_win"]]
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        avg_win_pct = (
            sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss_pct = (
            sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0.0
        )

        return {
            "win_rate": (len(wins) / len(trades)) * 100,
            "losing_trades": len(losses),
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "profit_factor": profit_factor,
            "stop_loss_exits": sum(
                1 for t in trades if t["exit_reason"] == "STOP_LOSS"
            ),
            "take_profit_exits": sum(
                1 for t in trades if t["exit_reason"] == "TAKE_PROFIT"
            ),
            "signal_exits": sum(1 for t in trades if t["exit_reason"] == "SIGNAL"),
        }

    def results(self) -> dict[str, Any]:
        if not self.equity_curve:
            raise RuntimeError("No backtest results. Call run() first.")

        initial = self.initial_capital
        final = self.equity_curve[-1]
        total_return_pct = ((final - initial) / initial) * 100

        years = self._years_elapsed()
        cagr = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 and initial > 0 else 0.0

        equity = pd.Series(self.equity_curve, dtype=float)
        daily_returns = equity.pct_change().dropna()
        rf_daily = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess = daily_returns - rf_daily
            sharpe_ratio = (excess.mean() / excess.std()) * math.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sharpe_ratio = 0.0

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown_pct = drawdown.min() * 100 if len(drawdown) > 0 else 0.0

        stats = self._trade_stats()
        profit_factor = stats["profit_factor"]
        if profit_factor == float("inf"):
            profit_factor_out = 999.99
        else:
            profit_factor_out = round(profit_factor, 2)

        return {
            "total_return_pct": round(total_return_pct, 2),
            "cagr": round(cagr, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "win_rate": round(stats["win_rate"], 2),
            "total_trades": len(self._completed_trades),
            "equity_curve": self.equity_curve,
            "benchmark_return_pct": round(self._benchmark_return_pct, 2),
            "losing_trades": stats["losing_trades"],
            "avg_win_pct": round(stats["avg_win_pct"], 2),
            "avg_loss_pct": round(stats["avg_loss_pct"], 2),
            "profit_factor": profit_factor_out,
            "stop_loss_exits": stats["stop_loss_exits"],
            "take_profit_exits": stats["take_profit_exits"],
            "signal_exits": stats["signal_exits"],
        }

    @property
    def dates(self) -> list[Any]:
        return self._dates

    @property
    def completed_trades(self) -> list[dict[str, Any]]:
        return self._completed_trades

    @property
    def trades_table(self) -> pd.DataFrame:
        if not self._completed_trades:
            return pd.DataFrame(
                columns=[
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "shares",
                    "pnl",
                    "pnl_pct",
                    "exit_reason",
                ]
            )
        return pd.DataFrame(self._completed_trades)

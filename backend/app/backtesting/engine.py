"""Bar-by-bar backtesting engine with next-bar execution (no lookahead)."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    entry_brokerage: float
    exit_brokerage: float
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    exit_reason: str
    days_held: int
    is_winner: bool


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_holding_days: int = 15
    brokerage_pct: float = 0.0003
    min_brokerage: float = 20.0
    slippage_pct: float = 0.001


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"date", "datetime", "timestamp", "time"}:
            rename[col] = "date"
        elif key in {"open", "high", "low", "close", "volume", "signal"}:
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


class Backtester:
    """Long-only backtester with stop-loss, take-profit, time, and signal exits."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_holding_days: int = 15,
        brokerage_pct: float = 0.0003,
        min_brokerage: float = 20.0,
        slippage_pct: float = 0.001,
    ) -> None:
        self.config = BacktestConfig(
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_holding_days=max_holding_days,
            brokerage_pct=brokerage_pct,
            min_brokerage=min_brokerage,
            slippage_pct=slippage_pct,
        )
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[dict] = []
        self.symbol: str = "STOCK"
        self._first_close: float = 0.0
        self._last_close: float = 0.0
        self._reset_state()

    def _reset_state(self) -> None:
        self.cash = self.config.initial_capital
        self.shares_held = 0
        self.entry_price: Optional[float] = None
        self.entry_date: Optional[str] = None
        self.days_in_trade = 0
        self.trade_id_counter = 0
        self.pending_exit = False
        self.pending_exit_reason: Optional[str] = None
        self.pending_buy = False
        self.entry_brokerage = 0.0

    def _calc_brokerage(self, trade_value: float) -> float:
        return max(self.config.min_brokerage, trade_value * self.config.brokerage_pct)

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        if is_buy:
            return price * (1 + self.config.slippage_pct)
        return price * (1 - self.config.slippage_pct)

    def _execute_buy(self, open_price: float, date: str) -> None:
        fill_price = self._apply_slippage(open_price, is_buy=True)
        shares = int(self.cash * 0.95 / fill_price)
        if shares < 1:
            return

        trade_value = shares * fill_price
        brokerage = self._calc_brokerage(trade_value)
        self.cash -= (shares * fill_price) + brokerage
        self.shares_held = shares
        self.entry_price = fill_price
        self.entry_date = date
        self.days_in_trade = 0
        self.entry_brokerage = brokerage
        print(
            f"  BUY  {date}: {shares} shares @ Rs{fill_price:.2f} | "
            f"Cash left: Rs{self.cash:.0f}"
        )

    def _execute_sell(self, open_price: float, date: str, reason: str) -> None:
        if self.shares_held == 0 or self.entry_price is None:
            return

        fill_price = self._apply_slippage(open_price, is_buy=False)
        trade_value = self.shares_held * fill_price
        brokerage = self._calc_brokerage(trade_value)
        proceeds = trade_value - brokerage
        self.cash += proceeds

        entry_value = self.entry_price * self.shares_held
        gross_pnl = (fill_price - self.entry_price) * self.shares_held
        net_pnl = gross_pnl - self.entry_brokerage - brokerage
        pnl_pct = (net_pnl / entry_value * 100) if entry_value > 0 else 0.0

        self.trade_id_counter += 1
        record = TradeRecord(
            trade_id=self.trade_id_counter,
            symbol=self.symbol,
            entry_date=self.entry_date or date,
            exit_date=date,
            entry_price=round(self.entry_price, 2),
            exit_price=round(fill_price, 2),
            shares=self.shares_held,
            entry_brokerage=round(self.entry_brokerage, 2),
            exit_brokerage=round(brokerage, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=reason,
            days_held=self.days_in_trade,
            is_winner=net_pnl > 0,
        )
        self.trades.append(record)
        print(
            f"  SELL {date}: {record.shares} shares @ Rs{fill_price:.2f} | "
            f"PnL: Rs{net_pnl:.0f} ({pnl_pct:.1f}%) | Reason: {reason}"
        )

        self.shares_held = 0
        self.entry_price = None
        self.entry_date = None
        self.days_in_trade = 0
        self.entry_brokerage = 0.0

    def run(self, df: pd.DataFrame, symbol: str = "STOCK") -> Backtester:
        self._reset_state()
        self.trades = []
        self.equity_curve = []
        self.symbol = symbol

        data = _standardize_columns(df.copy())
        if "date" not in data.columns:
            raise ValueError("DataFrame must contain a date column")

        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)

        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        if "signal" not in data.columns:
            data["signal"] = 0

        self._first_close = float(data["close"].iloc[0])
        self._last_close = float(data["close"].iloc[-1])

        total_bars = len(data)
        for i in range(total_bars):
            current_bar = data.iloc[i]
            current_date = str(current_bar["date"])[:10]
            current_close = float(current_bar["close"])
            current_signal = int(current_bar["signal"])

            if i > 0:
                current_open = float(current_bar["open"])

                if self.pending_exit:
                    self._execute_sell(
                        current_open, current_date, self.pending_exit_reason or "SIGNAL"
                    )
                    self.pending_exit = False
                    self.pending_exit_reason = None
                elif self.pending_buy and self.shares_held == 0:
                    self._execute_buy(current_open, current_date)
                    self.pending_buy = False

            if self.shares_held > 0 and self.entry_price is not None:
                self.days_in_trade += 1

                if current_close <= self.entry_price * (1 - self.config.stop_loss_pct):
                    self.pending_exit = True
                    self.pending_exit_reason = "STOP_LOSS"
                elif current_close >= self.entry_price * (
                    1 + self.config.take_profit_pct
                ):
                    self.pending_exit = True
                    self.pending_exit_reason = "TAKE_PROFIT"
                elif self.days_in_trade >= self.config.max_holding_days:
                    self.pending_exit = True
                    self.pending_exit_reason = "TIME_EXIT"
                elif current_signal == -1:
                    self.pending_exit = True
                    self.pending_exit_reason = "SIGNAL"

            if self.shares_held == 0 and not self.pending_buy:
                if current_signal == 1:
                    self.pending_buy = True

            portfolio_value = self.cash + (self.shares_held * current_close)
            self.equity_curve.append(
                {
                    "date": current_date,
                    "portfolio_value": portfolio_value,
                    "cash": self.cash,
                    "position_value": self.shares_held * current_close,
                    "shares_held": self.shares_held,
                }
            )

        if self.shares_held > 0:
            last_bar = data.iloc[-1]
            last_close = float(last_bar["close"])
            last_date = str(last_bar["date"])[:10]
            self._execute_sell(last_close, last_date, "END_OF_DATA")

        return self

    def results(self) -> dict:
        initial_capital = self.config.initial_capital
        final_portfolio_value = (
            self.equity_curve[-1]["portfolio_value"] if self.equity_curve else initial_capital
        )

        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                "symbol": self.symbol,
                "message": "No trades generated",
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_return_pct": 0.0,
                "cagr": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "avg_days_held": 0.0,
                "signal_exits": 0,
                "stop_loss_exits": 0,
                "take_profit_exits": 0,
                "time_exits": 0,
                "end_of_data_exits": 0,
                "benchmark_return_pct": 0.0,
                "alpha": 0.0,
                "initial_capital": initial_capital,
                "final_capital": round(final_portfolio_value, 2),
                "total_brokerage_paid": 0.0,
                "equity_curve": self.equity_curve,
                "trades": [],
            }

        winning_trades = sum(1 for t in self.trades if t.is_winner)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades * 100

        total_return_pct = (
            (final_portfolio_value - initial_capital) / initial_capital * 100
        )
        trading_days = len(self.equity_curve)
        years = trading_days / 252 if trading_days > 0 else 1 / 252
        if years > 0 and final_portfolio_value > 0 and initial_capital > 0:
            cagr = ((final_portfolio_value / initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0

        equity_series = pd.Series(
            [e["portfolio_value"] for e in self.equity_curve], dtype=float
        )
        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_returns = daily_returns - (0.065 / 252)
            sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * math.sqrt(
                252
            )
        else:
            sharpe_ratio = 0.0
        sharpe_ratio = round(sharpe_ratio, 2)

        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown_pct = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        avg_win_pct = (
            float(np.mean([t.pnl_pct for t in winners])) if winners else 0.0
        )
        avg_loss_pct = (
            float(np.mean([t.pnl_pct for t in losers])) if losers else 0.0
        )
        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = abs(sum(t.net_pnl for t in losers))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf")

        avg_days_held = float(np.mean([t.days_held for t in self.trades]))

        signal_exits = sum(1 for t in self.trades if t.exit_reason == "SIGNAL")
        stop_loss_exits = sum(1 for t in self.trades if t.exit_reason == "STOP_LOSS")
        take_profit_exits = sum(1 for t in self.trades if t.exit_reason == "TAKE_PROFIT")
        time_exits = sum(1 for t in self.trades if t.exit_reason == "TIME_EXIT")
        end_of_data_exits = sum(
            1 for t in self.trades if t.exit_reason == "END_OF_DATA"
        )

        benchmark_return_pct = (
            (self._last_close - self._first_close) / self._first_close * 100
            if self._first_close > 0
            else 0.0
        )
        alpha = total_return_pct - benchmark_return_pct

        total_brokerage_paid = sum(
            t.entry_brokerage + t.exit_brokerage for t in self.trades
        )

        pf_out = profit_factor
        if pf_out == float("inf"):
            pf_out = 999.99
        else:
            pf_out = round(pf_out, 2)

        return {
            "symbol": self.symbol,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 1),
            "total_return_pct": round(total_return_pct, 2),
            "cagr": round(cagr, 2),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "profit_factor": pf_out,
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "avg_days_held": round(avg_days_held, 1),
            "signal_exits": signal_exits,
            "stop_loss_exits": stop_loss_exits,
            "take_profit_exits": take_profit_exits,
            "time_exits": time_exits,
            "end_of_data_exits": end_of_data_exits,
            "benchmark_return_pct": round(benchmark_return_pct, 2),
            "alpha": round(alpha, 2),
            "initial_capital": initial_capital,
            "final_capital": round(final_portfolio_value, 2),
            "total_brokerage_paid": round(total_brokerage_paid, 2),
            "equity_curve": self.equity_curve,
            "trades": [vars(t) for t in self.trades],
        }

    def print_summary(self) -> Backtester:
        r = self.results()
        if r.get("message") == "No trades generated":
            print(f"\nNo trades generated for {r['symbol']}.")
            return self

        first_date = self.equity_curve[0]["date"] if self.equity_curve else "N/A"
        last_date = self.equity_curve[-1]["date"] if self.equity_curve else "N/A"
        pf = r["profit_factor"]
        pf_str = "inf" if pf >= 999 else f"{pf:.2f}"

        print()
        print("=" * 48)
        print(f"BACKTEST RESULTS - {r['symbol']} RSI Strategy")
        print("=" * 48)
        print(f"Period        : {first_date} to {last_date}")
        print(f"Capital       : Rs. {r['initial_capital']:,.0f}")
        print("-" * 48)
        print(f"Total Return  : {r['total_return_pct']:+.2f}%")
        print(f"CAGR          : {r['cagr']:+.2f}%")
        print(f"Sharpe Ratio  : {r['sharpe_ratio']:.2f}")
        print(f"Max Drawdown  : {r['max_drawdown_pct']:.2f}%")
        print("-" * 48)
        print(f"Total Trades  : {r['total_trades']}")
        print(f"Win Rate      : {r['win_rate']:.1f}%")
        print(f"Profit Factor : {pf_str}")
        print(f"Avg Win       : +{r['avg_win_pct']:.1f}%")
        print(f"Avg Loss      : {r['avg_loss_pct']:.1f}%")
        print(f"Avg Hold Days : {r['avg_days_held']:.0f} days")
        print("-" * 48)
        print("Exit Reasons:")
        print(f"  Signal exits     : {r['signal_exits']}")
        print(f"  Stop loss exits  : {r['stop_loss_exits']}")
        print(f"  Take profit exits: {r['take_profit_exits']}")
        print(f"  Time exits       : {r['time_exits']}")
        print(f"  End of data      : {r['end_of_data_exits']}")
        print("-" * 48)
        print(f"Benchmark (B&H) : {r['benchmark_return_pct']:+.2f}%")
        print(f"Alpha           : {r['alpha']:+.2f}%")
        print(f"Final Capital   : Rs. {r['final_capital']:,.0f}")
        print(f"Brokerage paid  : Rs. {r['total_brokerage_paid']:,.0f}")
        print("=" * 48)
        print()
        return self

    @property
    def dates(self) -> list[str]:
        return [e["date"] for e in self.equity_curve]

    @property
    def trades_table(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "shares",
                    "net_pnl",
                    "pnl_pct",
                    "exit_reason",
                ]
            )
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "pnl": t.net_pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                }
            )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    from backend.app.strategies.rsi_strategy import RSIStrategy

    project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    csv_path = os.path.join(project_root, "data", "raw", "RELIANCE.csv")
    df = pd.read_csv(csv_path)
    strategy = RSIStrategy(oversold=45, overbought=60)
    df = strategy.generate_signals(df)

    backtester = Backtester(
        initial_capital=100_000,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_holding_days=15,
    )
    backtester.run(df, "RELIANCE").print_summary()

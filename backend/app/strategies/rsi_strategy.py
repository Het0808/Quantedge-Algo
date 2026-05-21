"""RSI mean-reversion strategy with optional NIFTY50 regime filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema, rsi


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names to lowercase standard names."""
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"date", "datetime", "timestamp", "time"}:
            rename[col] = "date"
        elif key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key
        elif key in {"close price", "adj close", "adj_close"} or key.endswith("close"):
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


class RSIStrategy:
    """RSI crossover strategy with NIFTY50 uptrend regime filter on entries."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 45,
        overbought: float = 60,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_holding_days: int = 15,
        nifty_df: pd.DataFrame | None = None,
    ) -> None:
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self.nifty_df = nifty_df

    def _prepare_nifty_regime(self) -> pd.Series | None:
        if self.nifty_df is None:
            return None

        nifty = _standardize_columns(self.nifty_df.copy())
        if "close" not in nifty.columns:
            raise ValueError("nifty_df must contain a close column")

        nifty = ema(nifty, period=50, column="close")
        nifty["regime"] = np.where(nifty["close"] > nifty["ema_50"], 1, 0)
        nifty["date"] = pd.to_datetime(nifty["date"])
        nifty = nifty.sort_values("date").set_index("date")
        return nifty["regime"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _standardize_columns(df.copy())
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df = rsi(df, period=self.rsi_period)
        df = ema(df, period=50, column="close")

        regime_series = self._prepare_nifty_regime()
        if regime_series is not None:
            regime_df = regime_series.reset_index()
            regime_df.columns = ["date", "regime"]
            df = df.merge(regime_df, on="date", how="left")
            df["regime"] = df["regime"].fillna(0).astype(int)
        else:
            df["regime"] = 1

        rsi_prev = df["rsi"].shift(1)
        rsi_curr = df["rsi"]
        buy_crossover = (rsi_prev >= self.oversold) & (rsi_curr < self.oversold)
        sell_crossover = (rsi_prev <= self.overbought) & (rsi_curr > self.overbought)

        df["rsi_prev"] = rsi_prev
        df["buy_crossover"] = buy_crossover
        df["regime_filter_blocked"] = buy_crossover & (df["regime"] == 0)

        df["signal"] = 0
        df.loc[buy_crossover & (df["regime"] == 1), "signal"] = 1
        df.loc[sell_crossover, "signal"] = -1

        return df

    def get_signal_stats(self, df: pd.DataFrame) -> dict:
        if "buy_crossover" not in df.columns:
            df = self.generate_signals(df)

        total_buy_crossovers = int(df["buy_crossover"].sum())
        regime_blocked = int(df["regime_filter_blocked"].sum())
        signals_generated = int((df["signal"] == 1).sum())
        sell_signals = int((df["signal"] == -1).sum())

        return {
            "total_buy_crossovers": total_buy_crossovers,
            "regime_blocked": regime_blocked,
            "signals_generated": signals_generated,
            "sell_signals": sell_signals,
        }

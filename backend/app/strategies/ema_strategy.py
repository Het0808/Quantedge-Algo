"""EMA crossover strategy with volume and NIFTY50 regime filters."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    Uses pandas ewm with adjust=False which matches
    TradingView and most professional platforms.
    span=period, min_periods=period to avoid
    partial EMA values at the start.
    Returns pd.Series of same length as input,
    NaN for first (period-1) values.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_nifty_regime(
    nifty_df: pd.DataFrame, ema_period: int = 50
) -> pd.Series:
    """
    Calculate NIFTY50 regime filter.
    Returns a Series indexed by date string 'YYYY-MM-DD'.
    Value = 1 when NIFTY50 close > EMA50 (bull regime).
    Value = 0 when NIFTY50 close <= EMA50 (bear regime).
    """
    df = nifty_df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")

    df = df.sort_index()
    df["nifty_ema"] = calculate_ema(df["close"], ema_period)
    regime = (df["close"] > df["nifty_ema"]).astype(int)
    regime.name = "regime"
    return regime


class EMAStrategy:
    """
    EMA Crossover Strategy for Indian Stock Markets.

    Logic:
    - BUY  when fast EMA crosses ABOVE slow EMA
    - SELL when fast EMA crosses BELOW slow EMA

    Optional filters:
    - Volume filter: only buy when volume > 20-day avg
    - NIFTY50 regime: only buy in bull market

    No lookahead bias: all comparisons use shift(1)
    for previous bar values.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        volume_filter: bool = True,
        volume_lookback: int = 20,
        nifty_df: Optional[pd.DataFrame] = None,
        nifty_ema_period: int = 50,
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be "
                f"less than slow_period ({slow_period})"
            )
        if fast_period < 2:
            raise ValueError("fast_period must be >= 2")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_filter = volume_filter
        self.volume_lookback = volume_lookback
        self.nifty_df = nifty_df
        self.nifty_ema_period = nifty_ema_period

        self._regime_series: pd.Series | None = None
        if nifty_df is not None:
            try:
                self._regime_series = calculate_nifty_regime(
                    nifty_df, nifty_ema_period
                )
                print(
                    f"  EMAStrategy: NIFTY50 regime filter loaded "
                    f"({len(self._regime_series)} days)"
                )
            except Exception as exc:
                print(f"  EMAStrategy WARNING: Could not load regime filter: {exc}")
                self._regime_series = None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals on OHLCV DataFrame.

        Returns df with added columns:
        - ema_fast: fast EMA values
        - ema_slow: slow EMA values
        - ema_diff: fast - slow (positive = bullish)
        - volume_ma: 20-day volume moving average
        - signal: 1=buy, -1=sell, 0=hold
        - raw_buy_crossover: bool, crossover before filters
        - volume_blocked: bool, blocked by volume filter
        - regime_blocked: bool, blocked by regime filter
        """
        result = df.copy()
        result.columns = [str(c).lower().strip() for c in result.columns]

        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
        result = result.sort_values("date").reset_index(drop=True)

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in result.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        result["ema_fast"] = calculate_ema(result["close"], self.fast_period)
        result["ema_slow"] = calculate_ema(result["close"], self.slow_period)
        result["ema_diff"] = result["ema_fast"] - result["ema_slow"]

        result["volume_ma"] = (
            result["volume"]
            .rolling(window=self.volume_lookback, min_periods=1)
            .mean()
        )

        prev_fast = result["ema_fast"].shift(1)
        prev_slow = result["ema_slow"].shift(1)
        curr_fast = result["ema_fast"]
        curr_slow = result["ema_slow"]

        result["raw_buy_crossover"] = (
            (prev_fast <= prev_slow)
            & (curr_fast > curr_slow)
            & curr_fast.notna()
            & curr_slow.notna()
            & prev_fast.notna()
            & prev_slow.notna()
        )

        result["raw_sell_crossover"] = (
            (prev_fast >= prev_slow)
            & (curr_fast < curr_slow)
            & curr_fast.notna()
            & curr_slow.notna()
        )

        result["volume_blocked"] = False
        if self.volume_filter:
            volume_ok = result["volume"] > result["volume_ma"]
            result["volume_blocked"] = result["raw_buy_crossover"] & ~volume_ok

        result["regime_blocked"] = False
        result["regime"] = 1

        if self._regime_series is not None:
            result["regime"] = (
                result["date"].map(self._regime_series).fillna(0).astype(int)
            )
            result["regime_blocked"] = (
                result["raw_buy_crossover"]
                & ~result["volume_blocked"]
                & (result["regime"] == 0)
            )

        result["signal"] = 0

        buy_mask = (
            result["raw_buy_crossover"]
            & ~result["volume_blocked"]
            & ~result["regime_blocked"]
        )
        sell_mask = result["raw_sell_crossover"]

        result.loc[buy_mask, "signal"] = 1
        result.loc[sell_mask, "signal"] = -1

        both_mask = buy_mask & sell_mask
        result.loc[both_mask, "signal"] = -1

        return result

    def get_signal_stats(self, df: pd.DataFrame) -> dict:
        """Return signal statistics after generate_signals()."""
        if "signal" not in df.columns:
            df = self.generate_signals(df)

        raw_buys = (
            int(df["raw_buy_crossover"].sum())
            if "raw_buy_crossover" in df.columns
            else 0
        )
        vol_blocked = (
            int(df["volume_blocked"].sum()) if "volume_blocked" in df.columns else 0
        )
        regime_blocked = (
            int(df["regime_blocked"].sum()) if "regime_blocked" in df.columns else 0
        )
        final_buys = int((df["signal"] == 1).sum())
        final_sells = int((df["signal"] == -1).sum())

        return {
            "total_buy_crossovers": raw_buys,
            "volume_blocked": vol_blocked,
            "regime_blocked": regime_blocked,
            "signals_generated": final_buys,
            "sell_signals": final_sells,
            "total_buy_signals": final_buys,
        }

    def get_params(self) -> dict:
        """Return strategy parameters as dict."""
        return {
            "strategy": "EMA Crossover",
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "volume_filter": self.volume_filter,
            "nifty_filter": self._regime_series is not None,
        }

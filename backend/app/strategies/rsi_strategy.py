"""RSI mean-reversion strategy: buy oversold, sell overbought."""

import pandas as pd

from backend.app.strategies.indicators import rsi


class RSIStrategy:
    """Generate buy/sell signals from RSI threshold crossovers."""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.rsi_col = f"rsi_{period}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Buy when RSI crosses below oversold (30).
        Sell when RSI crosses above overbought (70).

        Adds signal column: 1=buy, -1=sell, 0=hold.
        """
        result = rsi(df, period=self.period)
        rsi_series = result[self.rsi_col]
        prev_rsi = rsi_series.shift(1)

        buy = (prev_rsi >= self.oversold) & (rsi_series < self.oversold)
        sell = (prev_rsi <= self.overbought) & (rsi_series > self.overbought)

        result["signal"] = 0
        result.loc[buy, "signal"] = 1
        result.loc[sell, "signal"] = -1
        return result

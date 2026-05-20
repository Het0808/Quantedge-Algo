"""RSI mean-reversion strategy."""

import pandas as pd

from backend.app.strategies.indicators import rsi


class RSIStrategy:
    """Generate buy/sell signals from RSI threshold crossovers."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
    ) -> None:
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Buy when RSI crosses below oversold (was above, now below).
        Sell when RSI crosses above overbought (was above threshold, now above).

        Adds 'signal' column: 1=buy, -1=sell, 0=hold. No lookahead bias.
        """
        result = rsi(df, period=self.rsi_period)
        rsi_series = result["rsi"]
        prev_rsi = rsi_series.shift(1)

        buy = (prev_rsi >= self.oversold) & (rsi_series < self.oversold)
        sell = (prev_rsi <= self.overbought) & (rsi_series > self.overbought)

        result["signal"] = 0
        result.loc[buy, "signal"] = 1
        result.loc[sell, "signal"] = -1
        return result

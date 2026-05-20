"""EMA crossover strategy: buy on golden cross, sell on death cross."""

import pandas as pd

from backend.app.strategies.indicators import ema


class EMAStrategy:
    """Generate buy/sell signals from fast/slow EMA crossovers."""

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_col = f"ema_{fast_period}"
        self.slow_col = f"ema_{slow_period}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Buy when fast EMA crosses above slow EMA.
        Sell when fast EMA crosses below slow EMA.

        Adds signal column: 1=buy, -1=sell, 0=hold.
        """
        result = ema(df, self.fast_period)
        result = ema(result, self.slow_period)

        fast_ema = result[self.fast_col]
        slow_ema = result[self.slow_col]
        prev_fast = fast_ema.shift(1)
        prev_slow = slow_ema.shift(1)

        buy = (prev_fast <= prev_slow) & (fast_ema > slow_ema)
        sell = (prev_fast >= prev_slow) & (fast_ema < slow_ema)

        result["signal"] = 0
        result.loc[buy, "signal"] = 1
        result.loc[sell, "signal"] = -1
        return result

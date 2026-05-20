"""Technical indicators for OHLCV DataFrames."""

import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index using Wilder's smoothing.

    Adds column 'rsi' and returns the DataFrame.
    """
    result = df.copy()
    delta = result["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    result["rsi"] = 100 - (100 / (1 + rs))
    return result


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    """Exponential moving average; adds column ema_{period}."""
    result = df.copy()
    result[f"ema_{period}"] = result[column].ewm(span=period, adjust=False).mean()
    return result


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    result = df.copy()
    ema_fast = result["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow, adjust=False).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    return result


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger upper, middle, and lower bands."""
    result = df.copy()
    middle = result["close"].rolling(window=period).mean()
    std = result["close"].rolling(window=period).std()
    result["bb_middle"] = middle
    result["bb_upper"] = middle + num_std * std
    result["bb_lower"] = middle - num_std * std
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range as atr_{period}."""
    result = df.copy()
    prev_close = result["close"].shift(1)
    tr = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - prev_close).abs(),
            (result["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result[f"atr_{period}"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return result


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative volume-weighted average price."""
    result = df.copy()
    typical_price = (result["high"] + result["low"] + result["close"]) / 3
    cumulative_tp_vol = (typical_price * result["volume"]).cumsum()
    cumulative_vol = result["volume"].cumsum()
    result["vwap"] = cumulative_tp_vol / cumulative_vol
    return result

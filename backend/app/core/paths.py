from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data"]
PAPER_TRADES_FILE = PROJECT_ROOT / "paper_trades.json"

KNOWN_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
VALID_STRATEGIES = {"rsi", "ema"}

# Nifty 50 top constituents by index weight
NIFTY_TOP_5 = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]

QUOTE_KEY_PREFIX = "quote:"
SIGNALS_LATEST_KEY = "signals:latest"


def quote_key(symbol: str) -> str:
    return f"{QUOTE_KEY_PREFIX}{symbol.upper()}"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data"]
PAPER_TRADES_FILE = PROJECT_ROOT / "paper_trades.json"

KNOWN_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
VALID_STRATEGIES = {"rsi", "ema"}

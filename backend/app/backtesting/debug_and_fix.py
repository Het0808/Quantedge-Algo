"""
Diagnose and fix the data + RSI strategy pipeline.

Run from project root:
    python backend/app/backtesting/debug_and_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "strategy_comparison.csv"

SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
TRADING_DAYS = 756
MIN_TRADES_PER_SYMBOL = 15

STARTING_PRICES = {
    "RELIANCE": 2400.0,
    "TCS": 3200.0,
    "INFY": 1400.0,
    "HDFCBANK": 1600.0,
    "ICICIBANK": 900.0,
}

_SYMBOL_SEEDS = {
    "RELIANCE": 42,
    "TCS": 101,
    "INFY": 202,
    "HDFCBANK": 303,
    "ICICIBANK": 404,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.backtesting.engine import Backtester
from backend.app.strategies.indicators import rsi
from backend.app.strategies.rsi_strategy import RSIStrategy

# ---------------------------------------------------------------------------
# State collected across steps
# ---------------------------------------------------------------------------
diagnosis_state: dict = {
    "files_missing": [],
    "files_regenerated": [],
    "column_fix_applied": False,
    "recommended_oversold": 40,
    "recommended_overbought": 60,
    "use_trend_filter": False,
    "symbol_data": {},
    "backtest_results": [],
}


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# STEP 1 — Data diagnosis
# ---------------------------------------------------------------------------


def step1_data_diagnosis() -> None:
    _banner("STEP 1 — DATA DIAGNOSIS")
    for symbol in SYMBOLS:
        path = DATA_DIR / f"{symbol}.csv"
        print(f"\n--- {symbol} ---")
        if not path.exists():
            print("MISSING — will generate synthetic data")
            diagnosis_state["files_missing"].append(symbol)
            continue
        df = pd.read_csv(path)
        print(f"File: {path}")
        print(f"Row count: {len(df)}")
        print(f"Column names: {list(df.columns)}")
        print("First 3 rows:")
        print(df.head(3).to_string(index=False))
        print("Last 3 rows:")
        print(df.tail(3).to_string(index=False))
        close_col = _find_close_column(df)
        if close_col:
            print(f"Close min/max: {df[close_col].min()} / {df[close_col].max()}")
        print("NaN counts per column:")
        print(df.isna().sum().to_string())


def _find_close_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"close", "close price"} or key.endswith("close"):
            return col
    return None


# ---------------------------------------------------------------------------
# STEP 2 — Fix column names
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def _normalize_column_name(col: str) -> str | None:
    key = str(col).strip().lower()
    if key in _COLUMN_ALIASES:
        return _COLUMN_ALIASES[key]
    if key in {"close price", "adj close", "adj_close", "4. close"}:
        return "close"
    if key.endswith(". close") or key.endswith(" close"):
        return "close"
    if "open" in key and "close" not in key:
        return "open"
    if "high" in key:
        return "high"
    if "low" in key:
        return "low"
    if "volume" in key or key == "vol":
        return "volume"
    if key in {"timestamp", "datetime", "time"}:
        return "date"
    return None


def standardize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Rename columns, parse dates, sort, drop bad rows."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        target = _normalize_column_name(col)
        if target and target not in rename_map.values():
            rename_map[col] = target

    out = df.rename(columns=rename_map).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{symbol}: cannot standardize, missing {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    out = out.sort_values("date").reset_index(drop=True)
    return out[required]


def step2_fix_columns() -> None:
    _banner("STEP 2 — FIX COLUMN NAMES")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    any_fix = False

    for symbol in SYMBOLS:
        path = DATA_DIR / f"{symbol}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        before_cols = list(raw.columns)
        try:
            fixed = standardize_ohlcv(raw, symbol)
        except ValueError as exc:
            print(f"[ERROR] {symbol}: {exc}")
            continue

        if before_cols != list(fixed.columns) or len(raw) != len(fixed):
            any_fix = True
        fixed.to_csv(path, index=False)
        diagnosis_state["symbol_data"][symbol] = fixed
        print(f"Fixed columns for {symbol}: now has columns {list(fixed.columns)} "
              f"({len(fixed)} rows)")

    diagnosis_state["column_fix_applied"] = any_fix or bool(diagnosis_state["symbol_data"])


# ---------------------------------------------------------------------------
# STEP 3 — Generate synthetic data
# ---------------------------------------------------------------------------


def generate_synthetic_ohlcv(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=TRADING_DAYS)
    seed = _SYMBOL_SEEDS.get(symbol, hash(symbol) % 10_000)
    rng = np.random.default_rng(seed)
    start_price = STARTING_PRICES.get(symbol, 1000.0)

    daily_returns = rng.normal(loc=0.0003, scale=0.015, size=TRADING_DAYS)
    close = start_price * np.cumprod(1 + daily_returns)

    rows: list[dict] = []
    prev_close = close[0]
    for i, dt in enumerate(dates):
        c = float(close[i])
        h = c * (1 + abs(rng.random() * 0.008))
        l = c * (1 - abs(rng.random() * 0.008))
        o = prev_close * (1 + rng.random() * 0.005 - 0.0025)
        vol = int(rng.integers(1_000_000, 5_000_000))
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "open": round(o, 2),
                "high": round(max(h, o, c), 2),
                "low": round(min(l, o, c), 2),
                "close": round(c, 2),
                "volume": vol,
            }
        )
        prev_close = c

    return pd.DataFrame(rows)


def step3_generate_missing() -> None:
    _banner("STEP 3 — GENERATE SYNTHETIC DATA (if missing)")
    for symbol in SYMBOLS:
        path = DATA_DIR / f"{symbol}.csv"
        if path.exists() and symbol in diagnosis_state["symbol_data"]:
            continue
        if path.exists():
            df = standardize_ohlcv(pd.read_csv(path), symbol)
            df.to_csv(path, index=False)
            diagnosis_state["symbol_data"][symbol] = df
            continue

        df = generate_synthetic_ohlcv(symbol)
        df.to_csv(path, index=False)
        diagnosis_state["symbol_data"][symbol] = df
        diagnosis_state["files_regenerated"].append(symbol)
        print(f"Generated synthetic data for {symbol}: {len(df)} rows")


def _load_symbol_df(symbol: str) -> pd.DataFrame:
    if symbol in diagnosis_state["symbol_data"]:
        return diagnosis_state["symbol_data"][symbol].copy()
    path = DATA_DIR / f"{symbol}.csv"
    df = standardize_ohlcv(pd.read_csv(path), symbol)
    diagnosis_state["symbol_data"][symbol] = df
    return df.copy()


# ---------------------------------------------------------------------------
# STEP 4 — RSI calculation test
# ---------------------------------------------------------------------------


def step4_rsi_test() -> pd.DataFrame:
    _banner("STEP 4 — TEST RSI CALCULATION (RELIANCE)")
    df = _load_symbol_df("RELIANCE")
    df_rsi = rsi(df, period=14)
    series = df_rsi["rsi"].dropna()

    stats = {
        "min": series.min(),
        "max": series.max(),
        "below_30": int((series < 30).sum()),
        "below_40": int((series < 40).sum()),
        "below_45": int((series < 45).sum()),
        "above_60": int((series > 60).sum()),
        "above_70": int((series > 70).sum()),
    }

    print(f"Min RSI: {stats['min']:.2f}")
    print(f"Max RSI: {stats['max']:.2f}")
    print(f"Days RSI < 30: {stats['below_30']}")
    print(f"Days RSI < 40: {stats['below_40']}")
    print(f"Days RSI < 45: {stats['below_45']}")
    print(f"Days RSI > 60: {stats['above_60']}")
    print(f"Days RSI > 70: {stats['above_70']}")

    # Crossover signal counts (what backtester actually uses)
    for tf in (True, False):
        strat = RSIStrategy(oversold=40, overbought=60, trend_filter=tf)
        sig = strat.generate_signals(df)
        buys = int((sig["signal"] == 1).sum())
        sells = int((sig["signal"] == -1).sum())
        print(f"Crossover signals (OS=40/OB=60, trend={tf}): {buys} buys, {sells} sells")

    return df


# ---------------------------------------------------------------------------
# STEP 5 — Auto-select thresholds
# ---------------------------------------------------------------------------


def _recommend_thresholds_for_symbol(df: pd.DataFrame) -> tuple[float, float]:
    series = rsi(df, period=14)["rsi"].dropna()
    below_30 = int((series < 30).sum())
    below_40 = int((series < 40).sum())
    below_45 = int((series < 45).sum())

    if below_30 < 5:
        oversold = 40.0
        reason = "days_below_30 < 5, using oversold=40"
    elif below_40 < 10:
        oversold = 45.0
        reason = "days_below_40 < 10, using oversold=45"
    elif below_45 > 10:
        oversold = 45.0
        reason = "days_below_45 > 10, oversold=45 is workable"
    else:
        oversold = 45.0
        reason = "defaulting to oversold=45 for more crossovers"

    overbought = 60.0 if int((series > 60).sum()) > 20 else 55.0
    print(f"  -> oversold={oversold}, overbought={overbought} ({reason})")
    return oversold, overbought


def step5_auto_thresholds(reliance_df: pd.DataFrame) -> None:
    _banner("STEP 5 — AUTO-SELECT BEST THRESHOLDS")
    series = rsi(reliance_df, period=14)["rsi"].dropna()
    below_30 = int((series < 30).sum())
    below_40 = int((series < 40).sum())
    below_45 = int((series < 45).sum())

    print("RELIANCE RSI distribution drives global defaults:")
    if below_30 < 5:
        print("  days_below_30 < 5: oversold=30 too strict -> recommend oversold=40")
        diagnosis_state["recommended_oversold"] = 40
    elif below_40 < 10:
        print("  days_below_40 < 10: still strict -> recommend oversold=45")
        diagnosis_state["recommended_oversold"] = 45
    elif below_45 > 10:
        print("  days_below_45 > 10: oversold=45 is workable")
        diagnosis_state["recommended_oversold"] = 45
    else:
        diagnosis_state["recommended_oversold"] = 45

    diagnosis_state["recommended_overbought"] = 60

    print("\nPer-symbol threshold check:")
    symbol_thresholds: dict[str, tuple[float, float]] = {}
    for symbol in SYMBOLS:
        print(f"{symbol}:")
        df = _load_symbol_df(symbol)
        symbol_thresholds[symbol] = _recommend_thresholds_for_symbol(df)

    diagnosis_state["symbol_thresholds"] = symbol_thresholds

    # Trend filter kills buys when RSI dips but price is below EMA50
    test_os = diagnosis_state["recommended_oversold"]
    test_ob = diagnosis_state["recommended_overbought"]
    strat_trend = RSIStrategy(
        oversold=test_os, overbought=test_ob, trend_filter=True
    )
    strat_no = RSIStrategy(
        oversold=test_os, overbought=test_ob, trend_filter=False
    )
    buys_trend = 0
    buys_no = 0
    for symbol in SYMBOLS:
        df = _load_symbol_df(symbol)
        buys_trend += int((strat_trend.generate_signals(df)["signal"] == 1).sum())
        buys_no += int((strat_no.generate_signals(df)["signal"] == 1).sum())

    diagnosis_state["use_trend_filter"] = buys_trend >= MIN_TRADES_PER_SYMBOL * len(
        SYMBOLS
    )
    print(
        f"\nTrend filter analysis: {buys_trend} total buy crossovers WITH trend, "
        f"{buys_no} WITHOUT trend"
    )
    if not diagnosis_state["use_trend_filter"]:
        print(
            "  -> Disabling trend_filter for backtest (EMA50 blocked almost all entries)"
        )


# ---------------------------------------------------------------------------
# STEP 6 — Run fixed backtest
# ---------------------------------------------------------------------------


def _count_signals(df: pd.DataFrame, oversold: float, overbought: float, trend: bool) -> int:
    strat = RSIStrategy(
        oversold=oversold, overbought=overbought, trend_filter=trend
    )
    sig = strat.generate_signals(df)
    return int((sig["signal"] == 1).sum())


def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    oversold: float,
    overbought: float,
    trend_filter: bool,
) -> dict:
    strategy = RSIStrategy(
        oversold=oversold,
        overbought=overbought,
        trend_filter=trend_filter,
    )
    df_signals = strategy.generate_signals(df)
    buy_signals = int((df_signals["signal"] == 1).sum())

    backtester = Backtester(
        initial_capital=100_000,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    backtester.run(df_signals, symbol=symbol)
    metrics = backtester.results()
    metrics["symbol"] = symbol
    metrics["buy_signals"] = buy_signals
    return metrics


def step6_run_backtest() -> None:
    _banner("STEP 6 — RUN FIXED BACKTEST")
    oversold = diagnosis_state["recommended_oversold"]
    overbought = diagnosis_state["recommended_overbought"]
    trend = diagnosis_state["use_trend_filter"]
    thresholds = diagnosis_state.get("symbol_thresholds", {})

    print(
        f"Settings: oversold={oversold}, overbought={overbought}, "
        f"trend_filter={trend}"
    )

    results: list[dict] = []
    for symbol in SYMBOLS:
        df = _load_symbol_df(symbol)
        os, ob = thresholds.get(symbol, (oversold, overbought))
        metrics = backtest_symbol(symbol, df, os, ob, trend)
        trades = metrics["total_trades"]
        print(
            f"  {symbol}: {trades} round-trip trades, "
            f"{metrics['buy_signals']} buy signals"
        )
        if trades < MIN_TRADES_PER_SYMBOL and not trend:
            print(f"    [WARN] Still below {MIN_TRADES_PER_SYMBOL} trades — data may be short")
        results.append(metrics)

    diagnosis_state["backtest_results"] = results

    _print_comparison_table(results)


def _fmt_pf(value: float) -> str:
    return "inf" if value >= 999 else f"{value:.2f}"


def _print_comparison_table(rows: list[dict]) -> None:
    width = 69
    print(f"\n{'=' * width}")
    print(
        "STRATEGY COMPARISON — RSI Mean Reversion "
        f"({diagnosis_state['recommended_oversold']:.0f}/"
        f"{diagnosis_state['recommended_overbought']:.0f}"
        f"{' + Trend' if diagnosis_state['use_trend_filter'] else ''})"
    )
    print("=" * width)
    print(
        f"{'Symbol':<12}{'CAGR':>8}{'Sharpe':>9}{'MaxDD':>9}"
        f"{'WinRate':>10}{'Trades':>9}{'ProfitFactor':>14}"
    )
    print("-" * width)
    for m in rows:
        print(
            f"{m['symbol']:<12}"
            f"{m['cagr']:>7.1f}%"
            f"{m['sharpe_ratio']:>9.2f}"
            f"{m['max_drawdown_pct']:>8.1f}%"
            f"{m['win_rate']:>9.1f}%"
            f"{m['total_trades']:>9}"
            f"{_fmt_pf(m['profit_factor']):>14}"
        )
    print("-" * width)
    if rows:
        n = len(rows)
        print(
            f"{'AVERAGE':<12}"
            f"{sum(m['cagr'] for m in rows) / n:>7.1f}%"
            f"{sum(m['sharpe_ratio'] for m in rows) / n:>9.2f}"
            f"{sum(m['max_drawdown_pct'] for m in rows) / n:>8.1f}%"
            f"{sum(m['win_rate'] for m in rows) / n:>9.1f}%"
            f"{sum(m['total_trades'] for m in rows) / n:>9.1f}"
        )
    print("=" * width)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# STEP 7 — Final diagnosis report
# ---------------------------------------------------------------------------


def step7_final_report() -> None:
    _banner("STEP 7 — FINAL DIAGNOSIS REPORT")
    missing = diagnosis_state["files_missing"]
    regen = diagnosis_state["files_regenerated"]
    results = diagnosis_state["backtest_results"]

    if missing and regen:
        data_status = f"{len(regen)} file(s) were missing and regenerated"
    elif missing:
        data_status = f"{len(missing)} missing (not all regenerated)"
    else:
        data_status = "all files present"

    total_trades = sum(m["total_trades"] for m in results)
    avg_trades = total_trades / len(results) if results else 0
    min_trades = min((m["total_trades"] for m in results), default=0)

    print("=== DIAGNOSIS COMPLETE ===")
    print(f"Data status: {data_status}")
    print(
        f"Column fix applied: "
        f"{'yes' if diagnosis_state['column_fix_applied'] else 'no'}"
    )
    print(
        f"Recommended RSI thresholds: "
        f"oversold={diagnosis_state['recommended_oversold']} "
        f"overbought={diagnosis_state['recommended_overbought']}"
    )
    print(f"Trend filter enabled: {diagnosis_state['use_trend_filter']}")
    print(f"Total trades generated: {total_trades} across {len(results)} symbols")
    print(f"Avg trades per symbol: {avg_trades:.1f}")
    print(f"Min trades per symbol: {min_trades}")

    if avg_trades >= MIN_TRADES_PER_SYMBOL and total_trades > 0:
        status = "READY FOR NEXT STEP"
        print(f"Status: {status}")
    else:
        reasons = []
        if diagnosis_state["use_trend_filter"]:
            reasons.append("trend_filter still blocking entries")
        if avg_trades < MIN_TRADES_PER_SYMBOL:
            reasons.append(
                f"avg trades ({avg_trades:.1f}) < target {MIN_TRADES_PER_SYMBOL}"
            )
        if not reasons:
            reasons.append("insufficient crossover signals in price data")
        status = f"STILL BROKEN ({'; '.join(reasons)})"
        print(f"Status: {status}")

    print("=" * 26)


def main() -> None:
    try:
        step1_data_diagnosis()
        step2_fix_columns()
        step3_generate_missing()
        reliance_df = step4_rsi_test()
        step5_auto_thresholds(reliance_df)
        step6_run_backtest()
        step7_final_report()
    except Exception as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

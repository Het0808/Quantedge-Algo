"""
QuantEdge Streamlit dashboard.

Run from project root:
    streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data"]
SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
STRATEGIES = {
    "RSI Mean Reversion": "rsi",
    "EMA Crossover": "ema",
}
INITIAL_CAPITAL = 100_000

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.backtesting.engine import Backtester
from backend.app.strategies.ema_strategy import EMAStrategy
from backend.app.strategies.indicators import rsi
from backend.app.strategies.rsi_strategy import RSIStrategy


@st.cache_data
def load_symbol_data(symbol: str) -> pd.DataFrame:
    for data_dir in DATA_DIRS:
        path = data_dir / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"{path.name} missing columns: {missing}")
            return df.sort_values("date").reset_index(drop=True)
    raise FileNotFoundError(
        f"No CSV found for {symbol}. Expected {SYMBOLS[0]}.csv in data/raw/ or data/. "
        "Run: python backend/app/data/kite_collector.py"
    )


def get_strategy(strategy_key: str):
    if strategy_key == "rsi":
        return RSIStrategy()
    if strategy_key == "ema":
        return EMAStrategy()
    raise ValueError(f"Unknown strategy: {strategy_key}")


def filter_by_date_range(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    filtered = df.loc[mask].reset_index(drop=True)
    if len(filtered) < 2:
        raise ValueError("Selected date range must contain at least 2 trading days.")
    return filtered


def compute_buy_and_hold(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    first_close = float(df["close"].iloc[0])
    shares = initial_capital / first_close
    return shares * df["close"]


def run_backtest(
    df: pd.DataFrame, symbol: str, strategy_key: str
) -> tuple[dict, pd.DataFrame, Backtester]:
    strategy = get_strategy(strategy_key)
    df_signals = strategy.generate_signals(df)
    backtester = Backtester(initial_capital=INITIAL_CAPITAL)
    backtester.run(df_signals, symbol=symbol)
    metrics = backtester.results()
    return metrics, df_signals, backtester


def plot_equity_curve(
    dates: list, strategy_equity: list[float], buy_hold: pd.Series
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strategy_equity,
            mode="lines",
            name="Strategy",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=buy_hold.tolist(),
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#94a3b8", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Portfolio Value vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Value (₹)",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_signals_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    df_rsi = rsi(df.copy(), period=14)
    rsi_col = "rsi_14"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} — Last 60 Days", "RSI (14)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["date"],
                y=buys["low"] * 0.98,
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=14, color="#16a34a"),
            ),
            row=1,
            col=1,
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["date"],
                y=sells["high"] * 1.02,
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=14, color="#dc2626"),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df_rsi["date"],
            y=df_rsi[rsi_col],
            mode="lines",
            name="RSI",
            line=dict(color="#7c3aed", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16a34a", row=2, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    return fig


def page_backtest() -> None:
    st.header("Backtest")
    st.caption("Run historical simulations on NSE daily data from CSV files.")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Symbol", SYMBOLS, key="bt_symbol")
    with col2:
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), key="bt_strategy")
    with col3:
        st.write("")

    try:
        df_full = load_symbol_data(symbol)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="bt_dates",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    if st.button("Run Backtest", type="primary", key="bt_run"):
        try:
            df = filter_by_date_range(
                df_full,
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
            )
            strategy_key = STRATEGIES[strategy_name]
            metrics, df_bt, _ = run_backtest(df, symbol, strategy_key)

            st.session_state["bt_metrics"] = metrics
            st.session_state["bt_df"] = df_bt
            st.session_state["bt_buy_hold"] = compute_buy_and_hold(df, INITIAL_CAPITAL)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

    if "bt_metrics" not in st.session_state:
        st.info("Select options and click **Run Backtest** to see results.")
        return

    metrics = st.session_state["bt_metrics"]
    df_bt = st.session_state["bt_df"]
    buy_hold = st.session_state["bt_buy_hold"]
    dates = metrics.get("dates") or df_bt["date"].tolist()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
    m2.metric("CAGR", f"{metrics['cagr']:.2f}%")
    m3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    m4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    m5.metric("Win Rate", f"{metrics['win_rate']:.2f}%")

    st.plotly_chart(
        plot_equity_curve(dates, metrics["equity_curve"], buy_hold),
        use_container_width=True,
    )

    trades_df = metrics.get("trades_table", pd.DataFrame())
    if trades_df is not None and not trades_df.empty:
        display_trades = trades_df.copy()
        if "date" in display_trades.columns:
            display_trades["date"] = pd.to_datetime(display_trades["date"]).dt.strftime(
                "%Y-%m-%d"
            )
        st.subheader("Trade List")
        st.dataframe(display_trades, use_container_width=True, hide_index=True)
    else:
        st.subheader("Trade List")
        st.info("No trades executed in this period.")


def page_signals() -> None:
    st.header("Strategy Signals")
    st.caption("Recent price action with buy/sell signals and RSI.")

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Symbol", SYMBOLS, key="sig_symbol")
    with col2:
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), key="sig_strategy")

    try:
        df_full = load_symbol_data(symbol)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    strategy_key = STRATEGIES[strategy_name]
    df_signals = get_strategy(strategy_key).generate_signals(df_full)
    df_recent = df_signals.tail(60).reset_index(drop=True)

    if df_recent.empty:
        st.warning("Not enough data to display.")
        return

    st.plotly_chart(
        plot_signals_chart(df_recent, symbol),
        use_container_width=True,
    )

    signal_rows = df_recent[df_recent["signal"] != 0][
        ["date", "open", "high", "low", "close", "signal"]
    ].copy()
    if not signal_rows.empty:
        signal_rows["signal"] = signal_rows["signal"].map({1: "BUY", -1: "SELL"})
        signal_rows["date"] = signal_rows["date"].dt.strftime("%Y-%m-%d")
        st.subheader("Signals (Last 60 Days)")
        st.dataframe(signal_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No buy or sell signals in the last 60 days.")


def main() -> None:
    st.set_page_config(
        page_title="QuantEdge",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("QuantEdge")
    st.sidebar.caption("AI Algorithmic Trading Platform")

    pages = [
        st.Page(page_backtest, title="Backtest", icon="📊"),
        st.Page(page_signals, title="Strategy Signals", icon="📈"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()

"""
QuantEdge Streamlit dashboard.

Run from project root:
    streamlit run frontend/dashboard.py

Requires FastAPI backend:
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import time
from datetime import datetime

import httpx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

API_BASE = "http://localhost:8000"
WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
STRATEGIES = {
    "RSI Mean Reversion": "rsi",
    "EMA Crossover": "ema",
}
INITIAL_CAPITAL = 100_000
LIVE_REFRESH_SECONDS = 10


def api_get(path: str, params: dict | None = None) -> dict | list:
    with httpx.Client(base_url=API_BASE, timeout=30.0) as client:
        response = client.get(path, params=params)
        response.raise_for_status()
        return response.json()


def api_post(path: str, payload: dict) -> dict:
    with httpx.Client(base_url=API_BASE, timeout=60.0) as client:
        response = client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


def fetch_symbols() -> list[str]:
    data = api_get("/api/symbols")
    symbols = data.get("symbols", [])
    return symbols if symbols else WATCHLIST


def fetch_historical(symbol: str, days: int) -> pd.DataFrame:
    data = api_get("/api/historical", {"symbol": symbol, "days": days})
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_backtest(
    symbol: str,
    strategy: str,
    start_date: str,
    end_date: str,
    capital: float,
) -> dict:
    return api_post(
        "/api/backtest",
        {
            "symbol": symbol,
            "strategy": strategy,
            "start_date": start_date,
            "end_date": end_date,
            "capital": capital,
        },
    )


def fetch_signals(symbol: str, strategy: str, days: int = 60) -> pd.DataFrame:
    data = api_get(
        "/api/signals",
        {"symbol": symbol, "strategy": strategy, "days": days},
    )
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_live_quotes() -> dict:
    return api_get("/api/live/quotes")


def fetch_portfolio() -> dict:
    return api_get("/api/paper-trade/portfolio")


def place_paper_order(symbol: str, side: str, quantity: int, strategy: str) -> dict:
    return api_post(
        "/api/paper-trade/order",
        {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "strategy": strategy,
        },
    )


def compute_buy_and_hold(df: pd.DataFrame, initial_capital: float) -> list[float]:
    if df.empty:
        return []
    first_close = float(df["close"].iloc[0])
    shares = initial_capital / first_close
    return (shares * df["close"]).tolist()


def plot_equity_curve(
    dates: list, strategy_equity: list[float], buy_hold: list[float]
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
            y=buy_hold,
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
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df = df.copy()
    df["rsi_14"] = 100 - (100 / (1 + rs))

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
            x=df["date"],
            y=df["rsi_14"],
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


def signal_label(signal: int) -> str:
    if signal == 1:
        return "BUY"
    if signal == -1:
        return "SELL"
    return "HOLD"


def signal_chip_html(label: str) -> str:
    colors = {
        "BUY": ("#dcfce7", "#166534"),
        "SELL": ("#fee2e2", "#991b1b"),
        "HOLD": ("#f1f5f9", "#475569"),
    }
    bg, fg = colors.get(label, colors["HOLD"])
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85rem;">{label}</span>'
    )


def get_symbol_signal(symbol: str, latest_signal: dict | None) -> str:
    if latest_signal and latest_signal.get("symbol", "").upper() == symbol.upper():
        return signal_label(int(latest_signal.get("signal", 0)))
    try:
        df = fetch_signals(symbol, "rsi", days=14)
        if df.empty:
            return "HOLD"
        active = df[df["signal"] != 0]
        if active.empty:
            return "HOLD"
        return signal_label(int(active.iloc[-1]["signal"]))
    except Exception:
        return "HOLD"


def build_watchlist(live_data: dict) -> pd.DataFrame:
    quotes = {q["symbol"]: q for q in live_data.get("quotes", [])}
    latest_signal = live_data.get("latest_signal")
    rows = []

    for symbol in WATCHLIST:
        price = 0.0
        pct_change = 0.0
        prev_close = None

        try:
            hist = fetch_historical(symbol, days=5)
            if not hist.empty:
                prev_close = float(hist["close"].iloc[-2]) if len(hist) >= 2 else None
        except Exception:
            hist = pd.DataFrame()

        quote = quotes.get(symbol)
        if quote:
            price = float(quote["price"])
        elif not hist.empty:
            price = float(hist["close"].iloc[-1])

        if prev_close and prev_close > 0 and price > 0:
            pct_change = ((price - prev_close) / prev_close) * 100

        sig = get_symbol_signal(symbol, latest_signal)
        rows.append(
            {
                "Symbol": symbol,
                "Price": f"₹{price:,.2f}" if price else "—",
                "% Change": f"{pct_change:+.2f}%" if price else "—",
                "Signal": sig,
            }
        )

    return pd.DataFrame(rows)


def page_backtest() -> None:
    st.header("Backtest")
    st.caption("Run historical simulations via FastAPI backend.")

    try:
        symbols = fetch_symbols()
    except httpx.HTTPError as exc:
        st.error(f"API unavailable at {API_BASE}: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Symbol", symbols, key="bt_symbol")
    with col2:
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), key="bt_strategy")
    with col3:
        capital = st.number_input(
            "Capital (₹)", min_value=1000, value=INITIAL_CAPITAL, step=1000, key="bt_capital"
        )

    try:
        df_full = fetch_historical(symbol, days=5000)
    except httpx.HTTPError as exc:
        st.error(f"Failed to load historical data: {exc}")
        return

    if df_full.empty:
        st.error(f"No historical data for {symbol}.")
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
            result = fetch_backtest(
                symbol=symbol,
                strategy=STRATEGIES[strategy_name],
                start_date=str(start_date),
                end_date=str(end_date),
                capital=float(capital),
            )
            hist = fetch_historical(symbol, days=5000)
            hist = hist[
                (hist["date"] >= pd.Timestamp(start_date))
                & (hist["date"] <= pd.Timestamp(end_date))
            ]
            st.session_state["bt_result"] = result
            st.session_state["bt_buy_hold"] = compute_buy_and_hold(hist, float(capital))
        except httpx.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Backtest failed: {detail}")
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

    if "bt_result" not in st.session_state:
        st.info("Select options and click **Run Backtest** to see results.")
        return

    result = st.session_state["bt_result"]
    buy_hold = st.session_state.get("bt_buy_hold", [])
    equity_curve = result.get("equity_curve", [])
    dates = [p["date"] for p in equity_curve]
    values = [p["value"] for p in equity_curve]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Return", f"{result['total_return_pct']:.2f}%")
    m2.metric("CAGR", f"{result['cagr']:.2f}%")
    m3.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    m4.metric("Max Drawdown", f"{result['max_drawdown_pct']:.2f}%")
    m5.metric("Win Rate", f"{result['win_rate']:.2f}%")

    if dates and values:
        st.plotly_chart(
            plot_equity_curve(dates, values, buy_hold[: len(values)]),
            use_container_width=True,
        )

    trades = result.get("trades", [])
    st.subheader("Trade List")
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed in this period.")


def page_signals() -> None:
    st.header("Strategy Signals")
    st.caption("Recent price action with buy/sell signals and RSI.")

    try:
        symbols = fetch_symbols()
    except httpx.HTTPError as exc:
        st.error(f"API unavailable at {API_BASE}: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Symbol", symbols, key="sig_symbol")
    with col2:
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), key="sig_strategy")

    try:
        df_recent = fetch_signals(symbol, STRATEGIES[strategy_name], days=60)
    except httpx.HTTPError as exc:
        st.error(f"Failed to load signals: {exc}")
        return

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


def page_live_trading() -> None:
    st.header("Live Trading")
    st.caption(
        f"Auto-refreshes every {LIVE_REFRESH_SECONDS}s · API: {API_BASE}"
    )

    try:
        live_data = fetch_live_quotes()
        portfolio = fetch_portfolio()
    except httpx.HTTPError as exc:
        st.error(f"API unavailable at {API_BASE}: {exc}")
        st.info("Start backend: `uvicorn backend.app.main:app --reload`")
        st.info("Start live feed: `python backend/app/data/live_feed.py`")
        time.sleep(LIVE_REFRESH_SECONDS)
        st.rerun()
        return

    st.subheader("Live Watchlist")
    watchlist_df = build_watchlist(live_data)

    header_cols = st.columns([2, 2, 2, 2])
    headers = ["Symbol", "Price", "% Change", "Signal"]
    for col, title in zip(header_cols, headers):
        col.markdown(f"**{title}**")

    for _, row in watchlist_df.iterrows():
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        c1.write(row["Symbol"])
        c2.write(row["Price"])
        c3.write(row["% Change"])
        c4.markdown(signal_chip_html(row["Signal"]), unsafe_allow_html=True)

    st.divider()

    st.subheader("Paper Portfolio")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Cash", f"₹{portfolio['cash']:,.2f}")
    p2.metric("Total Value", f"₹{portfolio['total_value']:,.2f}")
    p3.metric("Total P&L", f"₹{portfolio['total_pnl']:,.2f}")
    p4.metric("Unrealized P&L", f"₹{portfolio['unrealized_pnl']:,.2f}")

    positions = portfolio.get("positions", [])
    st.markdown("**Open Positions**")
    if positions:
        pos_df = pd.DataFrame(positions)
        pos_df = pos_df.rename(
            columns={
                "symbol": "Symbol",
                "quantity": "Qty",
                "avg_price": "Avg Price",
                "current_price": "Current",
                "market_value": "Mkt Value",
                "unrealized_pnl": "Unrealized P&L",
                "strategy": "Strategy",
            }
        )
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    st.divider()

    col_form, col_trades = st.columns([1, 1])

    with col_form:
        st.subheader("Place Paper Trade")
        with st.form("paper_trade_form", clear_on_submit=True):
            trade_symbol = st.selectbox("Symbol", WATCHLIST, key="pt_symbol")
            trade_side = st.selectbox("Side", ["buy", "sell"], key="pt_side")
            trade_qty = st.number_input("Quantity", min_value=1, value=1, key="pt_qty")
            trade_strategy = st.selectbox(
                "Strategy", list(STRATEGIES.values()), key="pt_strategy"
            )
            submitted = st.form_submit_button("Place Order", type="primary")

        if submitted:
            try:
                result = place_paper_order(
                    symbol=trade_symbol,
                    side=trade_side,
                    quantity=int(trade_qty),
                    strategy=trade_strategy,
                )
                st.success(result.get("message", "Order placed"))
                time.sleep(1)
                st.rerun()
            except httpx.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Order failed: {detail}")

    with col_trades:
        st.subheader("Last 5 Paper Trades")
        recent = portfolio.get("recent_trades", [])
        if recent:
            trades_df = pd.DataFrame(recent)
            trades_df = trades_df.rename(
                columns={
                    "symbol": "Symbol",
                    "side": "Side",
                    "quantity": "Qty",
                    "entry_price": "Entry",
                    "current_price": "Current",
                    "pnl": "P&L",
                    "timestamp": "Time",
                }
            )
            display_cols = ["Symbol", "Side", "Qty", "Entry", "Current", "P&L", "Time"]
            available = [c for c in display_cols if c in trades_df.columns]
            st.dataframe(
                trades_df[available],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No paper trades yet.")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    time.sleep(LIVE_REFRESH_SECONDS)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="QuantEdge",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("QuantEdge")
    st.sidebar.caption("AI Algorithmic Trading Platform")
    st.sidebar.markdown(f"**API:** `{API_BASE}`")

    pages = [
        st.Page(page_backtest, title="Backtest", icon="📊"),
        st.Page(page_signals, title="Strategy Signals", icon="📈"),
        st.Page(page_live_trading, title="Live Trading", icon="⚡"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()

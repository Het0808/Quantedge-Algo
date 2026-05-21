import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = _file_dir
for _ in range(5):
    if os.path.exists(os.path.join(_project_root, "data")):
        break
    _project_root = os.path.dirname(_project_root)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from backend.app.strategies.rsi_strategy import RSIStrategy
from backend.app.strategies.ema_strategy import EMAStrategy
from backend.app.backtesting.engine import Backtester

st.set_page_config(
    page_title="QuantEdge — Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
DATA_DIR = "data/raw"

BACKTEST_CONFIG = {
    "initial_capital": 100000,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.10,
    "max_holding_days": 15,
}

COLOR_GREEN = "#00C896"
COLOR_RED = "#FF4B4B"
COLOR_BLUE = "#4B8BFF"
COLOR_YELLOW = "#FFD700"
COLOR_BG_DARK = "#0E1117"
COLOR_BG_CARD = "#1A1D27"
COLOR_BG_CHART = "#131722"
COLOR_TEXT_MAIN = "#FFFFFF"
COLOR_TEXT_DIM = "#8B8FA8"
COLOR_GRID = "#2A2D3E"

st.markdown(
    """
<style>
.stApp { background-color: #0E1117; }
[data-testid="stSidebar"] {
    background-color: #131722;
    border-right: 1px solid #2A2D3E;
}
.metric-card {
    background: #1A1D27;
    border: 1px solid #2A2D3E;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 4px 0;
}
.metric-label {
    font-size: 12px;
    color: #8B8FA8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    margin: 4px 0;
}
.metric-green { color: #00C896; }
.metric-red   { color: #FF4B4B; }
.metric-blue  { color: #4B8BFF; }
.metric-white { color: #FFFFFF; }
.section-header {
    font-size: 16px;
    font-weight: 600;
    color: #FFFFFF;
    padding: 12px 0 8px 0;
    border-bottom: 1px solid #2A2D3E;
    margin-bottom: 16px;
}
.trade-win  { color: #00C896; }
.trade-loss { color: #FF4B4B; }
.page-title {
    font-size: 24px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 13px;
    color: #8B8FA8;
    margin-bottom: 24px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_stock_data(symbol: str) -> pd.DataFrame | None:
    """Load and cache stock CSV data."""
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


@st.cache_data
def load_nifty_data() -> pd.DataFrame | None:
    """Load and cache NIFTY50 data."""
    path = os.path.join(DATA_DIR, "NIFTY50.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def run_backtest_cached(
    symbol: str, strategy_name: str, start_date: str, end_date: str
) -> dict:
    """Run backtest and cache result."""
    stock_df = load_stock_data(symbol)
    nifty_df = load_nifty_data()

    if stock_df is None:
        return {"error": f"No data for {symbol}"}

    mask = (stock_df["date"] >= pd.to_datetime(start_date)) & (
        stock_df["date"] <= pd.to_datetime(end_date)
    )
    stock_df = stock_df[mask].copy()

    if len(stock_df) < 50:
        return {"error": "Not enough data for selected date range"}

    if strategy_name == "RSI Mean Reversion":
        strategy = RSIStrategy(
            rsi_period=14, oversold=45, overbought=60, nifty_df=nifty_df
        )
    else:
        strategy = EMAStrategy(
            fast_period=9, slow_period=21, volume_filter=True, nifty_df=nifty_df
        )

    df_with_signals = strategy.generate_signals(stock_df)

    bt = Backtester(**BACKTEST_CONFIG)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            bt.run(df_with_signals, symbol)
        finally:
            sys.stdout = old_stdout

    results = bt.results()
    results["df"] = df_with_signals
    return results


def build_equity_chart(results: dict, symbol: str) -> go.Figure:
    """Build equity curve chart with benchmark overlay."""
    equity_curve = results.get("equity_curve", [])
    if not equity_curve:
        return go.Figure()

    df_eq = pd.DataFrame(equity_curve)
    df_eq["date"] = pd.to_datetime(df_eq["date"])

    df_stock = results.get("df")
    initial_capital = BACKTEST_CONFIG["initial_capital"]

    if df_stock is not None:
        df_stock = df_stock.copy()
        df_stock["date"] = pd.to_datetime(df_stock["date"])
        first_close = df_stock["close"].iloc[0]
        shares_bh = initial_capital / first_close
        benchmark_values = df_stock["close"] * shares_bh
        benchmark_dates = df_stock["date"]
    else:
        benchmark_values = None
        benchmark_dates = None

    fig = go.Figure()

    if benchmark_values is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_dates,
                y=benchmark_values,
                name="Buy & Hold",
                line=dict(color=COLOR_TEXT_DIM, width=1.5, dash="dot"),
                hovertemplate="%{x}<br>B&H: Rs.%{y:,.0f}<extra></extra>",
            )
        )

    final_val = df_eq["portfolio_value"].iloc[-1]
    start_val = initial_capital
    pct_change = (final_val - start_val) / start_val * 100
    line_color = COLOR_GREEN if pct_change >= 0 else COLOR_RED

    r = int(line_color[1:3], 16)
    g = int(line_color[3:5], 16)
    b = int(line_color[5:7], 16)

    fig.add_trace(
        go.Scatter(
            x=df_eq["date"],
            y=df_eq["portfolio_value"],
            name="Strategy",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.05)",
            hovertemplate="%{x}<br>Portfolio: Rs.%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color=COLOR_TEXT_DIM,
        line_width=1,
        annotation_text="Initial Capital",
        annotation_font_color=COLOR_TEXT_DIM,
    )

    all_values = list(df_eq["portfolio_value"])
    if benchmark_values is not None:
        all_values += list(benchmark_values)
    min_val = min(all_values)
    max_val = max(all_values)

    fig.update_layout(
        title=dict(
            text=f"{symbol} — Portfolio Value Over Time",
            font=dict(color=COLOR_TEXT_MAIN, size=14),
        ),
        paper_bgcolor=COLOR_BG_CHART,
        plot_bgcolor=COLOR_BG_CHART,
        font=dict(color=COLOR_TEXT_MAIN),
        xaxis=dict(gridcolor=COLOR_GRID, showgrid=True, zeroline=False),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            showgrid=True,
            zeroline=False,
            tickprefix="Rs.",
            tickformat=",.0f",
            autorange=True,
            range=[min_val * 0.95, max_val * 1.05],
        ),
        legend=dict(
            bgcolor=COLOR_BG_CARD, bordercolor=COLOR_GRID, borderwidth=1
        ),
        hovermode="x unified",
        height=380,
        margin=dict(l=60, r=20, t=50, b=40),
    )

    return fig


def build_monthly_heatmap(results: dict) -> go.Figure:
    """Build monthly returns heatmap."""
    equity_curve = results.get("equity_curve", [])
    if not equity_curve:
        return go.Figure()

    df_eq = pd.DataFrame(equity_curve)
    df_eq["date"] = pd.to_datetime(df_eq["date"])
    df_eq = df_eq.set_index("date")

    monthly = df_eq["portfolio_value"].resample("ME").last()
    monthly_returns = monthly.pct_change() * 100
    monthly_returns = monthly_returns.dropna()

    if len(monthly_returns) < 2:
        return go.Figure()

    df_monthly = pd.DataFrame(
        {
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        }
    )

    pivot = df_monthly.pivot_table(
        index="year", columns="month", values="return", aggfunc="first"
    )

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    x_labels = [month_names[m - 1] for m in pivot.columns]
    y_labels = [str(y) for y in pivot.index]

    colorscale = [
        [0.0, "#FF4B4B"],
        [0.45, "#FF9999"],
        [0.5, "#2A2D3E"],
        [0.55, "#99FFCC"],
        [1.0, "#00C896"],
    ]

    z_values = pivot.values
    text_values = np.where(
        np.isnan(z_values),
        "",
        np.vectorize(lambda x: f"{x:+.1f}%")(np.nan_to_num(z_values)),
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            colorscale=colorscale,
            zmid=0,
            showscale=False,
            hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Monthly Returns Heatmap",
            font=dict(color=COLOR_TEXT_MAIN, size=14),
        ),
        paper_bgcolor=COLOR_BG_CHART,
        plot_bgcolor=COLOR_BG_CHART,
        font=dict(color=COLOR_TEXT_MAIN),
        xaxis=dict(side="top"),
        height=200,
        margin=dict(l=60, r=20, t=80, b=20),
    )

    return fig


def build_signal_chart(df: pd.DataFrame, symbol: str, days: int = 180) -> go.Figure:
    """Build candlestick chart with EMA overlays, signals, and RSI subplot."""
    df = df.tail(days).copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{symbol} — Price & Signals", "RSI (14)"],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color=COLOR_GREEN,
            decreasing_line_color=COLOR_RED,
            increasing_fillcolor=COLOR_GREEN,
            decreasing_fillcolor=COLOR_RED,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if "ema_fast" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["ema_fast"],
                name="EMA Fast",
                line=dict(color=COLOR_YELLOW, width=1.5),
                hovertemplate="EMA Fast: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if "ema_slow" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["ema_slow"],
                name="EMA Slow",
                line=dict(color=COLOR_BLUE, width=1.5),
                hovertemplate="EMA Slow: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    buy_df = df[df["signal"] == 1]
    if len(buy_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_df["date"],
                y=buy_df["low"] * 0.995,
                mode="markers",
                name="BUY",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=COLOR_GREEN,
                    line=dict(color="white", width=1),
                ),
                hovertemplate="BUY @ %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    sell_df = df[df["signal"] == -1]
    if len(sell_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_df["date"],
                y=sell_df["high"] * 1.005,
                mode="markers",
                name="SELL",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=COLOR_RED,
                    line=dict(color="white", width=1),
                ),
                hovertemplate="SELL @ %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rsi"],
                name="RSI",
                line=dict(color=COLOR_BLUE, width=1.5),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=60,
            line_dash="dash",
            line_color=COLOR_RED,
            line_width=1,
            row=2,
            col=1,
        )
        fig.add_hline(
            y=45,
            line_dash="dash",
            line_color=COLOR_GREEN,
            line_width=1,
            row=2,
            col=1,
        )
        fig.add_hrect(
            y0=45,
            y1=60,
            fillcolor="rgba(255,255,255,0.03)",
            line_width=0,
            row=2,
            col=1,
        )

    fig.update_layout(
        paper_bgcolor=COLOR_BG_CHART,
        plot_bgcolor=COLOR_BG_CHART,
        font=dict(color=COLOR_TEXT_MAIN),
        xaxis=dict(gridcolor=COLOR_GRID, rangeslider=dict(visible=False)),
        xaxis2=dict(gridcolor=COLOR_GRID),
        yaxis=dict(
            gridcolor=COLOR_GRID, tickprefix="Rs.", tickformat=",.0f"
        ),
        yaxis2=dict(
            gridcolor=COLOR_GRID,
            range=[0, 100],
            tickvals=[0, 30, 45, 60, 70, 100],
        ),
        legend=dict(
            bgcolor=COLOR_BG_CARD,
            bordercolor=COLOR_GRID,
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        height=550,
        margin=dict(l=60, r=20, t=60, b=20),
    )

    return fig


def render_metric_card(label: str, value: str, color_class: str) -> str:
    """Return HTML string for a metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


def render_metric_row(results: dict) -> None:
    """Render 4 metric cards in a row."""
    col1, col2, col3, col4 = st.columns(4)

    total_return = results.get("total_return_pct", 0)
    sharpe = results.get("sharpe_ratio", 0)
    max_dd = results.get("max_drawdown_pct", 0)
    win_rate = results.get("win_rate", 0)

    with col1:
        color = "metric-green" if total_return >= 0 else "metric-red"
        st.markdown(
            render_metric_card("Total Return", f"{total_return:+.1f}%", color),
            unsafe_allow_html=True,
        )

    with col2:
        color = (
            "metric-green"
            if sharpe >= 1.0
            else "metric-blue"
            if sharpe >= 0.5
            else "metric-red"
        )
        st.markdown(
            render_metric_card("Sharpe Ratio", f"{sharpe:.2f}", color),
            unsafe_allow_html=True,
        )

    with col3:
        color = "metric-green" if max_dd > -10 else "metric-red"
        st.markdown(
            render_metric_card("Max Drawdown", f"{max_dd:.1f}%", color),
            unsafe_allow_html=True,
        )

    with col4:
        color = "metric-green" if win_rate >= 50 else "metric-red"
        st.markdown(
            render_metric_card("Win Rate", f"{win_rate:.1f}%", color),
            unsafe_allow_html=True,
        )


def render_trade_table(results: dict) -> None:
    """Render trade history as a styled dataframe."""
    trades = results.get("trades", [])
    if not trades:
        st.info("No trades in this period.")
        return

    df_trades = pd.DataFrame(trades)

    cols_to_show = {
        "entry_date": "Entry Date",
        "exit_date": "Exit Date",
        "entry_price": "Entry (Rs.)",
        "exit_price": "Exit (Rs.)",
        "shares": "Shares",
        "net_pnl": "Net P&L (Rs.)",
        "pnl_pct": "P&L %",
        "exit_reason": "Exit Reason",
        "days_held": "Days Held",
    }

    existing = {k: v for k, v in cols_to_show.items() if k in df_trades.columns}
    df_show = df_trades[list(existing.keys())].copy()
    df_show = df_show.rename(columns=existing)

    for col in ["Entry (Rs.)", "Exit (Rs.)"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].round(2)

    if "Net P&L (Rs.)" in df_show.columns:
        df_show["Net P&L (Rs.)"] = df_show["Net P&L (Rs.)"].round(0).astype(int)

    if "P&L %" in df_show.columns:
        df_show["P&L %"] = df_show["P&L %"].apply(lambda x: f"{x:+.1f}%")

    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Net P&L (Rs.)": st.column_config.NumberColumn(format="Rs. %d"),
            "Exit Reason": st.column_config.TextColumn(),
        },
    )

    total_pnl = sum(t.get("net_pnl", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("net_pnl", 0) > 0)
    color = COLOR_GREEN if total_pnl >= 0 else COLOR_RED
    st.markdown(
        f'<p style="color:{color};font-size:13px;">'
        f"Total P&L: Rs.{total_pnl:+,.0f} | "
        f"Winners: {wins}/{len(trades)}</p>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """Render sidebar controls and return selected values."""
    import datetime

    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center;padding:20px 0 10px 0;">
            <div style="font-size:28px;">📈</div>
            <div style="font-size:20px;font-weight:700;
                        color:#FFFFFF;">QuantEdge</div>
            <div style="font-size:11px;color:#8B8FA8;">
                Algo Trading Dashboard</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "📊 Backtest Results",
                "⚔️ Strategy Comparison",
                "📡 Signal Chart",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown(
            '<div style="font-size:12px;color:#8B8FA8;'
            'margin-bottom:8px;">SETTINGS</div>',
            unsafe_allow_html=True,
        )

        symbol = st.selectbox("Symbol", SYMBOLS, index=0)

        strategy = st.selectbox(
            "Strategy",
            ["EMA Crossover", "RSI Mean Reversion"],
            index=0,
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From",
                value=datetime.date(2022, 1, 1),
                min_value=datetime.date(2020, 1, 1),
                max_value=datetime.date(2025, 12, 31),
            )
        with col2:
            end_date = st.date_input(
                "To",
                value=datetime.date(2024, 12, 31),
                min_value=datetime.date(2020, 1, 1),
                max_value=datetime.date(2025, 12, 31),
            )

        st.divider()

        with st.expander("Strategy Parameters"):
            st.markdown(
                """
            **EMA Crossover**
            - Fast EMA: 9 periods
            - Slow EMA: 21 periods
            - Volume filter: ON
            - NIFTY50 regime: ON

            **RSI Mean Reversion**
            - RSI Period: 14
            - Oversold: 45
            - Overbought: 60
            - NIFTY50 regime: ON

            **Risk Settings**
            - Stop Loss: 5%
            - Take Profit: 10%
            - Max Hold: 15 days
            - Capital: Rs.1,00,000
            """
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:10px;color:#8B8FA8;'
            f'text-align:center;">'
            f"Last updated<br>"
            f'{datetime.datetime.now().strftime("%d %b %Y %H:%M")}'
            f"</div>",
            unsafe_allow_html=True,
        )

        return {
            "page": page,
            "symbol": symbol,
            "strategy": strategy,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }


def page_backtest(config: dict) -> None:
    st.markdown(
        '<div class="page-title">📊 Backtest Results</div>'
        '<div class="page-subtitle">'
        "Test your strategy on historical data</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:11px;color:#8B8FA8;'
        'margin-bottom:6px;">Quick select:</div>',
        unsafe_allow_html=True,
    )
    display_symbol = st.session_state.get("quick_symbol", config["symbol"])
    btn_cols = st.columns(5)
    for i, sym in enumerate(SYMBOLS):
        with btn_cols[i]:
            is_selected = display_symbol == sym
            if st.button(
                sym,
                key=f"quick_{sym}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.quick_symbol = sym
                st.rerun()

    if st.session_state.get("quick_symbol"):
        config = {**config, "symbol": st.session_state.quick_symbol}

    active_symbol = config["symbol"]
    st.markdown(
        f'<div style="font-size:12px;color:{COLOR_GREEN};'
        f'margin-bottom:8px;">'
        f"Selected: <b>{active_symbol}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"Running **{config['strategy']}** on "
        f"**{config['symbol']}** | "
        f"{config['start_date']} to {config['end_date']}",
    )

    run_col, _ = st.columns([1, 4])
    with run_col:
        run_clicked = st.button(
            "▶ Run Backtest", type="primary", use_container_width=True
        )

    cache_key = f"{config['symbol']}_{config['strategy']}_{config['start_date']}_{config['end_date']}"

    if "last_results" not in st.session_state:
        st.session_state.last_results = {}

    if run_clicked:
        with st.spinner(
            f"Running {config['strategy']} on {config['symbol']}..."
        ):
            results = run_backtest_cached(
                config["symbol"],
                config["strategy"],
                config["start_date"],
                config["end_date"],
            )
        st.session_state.last_results[cache_key] = results

    if cache_key in st.session_state.last_results:
        results = st.session_state.last_results[cache_key]

        if "error" in results:
            st.error(f"Error: {results['error']}")
            return

        if results.get("total_trades", 0) == 0:
            st.warning(
                "No trades generated. "
                "Try a wider date range or different strategy."
            )

        st.markdown(
            '<div class="section-header">Performance Metrics</div>',
            unsafe_allow_html=True,
        )
        render_metric_row(results)

        total_return = results.get("total_return_pct", 0)
        sharpe = results.get("sharpe_ratio", 0)
        cagr = results.get("cagr", 0)

        if sharpe >= 1.0:
            verdict_color = "#00C896"
            verdict_icon = "★★★"
            verdict_text = "Excellent — Ready for paper trading"
        elif sharpe >= 0.5:
            verdict_color = "#4B8BFF"
            verdict_icon = "★★"
            verdict_text = "Good — Paper trade with small size"
        elif sharpe >= 0.0:
            verdict_color = "#FFD700"
            verdict_icon = "★"
            verdict_text = "Average — Keep improving"
        else:
            verdict_color = "#FF4B4B"
            verdict_icon = "✗"
            verdict_text = "Weak — Do not trade real money"

        st.markdown(
            f"""
<div style="background:#1A1D27;border:1px solid
{verdict_color}33;border-radius:12px;
padding:16px 20px;margin:12px 0;
display:flex;align-items:center;gap:16px;">
    <div style="font-size:28px;">{verdict_icon}</div>
    <div>
        <div style="color:{verdict_color};
        font-size:14px;font-weight:600;">
        {verdict_text}</div>
        <div style="color:#8B8FA8;font-size:12px;
        margin-top:4px;">
        CAGR {cagr:+.1f}% per year |
        Sharpe {sharpe:.2f} |
        Min acceptable Sharpe for trading: 0.50
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", results.get("total_trades", 0))
        c2.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
        c3.metric("Avg Days Held", f"{results.get('avg_days_held', 0):.0f}d")
        c4.metric("Alpha vs B&H", f"{results.get('alpha', 0):+.1f}%")

        st.markdown(
            '<div class="section-header">Equity Curve</div>',
            unsafe_allow_html=True,
        )
        fig_equity = build_equity_chart(results, config["symbol"])
        st.plotly_chart(
            fig_equity,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            '<div class="section-header">Monthly Returns</div>',
            unsafe_allow_html=True,
        )
        fig_heatmap = build_monthly_heatmap(results)
        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            '<div class="section-header">Exit Reason Breakdown</div>',
            unsafe_allow_html=True,
        )
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Stop Loss", results.get("stop_loss_exits", 0))
        ec2.metric("Take Profit", results.get("take_profit_exits", 0))
        ec3.metric("Time Exit", results.get("time_exits", 0))
        ec4.metric("Signal Exit", results.get("signal_exits", 0))

        st.markdown(
            '<div class="section-header">'
            f"Trade History ({results.get('total_trades', 0)} trades)</div>",
            unsafe_allow_html=True,
        )
        render_trade_table(results)

    else:
        st.info(
            "👆 Click **Run Backtest** to see results. "
            "First run takes ~5 seconds."
        )


def page_comparison(config: dict) -> None:
    st.markdown(
        '<div class="page-title">⚔️ Strategy Comparison</div>'
        '<div class="page-subtitle">'
        "RSI vs EMA across all 5 NIFTY50 stocks</div>",
        unsafe_allow_html=True,
    )

    if st.button("▶ Run All Strategies", type="primary"):
        st.session_state.comparison_done = True

    if not st.session_state.get("comparison_done"):
        st.info(
            "👆 Click **Run All Strategies** to compare "
            "RSI and EMA on all 5 stocks. Takes ~20 seconds."
        )
        return

    nifty_df = load_nifty_data()

    strategies = {
        "EMA Crossover": EMAStrategy(
            fast_period=9,
            slow_period=21,
            volume_filter=True,
            nifty_df=nifty_df,
        ),
        "RSI Mean Reversion": RSIStrategy(
            rsi_period=14, oversold=45, overbought=60, nifty_df=nifty_df
        ),
    }

    all_results = []
    progress = st.progress(0)
    total_runs = len(strategies) * len(SYMBOLS)
    run_count = 0

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            for strat_name, strategy in strategies.items():
                for symbol in SYMBOLS:
                    df = load_stock_data(symbol)
                    if df is None:
                        run_count += 1
                        progress.progress(run_count / total_runs)
                        continue
                    try:
                        df_sig = strategy.generate_signals(df.copy())
                        bt = Backtester(**BACKTEST_CONFIG)
                        bt.run(df_sig, symbol)
                        r = bt.results()
                        r["strategy"] = strat_name
                        all_results.append(r)
                    except Exception as exc:
                        st.warning(f"{strat_name} {symbol}: {exc}")
                    run_count += 1
                    progress.progress(run_count / total_runs)
        finally:
            sys.stdout = old_stdout

    progress.empty()

    if not all_results:
        st.error("No results generated.")
        return

    df_results = pd.DataFrame(all_results)

    st.markdown(
        '<div class="section-header">Sharpe Ratio by Symbol</div>',
        unsafe_allow_html=True,
    )

    fig_sharpe = go.Figure()
    for strat in df_results["strategy"].unique():
        sub = df_results[df_results["strategy"] == strat]
        fig_sharpe.add_trace(
            go.Bar(
                name=strat,
                x=sub["symbol"],
                y=sub["sharpe_ratio"],
                text=sub["sharpe_ratio"].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
            )
        )

    fig_sharpe.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=COLOR_GREEN,
        annotation_text="Target (0.5)",
        annotation_font_color=COLOR_GREEN,
    )
    fig_sharpe.add_hline(y=0, line_color=COLOR_TEXT_DIM, line_width=1)
    fig_sharpe.update_layout(
        paper_bgcolor=COLOR_BG_CHART,
        plot_bgcolor=COLOR_BG_CHART,
        font=dict(color=COLOR_TEXT_MAIN),
        barmode="group",
        yaxis=dict(gridcolor=COLOR_GRID, title="Sharpe Ratio"),
        xaxis=dict(gridcolor=COLOR_GRID),
        legend=dict(bgcolor=COLOR_BG_CARD, bordercolor=COLOR_GRID),
        height=320,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(
        fig_sharpe,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        '<div class="section-header">CAGR by Symbol</div>',
        unsafe_allow_html=True,
    )

    fig_cagr = go.Figure()
    for strat in df_results["strategy"].unique():
        sub = df_results[df_results["strategy"] == strat]
        fig_cagr.add_trace(
            go.Bar(
                name=strat,
                x=sub["symbol"],
                y=sub["cagr"],
                text=sub["cagr"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            )
        )

    fig_cagr.update_layout(
        paper_bgcolor=COLOR_BG_CHART,
        plot_bgcolor=COLOR_BG_CHART,
        font=dict(color=COLOR_TEXT_MAIN),
        barmode="group",
        yaxis=dict(gridcolor=COLOR_GRID, title="CAGR %", ticksuffix="%"),
        xaxis=dict(gridcolor=COLOR_GRID),
        legend=dict(bgcolor=COLOR_BG_CARD, bordercolor=COLOR_GRID),
        height=320,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(
        fig_cagr, use_container_width=True, config={"displayModeBar": False}
    )

    st.markdown(
        '<div class="section-header">Full Comparison Table</div>',
        unsafe_allow_html=True,
    )

    show_cols = [
        "strategy",
        "symbol",
        "cagr",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate",
        "total_trades",
        "profit_factor",
        "alpha",
    ]
    existing = [c for c in show_cols if c in df_results.columns]
    df_show = df_results[existing].copy()
    rename_map = {
        "strategy": "Strategy",
        "symbol": "Symbol",
        "cagr": "CAGR %",
        "sharpe_ratio": "Sharpe",
        "max_drawdown_pct": "Max DD %",
        "win_rate": "Win Rate %",
        "total_trades": "Trades",
        "profit_factor": "Profit Factor",
        "alpha": "Alpha %",
    }
    df_show = df_show.rename(
        columns={k: rename_map[k] for k in existing if k in rename_map}
    )

    for col in df_show.select_dtypes("float").columns:
        df_show[col] = df_show[col].round(2)

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="section-header">Average Metrics by Strategy</div>',
        unsafe_allow_html=True,
    )

    avg = (
        df_results.groupby("strategy")[
            ["sharpe_ratio", "cagr", "win_rate", "max_drawdown_pct", "total_trades"]
        ]
        .mean()
        .round(2)
    )

    st.dataframe(avg, use_container_width=True)

    best_strategy = avg["sharpe_ratio"].idxmax()
    best_sharpe = avg["sharpe_ratio"].max()
    verdict = (
        "Ready for paper trading!" if best_sharpe >= 0.5 else "Needs more work."
    )
    st.markdown(
        f"**Best strategy: {best_strategy}** "
        f"(avg Sharpe: {best_sharpe:.2f}) — {verdict}"
    )


def page_signals(config: dict) -> None:
    st.markdown(
        '<div class="page-title">📡 Signal Chart</div>'
        '<div class="page-subtitle">'
        "Live price chart with strategy signals</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        days = st.slider(
            "Days to show", min_value=60, max_value=365, value=180, step=30
        )
    with c2:
        chart_strategy = st.selectbox(
            "Strategy for signals",
            ["EMA Crossover", "RSI Mean Reversion"],
            key="chart_strategy",
        )
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button(
            "📡 Generate", type="primary", use_container_width=True
        )

    if generate_clicked or st.session_state.get("signal_chart_done"):
        st.session_state.signal_chart_done = True

        with st.spinner("Generating signals..."):
            df = load_stock_data(config["symbol"])
            nifty_df = load_nifty_data()

            if df is None:
                st.error(
                    f"No data for {config['symbol']}. "
                    f"Run download_nifty.py first."
                )
                return

            if chart_strategy == "EMA Crossover":
                strategy = EMAStrategy(
                    fast_period=9,
                    slow_period=21,
                    volume_filter=True,
                    nifty_df=nifty_df,
                )
            else:
                strategy = RSIStrategy(
                    rsi_period=14,
                    oversold=45,
                    overbought=60,
                    nifty_df=nifty_df,
                )

            df_signals = strategy.generate_signals(df.copy())

        recent = df_signals.tail(days)
        buy_count = int((recent["signal"] == 1).sum())
        sell_count = int((recent["signal"] == -1).sum())

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Last Price", f"Rs.{df_signals['close'].iloc[-1]:,.2f}")
        s2.metric("Buy Signals (period)", buy_count)
        s3.metric("Sell Signals (period)", sell_count)

        last_sig_df = df_signals[df_signals["signal"] != 0].tail(1)
        if len(last_sig_df) > 0:
            last_sig = last_sig_df.iloc[0]
            sig_type = "BUY 🟢" if last_sig["signal"] == 1 else "SELL 🔴"
            sig_date = str(last_sig["date"])[:10]
            s4.metric("Last Signal", f"{sig_type} on {sig_date}")

        st.markdown(
            '<div class="section-header">Price Chart with Signals</div>',
            unsafe_allow_html=True,
        )
        fig_signals = build_signal_chart(df_signals, config["symbol"], days=days)
        st.plotly_chart(
            fig_signals,
            use_container_width=True,
            config={"displayModeBar": True},
        )

        st.markdown(
            '<div class="section-header">Recent Signals</div>',
            unsafe_allow_html=True,
        )

        sig_df = (
            df_signals[df_signals["signal"] != 0]
            .tail(10)[["date", "signal", "close"]]
            .copy()
        )

        if len(sig_df) > 0:
            sig_df["Type"] = sig_df["signal"].map({1: "🟢 BUY", -1: "🔴 SELL"})
            sig_df = sig_df.rename(columns={"date": "Date", "close": "Price (Rs.)"})[
                ["Date", "Type", "Price (Rs.)"]
            ]
            sig_df["Price (Rs.)"] = sig_df["Price (Rs.)"].round(2)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals in this period.")

    else:
        st.info("👆 Click **Generate** to see signals on the chart.")


def main() -> None:
    config = render_sidebar()
    page = config["page"]

    if page == "📊 Backtest Results":
        page_backtest(config)
    elif page == "⚔️ Strategy Comparison":
        page_comparison(config)
    elif page == "📡 Signal Chart":
        page_signals(config)


if __name__ == "__main__":
    main()

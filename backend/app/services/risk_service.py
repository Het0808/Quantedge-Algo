"""Paper trading risk limits and status."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from backend.app.services.data_service import get_latest_close

MAX_POSITION_PCT = 20.0
MAX_OPEN_POSITIONS = 5
DAILY_LOSS_LIMIT_PCT = -5.0
WARNING_LOSS_PCT = -3.0


class RiskViolationError(Exception):
    """Raised when an order violates risk rules."""

    def __init__(self, error: str, message: str) -> None:
        self.error = error
        self.message = message
        super().__init__(message)


def _parse_order_date(timestamp: str) -> date:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()


def _portfolio_snapshot(state: dict[str, Any]) -> dict[str, float]:
    cash = float(state["cash"])
    positions_value = 0.0
    position_values: dict[str, float] = {}

    for symbol, pos in state["positions"].items():
        qty = int(pos["quantity"])
        current_price = get_latest_close(symbol)
        market_value = current_price * qty
        positions_value += market_value
        position_values[symbol] = market_value

    total_value = cash + positions_value
    return {
        "cash": cash,
        "positions_value": positions_value,
        "total_value": total_value,
        "position_values": position_values,
    }


def _today_realized_pnl(state: dict[str, Any]) -> float:
    today = datetime.now(timezone.utc).date()
    total = 0.0
    for order in state["orders"]:
        if order.get("side") != "sell":
            continue
        if _parse_order_date(order["timestamp"]) != today:
            continue
        total += float(order.get("realized_pnl", 0))
    return total


def get_risk_metrics(state: dict[str, Any] | None = None) -> dict[str, Any]:
    from backend.app.services.paper_trade_service import _load_state

    if state is None:
        state = _load_state()

    initial_capital = float(state["initial_capital"])
    snapshot = _portfolio_snapshot(state)
    total_value = snapshot["total_value"]
    positions = state["positions"]
    open_positions_count = len(positions)

    if total_value > 0:
        largest_position_value = (
            max(snapshot["position_values"].values()) if snapshot["position_values"] else 0.0
        )
        total_exposure_pct = (snapshot["positions_value"] / total_value) * 100
        largest_position_pct = (largest_position_value / total_value) * 100
    else:
        total_exposure_pct = 0.0
        largest_position_pct = 0.0

    today_realized = _today_realized_pnl(state)
    today_pnl_pct = (today_realized / initial_capital * 100) if initial_capital else 0.0

    risk_status = "NORMAL"
    if today_pnl_pct <= DAILY_LOSS_LIMIT_PCT:
        risk_status = "HALT"
    elif (
        today_pnl_pct <= WARNING_LOSS_PCT
        or open_positions_count >= MAX_OPEN_POSITIONS - 1
        or largest_position_pct >= MAX_POSITION_PCT - 2
        or total_exposure_pct >= 80
    ):
        risk_status = "WARNING"

    return {
        "total_exposure_pct": round(total_exposure_pct, 2),
        "open_positions_count": open_positions_count,
        "today_pnl_pct": round(today_pnl_pct, 2),
        "today_realized_pnl": round(today_realized, 2),
        "largest_position_pct": round(largest_position_pct, 2),
        "max_position_pct_limit": MAX_POSITION_PCT,
        "max_open_positions_limit": MAX_OPEN_POSITIONS,
        "daily_loss_limit_pct": DAILY_LOSS_LIMIT_PCT,
        "risk_status": risk_status,
        "total_portfolio_value": round(total_value, 2),
        "initial_capital": round(initial_capital, 2),
    }


def validate_buy_order(
    state: dict[str, Any],
    symbol: str,
    quantity: int,
    price: float,
) -> None:
    metrics = get_risk_metrics(state)
    initial_capital = float(state["initial_capital"])
    snapshot = _portfolio_snapshot(state)
    total_value = snapshot["total_value"]
    positions = state["positions"]

    if metrics["today_pnl_pct"] <= DAILY_LOSS_LIMIT_PCT:
        raise RiskViolationError(
            "DAILY_LOSS_LIMIT",
            (
                f"Daily loss limit reached ({metrics['today_pnl_pct']:.2f}%). "
                f"New buy orders blocked until tomorrow. Limit: {DAILY_LOSS_LIMIT_PCT}% "
                f"of starting capital (₹{initial_capital:,.0f})."
            ),
        )

    order_value = price * quantity
    existing_qty = int(positions.get(symbol, {}).get("quantity", 0))
    new_position_value = (existing_qty + quantity) * price
    max_position_value = total_value * (MAX_POSITION_PCT / 100)

    if total_value > 0 and new_position_value > max_position_value:
        raise RiskViolationError(
            "POSITION_LIMIT",
            (
                f"Position would exceed {MAX_POSITION_PCT}% of portfolio. "
                f"Max allowed: ₹{max_position_value:,.2f}"
            ),
        )

    is_new_position = symbol not in positions or existing_qty == 0
    if is_new_position and len(positions) >= MAX_OPEN_POSITIONS:
        raise RiskViolationError(
            "MAX_POSITIONS",
            (
                f"Maximum {MAX_OPEN_POSITIONS} open positions allowed. "
                f"Currently holding {len(positions)} symbols."
            ),
        )

    if order_value > snapshot["cash"]:
        raise RiskViolationError(
            "INSUFFICIENT_CASH",
            (
                f"Insufficient cash. Need ₹{order_value:,.2f}, "
                f"available ₹{snapshot['cash']:,.2f}"
            ),
        )

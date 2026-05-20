"""Paper trading portfolio persisted in paper_trades.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.app.core.paths import PAPER_TRADES_FILE
from backend.app.services.data_service import get_latest_close
from backend.app.services.risk_service import (
    RiskViolationError,
    get_risk_metrics,
    validate_buy_order,
)

# Re-export for API consumers
__all__ = [
    "RiskViolationError",
    "place_order",
    "get_portfolio",
    "get_risk_metrics",
    "get_recent_trades",
]

DEFAULT_CAPITAL = 100_000.0


def _default_state() -> dict[str, Any]:
    return {
        "initial_capital": DEFAULT_CAPITAL,
        "cash": DEFAULT_CAPITAL,
        "positions": {},
        "orders": [],
    }


def _load_state() -> dict[str, Any]:
    if not PAPER_TRADES_FILE.exists():
        return _default_state()
    with PAPER_TRADES_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("initial_capital", DEFAULT_CAPITAL)
    data.setdefault("cash", DEFAULT_CAPITAL)
    data.setdefault("positions", {})
    data.setdefault("orders", [])
    return data


def _save_state(state: dict[str, Any]) -> None:
    PAPER_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_TRADES_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def place_order(
    symbol: str, side: str, quantity: int, strategy: str
) -> dict[str, Any]:
    symbol = symbol.upper()
    side = side.lower().strip()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if quantity <= 0:
        raise ValueError("quantity must be positive")

    price = get_latest_close(symbol)
    state = _load_state()
    positions: dict[str, Any] = state["positions"]
    cash = float(state["cash"])

    if side == "buy":
        validate_buy_order(state, symbol, quantity, price)
        cost = price * quantity
        cash -= cost
        pos = positions.get(symbol, {"quantity": 0, "avg_price": 0.0, "strategy": strategy})
        old_qty = int(pos["quantity"])
        old_avg = float(pos["avg_price"])
        new_qty = old_qty + quantity
        new_avg = ((old_avg * old_qty) + (price * quantity)) / new_qty if new_qty else price
        positions[symbol] = {
            "quantity": new_qty,
            "avg_price": round(new_avg, 2),
            "strategy": strategy,
        }
        realized_pnl = None
    else:
        pos = positions.get(symbol)
        if not pos or int(pos["quantity"]) < quantity:
            held = int(pos["quantity"]) if pos else 0
            raise ValueError(f"Insufficient shares. Held {held}, tried to sell {quantity}")
        avg_price = float(pos["avg_price"])
        realized_pnl = round((price - avg_price) * quantity, 2)
        cash += price * quantity
        new_qty = int(pos["quantity"]) - quantity
        if new_qty == 0:
            del positions[symbol]
        else:
            positions[symbol] = {
                "quantity": new_qty,
                "avg_price": avg_price,
                "strategy": pos.get("strategy", strategy),
            }

    order = {
        "id": len(state["orders"]) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": round(price, 2),
        "strategy": strategy,
        "total_value": round(price * quantity, 2),
        "realized_pnl": realized_pnl,
    }
    state["cash"] = round(cash, 2)
    state["positions"] = positions
    state["orders"].append(order)
    _save_state(state)
    return order


def get_portfolio() -> dict[str, Any]:
    state = _load_state()
    initial = float(state["initial_capital"])
    cash = float(state["cash"])
    positions_out: list[dict[str, Any]] = []
    positions_value = 0.0
    unrealized_pnl = 0.0

    for symbol, pos in state["positions"].items():
        qty = int(pos["quantity"])
        avg_price = float(pos["avg_price"])
        current_price = get_latest_close(symbol)
        market_value = current_price * qty
        pos_pnl = (current_price - avg_price) * qty
        positions_value += market_value
        unrealized_pnl += pos_pnl
        positions_out.append(
            {
                "symbol": symbol,
                "quantity": qty,
                "avg_price": round(avg_price, 2),
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(pos_pnl, 2),
                "strategy": pos.get("strategy"),
            }
        )

    total_value = cash + positions_value
    total_pnl = total_value - initial
    total_pnl_pct = (total_pnl / initial * 100) if initial else 0.0
    risk = get_risk_metrics(state)

    return {
        "initial_capital": round(initial, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "positions": positions_out,
        "order_count": len(state["orders"]),
        "recent_trades": get_recent_trades(state=state, limit=5),
        "risk_status": risk["risk_status"],
    }


def get_recent_trades(
    state: dict[str, Any] | None = None, limit: int = 5
) -> list[dict[str, Any]]:
    if state is None:
        state = _load_state()
    orders = list(reversed(state["orders"][-limit:]))
    trades: list[dict[str, Any]] = []
    for order in orders:
        current_price = get_latest_close(order["symbol"])
        entry_price = float(order["price"])
        quantity = int(order["quantity"])
        if order["side"] == "buy":
            pnl = (current_price - entry_price) * quantity
        else:
            pnl = float(order.get("realized_pnl", (entry_price - current_price) * quantity))
        trades.append(
            {
                "id": order["id"],
                "timestamp": order["timestamp"],
                "symbol": order["symbol"],
                "side": order["side"].upper(),
                "quantity": quantity,
                "entry_price": round(entry_price, 2),
                "current_price": round(current_price, 2),
                "pnl": round(pnl, 2),
                "strategy": order.get("strategy"),
            }
        )
    return trades

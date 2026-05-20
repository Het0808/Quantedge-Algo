from typing import Any, Optional

from pydantic import BaseModel, Field


class OHLCVBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoricalResponse(BaseModel):
    symbol: str
    days: int
    data: list[OHLCVBar]


class SymbolsResponse(BaseModel):
    symbols: list[str]


class BacktestRequest(BaseModel):
    symbol: str
    strategy: str = Field(..., description="rsi or ema")
    start_date: str = Field(..., examples=["2022-01-01"])
    end_date: str = Field(..., examples=["2025-01-01"])
    capital: float = Field(100_000, gt=0)


class EquityPoint(BaseModel):
    date: str
    value: float


class TradeRecord(BaseModel):
    date: Optional[str] = None
    type: str
    price: float
    shares: int
    fee: float
    pnl: Optional[float] = None


class BacktestResponse(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float
    final_portfolio_value: float
    total_return_pct: float
    cagr: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    equity_curve: list[EquityPoint]
    trades: list[TradeRecord]


class SignalBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    signal: int


class SignalsResponse(BaseModel):
    symbol: str
    strategy: str
    days: int
    data: list[SignalBar]


class PaperOrderRequest(BaseModel):
    symbol: str
    side: str = Field(..., description="buy or sell")
    quantity: int = Field(..., gt=0)
    strategy: str


class PaperOrderResponse(BaseModel):
    message: str
    order: dict[str, Any]


class PositionInfo(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    strategy: Optional[str] = None


class RecentTrade(BaseModel):
    id: int
    timestamp: str
    symbol: str
    side: str
    quantity: int
    entry_price: float
    current_price: float
    pnl: float
    strategy: Optional[str] = None


class PortfolioResponse(BaseModel):
    initial_capital: float
    cash: float
    positions_value: float
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    unrealized_pnl: float
    positions: list[PositionInfo]
    order_count: int
    recent_trades: list[RecentTrade] = []


class LiveQuote(BaseModel):
    symbol: str
    price: float
    volume: int
    timestamp: str


class LiveQuotesResponse(BaseModel):
    quotes: list[LiveQuote]
    latest_signal: Optional[dict[str, Any]] = None

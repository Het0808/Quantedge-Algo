from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.trading import (
    BacktestRequest,
    BacktestResponse,
    HistoricalResponse,
    OHLCVBar,
    PaperOrderRequest,
    PaperOrderResponse,
    PortfolioResponse,
    SignalsResponse,
    SignalBar,
    SymbolsResponse,
)
from backend.app.services import data_service, paper_trade_service

router = APIRouter(prefix="/api", tags=["Trading"])


@router.get("/symbols", response_model=SymbolsResponse)
def get_symbols() -> SymbolsResponse:
    return SymbolsResponse(symbols=data_service.list_available_symbols())


@router.get("/historical", response_model=HistoricalResponse)
def get_historical(
    symbol: str = Query(..., examples=["RELIANCE"]),
    days: int = Query(365, ge=1, le=5000),
) -> HistoricalResponse:
    try:
        records = data_service.get_historical(symbol, days)
        return HistoricalResponse(
            symbol=symbol.upper(),
            days=days,
            data=[OHLCVBar(**row) for row in records],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest) -> BacktestResponse:
    try:
        result = data_service.run_backtest(
            symbol=request.symbol,
            strategy=request.strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            capital=request.capital,
        )
        return BacktestResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/signals", response_model=SignalsResponse)
def get_signals(
    symbol: str = Query(..., examples=["RELIANCE"]),
    strategy: str = Query("rsi", examples=["rsi"]),
    days: int = Query(30, ge=1, le=365),
) -> SignalsResponse:
    try:
        records = data_service.get_signals(symbol, strategy, days=days)
        return SignalsResponse(
            symbol=symbol.upper(),
            strategy=strategy.lower(),
            days=days,
            data=[SignalBar(**row) for row in records],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/paper-trade/order", response_model=PaperOrderResponse)
def paper_trade_order(request: PaperOrderRequest) -> PaperOrderResponse:
    try:
        order = paper_trade_service.place_order(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            strategy=request.strategy,
        )
        return PaperOrderResponse(
            message=f"Paper {request.side.lower()} order executed",
            order=order,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/paper-trade/portfolio", response_model=PortfolioResponse)
def paper_trade_portfolio() -> PortfolioResponse:
    try:
        portfolio = paper_trade_service.get_portfolio()
        return PortfolioResponse(**portfolio)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

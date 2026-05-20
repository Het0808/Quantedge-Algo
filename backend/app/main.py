from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.db_test import router as db_router
from backend.app.api.trading_api import router as trading_router
from backend.app.api.user_api import router as user_router

app = FastAPI(
    title="QuantEdge",
    description="AI Algorithmic Trading Platform API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trading_router)
app.include_router(db_router)
app.include_router(user_router)


@app.get("/")
def home():
    return {
        "message": "QuantEdge Running",
        "docs": "/docs",
        "api": "/api/symbols",
    }

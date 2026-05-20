from fastapi import FastAPI
from backend.app.api.db_test import router as db_router
from backend.app.api.user_api import router as user_router

app = FastAPI(title="QuantEdge")

app.include_router(db_router)
app.include_router(user_router)


@app.get("/")
def home():
    return {"message": "QuantEdge Running"}
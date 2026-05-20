from fastapi import APIRouter
from sqlalchemy import text
from backend.app.core.database import engine

router = APIRouter()

@router.get("/db-test")
async def db_test():
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        value = result.scalar()

    return {"database_connected": value == 1}
from backend.app.core.database import engine
from backend.app.domain.user_model import Base

import asyncio


async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


asyncio.run(init_models())
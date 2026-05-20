from fastapi import APIRouter
from sqlalchemy import select
from backend.app.core.database import AsyncSessionLocal
from backend.app.domain.user_model import User

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/")
async def create_user(username: str, email: str):
    async with AsyncSessionLocal() as session:
        user = User(username=username, email=email)

        session.add(user)
        await session.commit()
        await session.refresh(user)

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email
        }


@router.get("/")
async def get_users():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))

        users = result.scalars().all()

        return [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email
            }
            for u in users
        ]
"""Redis connection helper."""

import os

import redis
from dotenv import load_dotenv

from backend.app.core.paths import PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        load_dotenv(ENV_PATH, override=True)
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(url, decode_responses=True)
    return _redis_client

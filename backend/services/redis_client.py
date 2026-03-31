import json
import logging
import redis.asyncio as aioredis
from backend.config import settings
from backend.models.document import DocumentResult

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None
_fallback: dict[str, str] = {}  # in-memory fallback when Redis is unavailable
STATUS_TTL = 60 * 60 * 24  # 24 hours


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def set_status(doc_id: str, result: DocumentResult) -> None:
    data = result.model_dump_json()
    try:
        r = await get_redis()
        await r.setex(f"doc:status:{doc_id}", STATUS_TTL, data)
    except Exception as e:
        logger.warning("Redis set_status failed for %s: %s — using in-memory fallback", doc_id, e)
        _fallback[doc_id] = data


async def get_status(doc_id: str) -> DocumentResult | None:
    try:
        r = await get_redis()
        data = await r.get(f"doc:status:{doc_id}")
        if data:
            return DocumentResult.model_validate(json.loads(data))
    except Exception as e:
        logger.warning("Redis get_status failed for %s: %s — checking in-memory fallback", doc_id, e)

    # fallback
    data = _fallback.get(doc_id)
    if data:
        return DocumentResult.model_validate(json.loads(data))
    return None


async def close() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None

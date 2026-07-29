"""Redis caching utilities for ASR transcription results.

All public helpers are **fail-open**: if Redis is not initialised or
unreachable the caller silently falls through to normal transcription.
"""

import hashlib
import json
import logging

from redis import asyncio as aioredis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

redis: aioredis.Redis | None = None

# Fail-open only helps if it fails *fast*. Without these, an unreachable or
# wedged Redis stalls application startup and every cache lookup behind the
# client's default retry behaviour.
_CONNECT_TIMEOUT_S = 2.0
_SOCKET_TIMEOUT_S = 2.0


async def init_redis(url: str) -> None:
    """Create the singleton async Redis connection.

    Never raises: an unreachable Redis logs a warning and leaves caching
    disabled for the process.
    """
    global redis
    try:
        client = aioredis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=_CONNECT_TIMEOUT_S,
            socket_timeout=_SOCKET_TIMEOUT_S,
        )
        # Verify the connection is reachable at startup.
        await client.ping()
        redis = client
        log.info("Redis connected at %s", url)
    except Exception as exc:
        log.warning("Redis unavailable (%s) - caching disabled.", exc)
        redis = None


async def close_redis() -> None:
    """Gracefully close the Redis connection."""
    global redis
    if redis is not None:
        await redis.aclose()
        redis = None


def compute_audio_hash(pcm_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of canonical 16-bit PCM bytes."""
    return hashlib.sha256(pcm_bytes).hexdigest()


async def get_from_cache(key: str) -> dict | None:
    """Fetch and JSON-deserialize a cached value, or *None* on miss."""
    if redis is None:
        return None
    try:
        data = await redis.get(key)
        return json.loads(data) if data else None
    except RedisError as exc:
        log.warning("Redis GET failed (%s) - treating as cache miss.", exc)
        return None


async def set_in_cache(key: str, value: dict, ttl: int) -> None:
    """JSON-serialize *value* and store it in Redis with an expiry."""
    if redis is None:
        return
    try:
        await redis.set(key, json.dumps(value), ex=ttl)
    except RedisError as exc:
        log.warning("Redis SET failed (%s) - result not cached.", exc)


async def acquire_lock(key: str, timeout: int = 60) -> bool:
    """Try to acquire a per-key lock (SET NX EX). Returns *True* on success."""
    if redis is None:
        return True  # no Redis -> proceed with transcription
    try:
        return await redis.set(f"lock:{key}", "1", nx=True, ex=timeout)
    except RedisError as exc:
        log.warning("Redis lock failed (%s) - proceeding without lock.", exc)
        return True


async def release_lock(key: str) -> None:
    """Release the per-key lock."""
    if redis is None:
        return
    try:
        await redis.delete(f"lock:{key}")
    except RedisError as exc:
        log.warning("Redis unlock failed (%s).", exc)

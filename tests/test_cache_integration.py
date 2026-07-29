"""Redis cache behaviour, against a real Redis.

Marked integration: excluded from the default run and skipped if Redis is not
reachable. Run with::

    docker compose up -d redis
    pytest -m integration

The async Redis client has to be created inside the same event loop that serves
the requests, so these tests mount the routers on a purpose-built app whose
lifespan initialises Redis and nothing else -- no model weights. A separate sync
client handles flushing between tests.
"""

from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager

import pytest
import redis as redis_sync
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import cache
from app.routers import asr, asr_streaming
from tests.conftest import FakeTranscriber, make_wav_bytes

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
TRANSCRIBE = "/api/v1/transcribe"


@pytest.fixture
def flushed_redis():
    """A reachable, empty Redis -- or skip."""
    try:
        client = redis_sync.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis not reachable at {REDIS_URL}: {exc}")

    client.flushdb()
    yield client
    client.flushdb()
    client.close()


@pytest.fixture
def cache_app(app_config, fake_transcriber, flushed_redis):
    """The real routers, a fake transcriber, and a real Redis connection."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await cache.init_redis(REDIS_URL)
        assert cache.redis is not None, "init_redis failed inside the app loop"
        try:
            yield
        finally:
            await cache.close_redis()

    application = FastAPI(lifespan=lifespan)
    application.state.config = app_config
    application.state.transcriber = fake_transcriber
    application.state.inference_semaphore = asyncio.Semaphore(1)
    application.include_router(asr.router, prefix="/api/v1")
    application.include_router(asr_streaming.router, prefix="/api/v1")
    return application


@pytest.fixture
def cached_client(cache_app):
    # Context manager on purpose here: it runs the lifespan that connects Redis.
    with TestClient(cache_app) as client:
        yield client


def _upload(wav: bytes, name: str = "sample.wav") -> dict:
    return {"file": (name, io.BytesIO(wav), "audio/wav")}


def test_second_identical_upload_is_a_cache_hit(cached_client, short_wav_bytes, fake_transcriber):
    """The same audio must not be transcribed twice."""
    first = cached_client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    second = cached_client.post(TRANSCRIBE, files=_upload(short_wav_bytes))

    assert first.status_code == second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert first.json()["text"] == second.json()["text"]

    # The model ran once; the second response came from Redis.
    assert len(fake_transcriber.calls) == 1


def test_cache_key_ignores_container_format(cached_client, short_audio, fake_transcriber):
    """The same samples in a different container must hit the same key.

    The key hashes canonical 16 kHz PCM16 rather than the uploaded bytes, so a
    lossless re-encode still hits.

    The source is int16, deliberately. Handing float32 to both encoders does not
    test this: WAV PCM_16 and FLAC quantize floats differently -- one truncates,
    the other rounds -- so the files would hold genuinely different samples, and
    distinct keys would be the correct outcome.
    """
    import numpy as np
    import soundfile as sf

    audio, sample_rate = short_audio
    samples = np.round(audio * 32000).astype(np.int16)

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    flac_buffer = io.BytesIO()
    sf.write(flac_buffer, samples, sample_rate, format="FLAC", subtype="PCM_16")

    first = cached_client.post(
        TRANSCRIBE, files={"file": ("a.wav", wav_buffer.getvalue(), "audio/wav")}
    )
    second = cached_client.post(
        TRANSCRIBE, files={"file": ("a.flac", flac_buffer.getvalue(), "audio/flac")}
    )

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert len(fake_transcriber.calls) == 1


def test_different_audio_does_not_collide(cached_client, fake_transcriber):
    """Distinct audio must produce distinct cache entries."""
    first = cached_client.post(TRANSCRIBE, files=_upload(make_wav_bytes(3.0, freq=220.0)))
    second = cached_client.post(TRANSCRIBE, files=_upload(make_wav_bytes(3.0, freq=880.0)))

    assert first.json()["cached"] is False
    assert second.json()["cached"] is False
    assert len(fake_transcriber.calls) == 2


def test_failed_transcription_is_not_cached(cache_app, cached_client, short_wav_bytes):
    """A 500 must not poison the cache with a bad entry."""
    cache_app.state.transcriber = FakeTranscriber(error=RuntimeError("transient failure"))
    failed = cached_client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert failed.status_code == 500

    # Once the fault clears, a retry must run inference rather than serve an error.
    working = FakeTranscriber(text="recovered")
    cache_app.state.transcriber = working
    retry = cached_client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert retry.status_code == 200
    assert retry.json()["text"] == "recovered"
    assert len(working.calls) == 1


def test_lock_is_released_after_failure(cache_app, cached_client, short_wav_bytes, flushed_redis):
    """The per-key lock must not outlive a failed request."""
    cache_app.state.transcriber = FakeTranscriber(error=RuntimeError("boom"))
    cached_client.post(TRANSCRIBE, files=_upload(short_wav_bytes))

    leftover = flushed_redis.keys("lock:asr:*")
    assert leftover == [], f"lock left behind: {leftover}"


def test_cached_response_survives_a_new_connection(cache_app, short_wav_bytes, fake_transcriber):
    """Entries must outlive the process that wrote them, not just the request."""
    with TestClient(cache_app) as first_client:
        first = first_client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert first.json()["cached"] is False

    with TestClient(cache_app) as second_client:
        second = second_client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert second.json()["cached"] is True
    assert len(fake_transcriber.calls) == 1

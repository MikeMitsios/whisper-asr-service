"""ASR API routes."""

import asyncio
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.cache import (
    acquire_lock,
    compute_audio_hash,
    get_from_cache,
    release_lock,
    set_in_cache,
)
from app.routers._common import read_and_decode_audio, to_canonical_pcm16

router = APIRouter()

_LOCK_POLL_ATTEMPTS = 10
_LOCK_POLL_INTERVAL_S = 0.2


@router.post("/transcribe")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)"),
):
    """
    Transcribe an audio file to text.

    Audio longer than one 30s encoder window is handled in full -- see
    `long_form_mode` in the config. Results are cached in Redis so repeated
    uploads skip inference.

    Returns:
        ``{text, duration_s, inference_ms, cached, backend}``.
    """
    audio_array, sampling_rate = await read_and_decode_audio(file, request)
    duration_s = round(len(audio_array) / sampling_rate, 3)

    # --- cache lookup ---------------------------------------------------
    pcm_bytes = to_canonical_pcm16(audio_array, sampling_rate)
    cache_key = f"asr:{compute_audio_hash(pcm_bytes)}"

    cached = await get_from_cache(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    # --- lock to avoid duplicate parallel transcriptions ----------------
    if not await acquire_lock(cache_key):
        for _ in range(_LOCK_POLL_ATTEMPTS):
            await asyncio.sleep(_LOCK_POLL_INTERVAL_S)
            cached = await get_from_cache(cache_key)
            if cached is not None:
                return {**cached, "cached": True}
        raise HTTPException(503, "Transcription in progress, try again")

    # --- run inference & store result -----------------------------------
    try:
        transcriber = request.app.state.transcriber
        started = time.perf_counter()
        # Inference is synchronous and CPU/GPU-bound. Running it directly here
        # would block the event loop for its whole duration, stalling every
        # other request including health checks.
        async with request.app.state.inference_semaphore:
            text = await asyncio.to_thread(transcriber.transcribe, audio_array, sampling_rate)
        inference_ms = round((time.perf_counter() - started) * 1000, 1)

        result = {
            "text": text,
            "duration_s": duration_s,
            "inference_ms": inference_ms,
            "backend": transcriber.backend,
        }
        cache_ttl = request.app.state.config.get("cache_ttl", 3600)
        await set_in_cache(cache_key, result, ttl=cache_ttl)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {e}",
        ) from e
    finally:
        await release_lock(cache_key)

    # `cached` is set at response time, not stored, so the cached body stays
    # valid for both a hit and a miss.
    return {**result, "cached": False}

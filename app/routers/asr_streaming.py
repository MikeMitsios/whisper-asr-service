"""Streaming ASR API routes using Server-Sent Events."""

import asyncio
import json

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.routers._common import read_and_decode_audio

router = APIRouter()

# Distinguishes "generator exhausted" from a legitimately yielded value.
_EXHAUSTED = object()


@router.post("/transcribe/stream")
async def transcribe_audio_stream(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)"),
):
    """
    Stream a transcription as Server-Sent Events.

    Audio within one 30s encoder window streams token by token. Longer audio
    streams one timestamp-anchored window at a time -- see
    ``WhisperModel.transcribe_direct_stream`` for why per-token is not possible
    there.

    Event types:
      - ``data: {"token": "..."}``  -- a new fragment of transcribed text
      - ``data: {"done": true}``    -- the stream is complete
      - ``data: {"error": "..."}``  -- generation failed
    """
    audio_array, sampling_rate = await read_and_decode_audio(file, request)
    transcriber = request.app.state.transcriber
    semaphore = request.app.state.inference_semaphore

    async def _event_generator():
        """Yield SSE events, pulling the sync token stream off the event loop.

        Each ``next()`` runs in a worker thread, so generation never blocks the
        loop. The semaphore is held for the whole stream: one model instance
        cannot serve two generations at once.
        """
        async with semaphore:
            tokens = transcriber.transcribe_stream(audio_array, sampling_rate)
            try:
                while True:
                    token = await asyncio.to_thread(next, tokens, _EXHAUSTED)
                    if token is _EXHAUSTED:
                        break
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            finally:
                tokens.close()

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

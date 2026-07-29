"""Tests for the FastAPI endpoints.

These run in-process against the ASGI app with a fake transcriber -- no server
on a port, no model weights, no network.
"""

from __future__ import annotations

import asyncio
import io
import json
import time

import httpx
import pytest

from tests.conftest import FakeTranscriber, make_wav_bytes

TRANSCRIBE = "/api/v1/transcribe"
STREAM = "/api/v1/transcribe/stream"


def _upload(wav: bytes, name: str = "sample.wav", mime: str = "audio/wav") -> dict:
    return {"file": (name, io.BytesIO(wav), mime)}


# -- health and service info -------------------------------------------------


def test_health_returns_ok(client):
    """GET /health returns a dependency-free liveness response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_service_summary(client):
    """GET / reports status and how this instance is configured."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["default_backend"] in ("pipeline", "direct")
    assert data["model_id"] == "openai/whisper-tiny"
    assert data["long_form_mode"] == "auto"


# -- transcription happy path ------------------------------------------------


def test_transcribe_valid_audio(client, short_wav_bytes):
    """A valid upload returns 200 with non-empty text."""
    response = client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert response.status_code == 200
    assert isinstance(response.json()["text"], str)
    assert len(response.json()["text"]) > 0


def test_transcribe_response_shape(client, short_wav_bytes):
    """The response carries timing and provenance, not just text."""
    response = client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"text", "duration_s", "inference_ms", "cached", "backend"}
    assert body["duration_s"] == pytest.approx(3.0, abs=0.05)
    assert body["inference_ms"] >= 0
    assert body["cached"] is False
    assert body["backend"] == "direct"


def test_transcribe_passes_decoded_audio_to_transcriber(
    client, short_wav_bytes, fake_transcriber
):
    """The route decodes to a 16 kHz waveform before calling the transcriber."""
    client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert len(fake_transcriber.calls) == 1
    num_samples, sampling_rate = fake_transcriber.calls[0]
    assert sampling_rate == 16_000
    assert num_samples == pytest.approx(48_000, abs=800)


def test_transcribe_works_without_redis(client, short_wav_bytes):
    """With no Redis the cache is bypassed and requests still succeed."""
    first = client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    second = client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert first.status_code == second.status_code == 200
    assert first.json()["text"] == second.json()["text"]
    assert second.json()["cached"] is False


# -- error paths -------------------------------------------------------------


def test_no_file_returns_422(client):
    """POST without a file attachment returns 422."""
    response = client.post(TRANSCRIBE)
    assert response.status_code == 422


def test_invalid_format_returns_400(client):
    """POST with a non-audio extension returns 400."""
    response = client.post(TRANSCRIBE, files=_upload(b"not audio", "notes.txt", "text/plain"))
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_empty_file_returns_400(client):
    """POST with an empty file returns 400."""
    response = client.post(TRANSCRIBE, files=_upload(b"", "empty.wav"))
    assert response.status_code == 400


def test_corrupt_audio_returns_400(client):
    """POST with non-decodable bytes returns 400."""
    response = client.post(TRANSCRIBE, files=_upload(b"not-real-audio", "bad.wav"))
    assert response.status_code == 400
    assert "Could not decode audio file" in response.json()["detail"]


def test_transcriber_failure_returns_500(api_app, client, short_wav_bytes):
    """An exception from the model surfaces as 500, not a crash."""
    api_app.state.transcriber = FakeTranscriber(error=RuntimeError("CUDA out of memory"))
    response = client.post(TRANSCRIBE, files=_upload(short_wav_bytes))
    assert response.status_code == 500
    assert "CUDA out of memory" in response.json()["detail"]


# -- request limits ----------------------------------------------------------


def test_audio_longer_than_limit_returns_413(api_app, client):
    """Audio over max_audio_seconds is refused, not silently truncated."""
    api_app.state.config["max_audio_seconds"] = 5
    response = client.post(TRANSCRIBE, files=_upload(make_wav_bytes(12.0), "long.wav"))
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "Audio too long" in detail
    assert "12.0s" in detail and "5s" in detail


def test_upload_larger_than_limit_returns_413(api_app, client):
    """A body over max_upload_mb is refused."""
    api_app.state.config["max_upload_mb"] = 0.05  # 50 KB
    response = client.post(TRANSCRIBE, files=_upload(make_wav_bytes(10.0), "big.wav"))
    assert response.status_code == 413
    assert "Upload too large" in response.json()["detail"]


def test_audio_under_the_limit_is_accepted(api_app, client):
    """The duration guard is a ceiling, not an off-by-one."""
    api_app.state.config["max_audio_seconds"] = 10
    response = client.post(TRANSCRIBE, files=_upload(make_wav_bytes(9.5), "ok.wav"))
    assert response.status_code == 200


# -- streaming ---------------------------------------------------------------


def test_transcribe_stream(client, short_wav_bytes):
    """The SSE stream yields fragments and ends with a done event."""
    with client.stream("POST", STREAM, files=_upload(short_wav_bytes)) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        events = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert len(events) > 0
    assert events[-1].get("done") is True
    assert "".join(e["token"] for e in events if "token" in e) == "fake transcript"


def test_stream_reports_generation_errors(api_app, client, short_wav_bytes):
    """A mid-stream failure arrives as an error event, not a dropped connection."""
    api_app.state.transcriber = FakeTranscriber(error=RuntimeError("decoder exploded"))
    with client.stream("POST", STREAM, files=_upload(short_wav_bytes)) as response:
        assert response.status_code == 200
        events = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert events[-1]["error"] == "decoder exploded"


def test_stream_rejects_oversized_audio(api_app, client):
    """The streaming route enforces the same limits as the batch route."""
    api_app.state.config["max_audio_seconds"] = 5
    response = client.post(STREAM, files=_upload(make_wav_bytes(12.0), "long.wav"))
    assert response.status_code == 413


# -- event loop responsiveness ----------------------------------------------


@pytest.mark.anyio
async def test_inference_does_not_block_the_event_loop(api_app, short_wav_bytes):
    """A slow transcription must not stall unrelated requests.

    Inference is synchronous; if it ran on the event loop instead of in a worker
    thread, this health check would wait for it to finish.
    """
    api_app.state.transcriber = FakeTranscriber(delay_s=1.0)
    transport = httpx.ASGITransport(app=api_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        inflight = asyncio.create_task(
            client.post(TRANSCRIBE, files=_upload(short_wav_bytes), timeout=30)
        )
        await asyncio.sleep(0.2)  # let inference start

        started = time.perf_counter()
        health = await client.get("/health")
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert health.status_code == 200
        assert elapsed_ms < 400, f"event loop blocked: /health took {elapsed_ms:.0f}ms"

        response = await inflight
        assert response.status_code == 200

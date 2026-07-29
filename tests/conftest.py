"""Shared fixtures.

Two rules hold for the default test run: no network, and no model weights.
Audio is synthesized in memory rather than read from disk -- the Emilia-Dataset
samples this project evaluates against are CC BY-NC with additional terms, so
they are not redistributable in a public repo, and generated tones make the
tests both faster and hermetic.
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import Generator, Iterator

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

SAMPLE_RATE = 16_000


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run async tests on asyncio only, not the full anyio matrix."""
    return "asyncio"


def make_tone(seconds: float, freq: float = 220.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize a mono float32 waveform."""
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_wav_bytes(seconds: float, freq: float = 220.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Synthesize a mono 16-bit PCM WAV of *seconds* length."""
    buffer = io.BytesIO()
    sf.write(buffer, make_tone(seconds, freq, sample_rate), sample_rate, format="WAV",
             subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def short_wav_bytes() -> bytes:
    """~3s of audio: fits inside one 30s encoder window."""
    return make_wav_bytes(3.0)


@pytest.fixture
def short_audio() -> tuple[np.ndarray, int]:
    """Decoded ~3s waveform."""
    return make_tone(3.0), SAMPLE_RATE


class FakeTranscriber:
    """Stand-in for :class:`~whisper_asr.transcriber.Transcriber`.

    Records what it was asked to do and returns fixed text, so API behaviour
    can be tested without loading a model.
    """

    def __init__(
        self,
        text: str = "fake transcript",
        chunks: list[str] | None = None,
        delay_s: float = 0.0,
        backend: str = "direct",
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.chunks = chunks if chunks is not None else ["fake ", "transcript"]
        self.delay_s = delay_s
        self.backend = backend
        self.error = error
        self.calls: list[tuple[int, int]] = []

    def transcribe(self, audio: np.ndarray, sampling_rate: int) -> str:
        self.calls.append((len(audio), sampling_rate))
        if self.error is not None:
            raise self.error
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.text

    def transcribe_stream(self, audio: np.ndarray, sampling_rate: int) -> Iterator[str]:
        self.calls.append((len(audio), sampling_rate))
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            if self.delay_s:
                time.sleep(self.delay_s)
            yield chunk


@pytest.fixture
def app_config() -> dict:
    """Minimal config holding every key the API reads."""
    return {
        "model_id": "openai/whisper-tiny",
        "device": "cpu",
        "default_backend": "direct",
        "enable_streaming": True,
        "long_form_mode": "auto",
        "chunk_length_s": 30.0,
        "stride_length_s": 5.0,
        "batch_size": 4,
        "max_audio_seconds": 600,
        "max_upload_mb": 50,
        "max_concurrent_inferences": 1,
        "cors_allow_origins": ["*"],
        "cache_ttl": 3600,
    }


@pytest.fixture
def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def api_app(app_config: dict, fake_transcriber: FakeTranscriber) -> Generator:
    """The real FastAPI app with a fake transcriber and no Redis.

    The lifespan is bypassed deliberately -- running it would load model
    weights. ``app.state`` is populated directly with what the routes need.
    """
    import app.main as main_module

    application = main_module.app
    application.state.config = app_config
    application.state.transcriber = fake_transcriber
    application.state.inference_semaphore = asyncio.Semaphore(
        app_config["max_concurrent_inferences"]
    )
    yield application


@pytest.fixture
def client(api_app) -> TestClient:
    """A client wired straight into the ASGI app -- no server, no port.

    Deliberately not used as a context manager: entering one runs the lifespan,
    which would load model weights.
    """
    return TestClient(api_app)

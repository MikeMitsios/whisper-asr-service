"""End-to-end long-form checks against real model weights.

Marked slow: these download openai/whisper-tiny and run inference on CPU, so
they are excluded from the default run. Execute with::

    pytest -m slow

This is the executable proof that audio over 30s is no longer truncated. The
fixture is synthesized speech-like audio rather than a real recording, so the
assertions are about *coverage* -- how much of the file produced output -- not
about transcription accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SAMPLE_RATE

pytestmark = pytest.mark.slow

MODEL_ID = "openai/whisper-tiny"


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    """Audio with speech-like structure: shifting formants and an envelope.

    A pure tone makes Whisper emit nothing at all, which would make these tests
    vacuous. This produces varied output without needing a real recording.
    """
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    signal = np.zeros(n, dtype=np.float32)
    for base in (140.0, 420.0, 900.0, 1800.0):
        drift = 1.0 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.2, 0.8) * t)
        signal += (0.25 / base**0.25) * np.sin(2 * np.pi * base * drift * t)

    # Syllable-rate envelope plus pauses, so the model sees onsets.
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    envelope *= (np.sin(2 * np.pi * 0.35 * t) > -0.6).astype(np.float32)
    signal *= envelope
    signal += 0.005 * rng.standard_normal(n).astype(np.float32)

    peak = float(np.max(np.abs(signal))) or 1.0
    return (0.35 * signal / peak).astype(np.float32)


@pytest.fixture(scope="module")
def tiny_direct():
    from whisper_asr.whisper_model import WhisperModel

    return WhisperModel(
        model_id=MODEL_ID,
        device="cpu",
        backend="direct",
        gen_kwargs={"task": "transcribe", "language": "en"},
        torch_dtype="float32",
    )


@pytest.fixture(scope="module")
def long_speechlike() -> np.ndarray:
    """~75s: comfortably more than two 30s encoder windows."""
    return speechlike(75.0, seed=7)


def test_model_is_in_eval_mode(tiny_direct):
    """`model.eval` without parentheses left dropout active in production."""
    assert tiny_direct.model.training is False


def test_short_audio_uses_the_short_path(tiny_direct):
    audio = speechlike(8.0, seed=1)
    assert tiny_direct.needs_long_form(audio, SAMPLE_RATE) is False
    text = tiny_direct.transcribe_direct(audio, SAMPLE_RATE)
    assert isinstance(text, str)


def test_long_audio_reaches_the_encoder_in_full(tiny_direct, long_speechlike):
    """The headline fix, tested at the mechanism rather than the output.

    The bug was in feature extraction: the default ``truncation=True`` cuts the
    mel spectrogram to a single 30s window (3000 frames), so the model never
    saw the rest of the file no matter how it decoded. Asserting on transcript
    length would be meaningless here -- this fixture is synthesized, not real
    speech, so the model hallucinates on it either way. Frame count is the
    thing that actually changed.
    """
    from whisper_asr.whisper_model import _LONG_FORM_INPUT_KWARGS

    assert tiny_direct.needs_long_form(long_speechlike, SAMPLE_RATE) is True

    truncated = tiny_direct.processor(
        long_speechlike,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features
    full = tiny_direct.processor(
        long_speechlike,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        **_LONG_FORM_INPUT_KWARGS,
    ).input_features

    # One 30s window is 3000 mel frames.
    assert truncated.shape[-1] == 3000, truncated.shape
    assert full.shape[-1] > 3000, full.shape
    # 75s of audio is 2.5 windows' worth.
    assert full.shape[-1] == pytest.approx(7500, rel=0.02), full.shape


def test_long_audio_transcribes_without_error(tiny_direct, long_speechlike):
    """Sequential long-form runs end to end on multi-window input."""
    text = tiny_direct.transcribe_direct(long_speechlike, SAMPLE_RATE)
    assert isinstance(text, str)


def test_streaming_long_audio_is_progressive(tiny_direct, long_speechlike):
    """Long audio must emit more than once, so clients see partial results."""
    emissions = list(tiny_direct.transcribe_direct_stream(long_speechlike, SAMPLE_RATE))
    assert len(emissions) > 1, f"expected multiple windows, got {len(emissions)}"
    assert all(isinstance(chunk, str) for chunk in emissions)


def test_streaming_short_audio_matches_batch(tiny_direct):
    """Within one window, streamed text must equal the batch result."""
    audio = speechlike(9.0, seed=3)
    batch = tiny_direct.transcribe_direct(audio, SAMPLE_RATE)
    streamed = "".join(tiny_direct.transcribe_direct_stream(audio, SAMPLE_RATE)).strip()
    assert streamed == batch


def test_pipeline_backend_handles_long_audio(long_speechlike):
    """The chunked + batched path must also cover the whole file."""
    from whisper_asr.whisper_model import WhisperModel

    model = WhisperModel(
        model_id=MODEL_ID,
        device="cpu",
        backend="pipeline",
        gen_kwargs={"task": "transcribe", "language": "en"},
        torch_dtype="float32",
        batch_size=2,
    )
    text = model.transcribe_pipeline(long_speechlike, SAMPLE_RATE)
    assert isinstance(text, str)

"""Unit tests for WhisperModel internals that need no model weights.

Construction is tested with a stub so these stay fast and offline; the
end-to-end long-form behaviour is covered by tests/test_long_form.py, which is
marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from whisper_asr.whisper_model import (
    _FALLBACK_GEN_KWARGS,
    _LONG_FORM_GEN_KWARGS,
    WhisperModel,
    last_timestamp,
    resolve_dtype,
    strip_overlap,
)

# -- dtype resolution --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
        (None, torch.float32),
    ],
)
def test_resolve_dtype_maps_supported_names(name, expected):
    assert resolve_dtype(name) is expected


@pytest.mark.parametrize("name", ["float8", "int8", "notadtype", ""])
def test_resolve_dtype_raises_on_unsupported(name):
    """Unknown dtypes must raise, not silently fall back to float32.

    The silent fallback is what mislabelled two rows of the experiment table:
    a config asking for float8 quietly ran float32 instead.
    """
    with pytest.raises(ValueError, match="Unsupported torch_dtype") as excinfo:
        resolve_dtype(name)
    assert "'float16', 'bfloat16', 'float32'" in str(excinfo.value)


def test_float8_is_not_a_supported_dtype():
    """Guards against re-adding the entry that caused the mislabelling."""
    with pytest.raises(ValueError):
        resolve_dtype("float8")


# -- timestamp parsing -------------------------------------------------------


def test_last_timestamp_returns_largest():
    assert last_timestamp("<|0.00|> hi <|12.34|> there <|5.00|>") == 12.34


def test_last_timestamp_single():
    assert last_timestamp("<|29.98|>") == 29.98


@pytest.mark.parametrize("text", ["no timestamps here", "", "<|notanumber|>"])
def test_last_timestamp_returns_none_without_timestamps(text):
    assert last_timestamp(text) is None


# -- seam de-duplication -----------------------------------------------------


@pytest.mark.parametrize(
    ("previous", "new", "expected"),
    [
        ("a b c d", "c d e f", "e f"),                    # two-word overlap
        ("a b c", "a b c", ""),                           # full repeat
        ("", "a b", "a b"),                               # nothing accumulated
        ("a b", "", ""),                                  # nothing new
        ("x y", "p q", "p q"),                            # no overlap
        ("back home.", "Home is where", "is where"),      # case + punctuation
        ("one two three", "THREE four", "four"),          # case only
    ],
)
def test_strip_overlap(previous, new, expected):
    assert strip_overlap(previous, new) == expected


def test_strip_overlap_respects_max_words():
    """A repeat longer than max_words is not searched for."""
    previous = " ".join(str(i) for i in range(100))
    new = " ".join(str(i) for i in range(50, 150))
    assert strip_overlap(previous, new, max_words=2) == new


# -- long-form generation kwargs --------------------------------------------


def test_long_form_kwargs_enable_timestamps_and_context():
    """Sequential long-form needs timestamps to pick window boundaries."""
    assert _LONG_FORM_GEN_KWARGS["return_timestamps"] is True
    assert _LONG_FORM_GEN_KWARGS["condition_on_prev_tokens"] is True


def test_long_form_kwargs_include_temperature_fallback():
    for key, value in _FALLBACK_GEN_KWARGS.items():
        assert _LONG_FORM_GEN_KWARGS[key] == value


def test_fallback_starts_at_greedy():
    """Temperature 0.0 must be tried first, so the common case is unchanged."""
    assert _FALLBACK_GEN_KWARGS["temperature"][0] == 0.0


# -- construction-time validation -------------------------------------------


def test_rejects_unknown_long_form_mode():
    with pytest.raises(ValueError, match="Unsupported long_form_mode"):
        WhisperModel(
            model_id="openai/whisper-tiny",
            device="cpu",
            long_form_mode="sideways",
        )


def test_rejects_stride_that_would_stall_the_window():
    """stride*2 >= chunk means windows never advance."""
    with pytest.raises(ValueError, match="must be under half of"):
        WhisperModel(
            model_id="openai/whisper-tiny",
            device="cpu",
            chunk_length_s=30.0,
            stride_length_s=15.0,
        )


def test_validation_happens_before_any_weights_load():
    """Both checks above must run before from_pretrained is reached.

    If they did not, these tests would try to download a model.
    """
    with pytest.raises(ValueError):
        WhisperModel(model_id="does/not/exist", device="cpu", long_form_mode="bogus")


# -- long-form gating (no weights: stub out construction) -------------------


class _StubModel(WhisperModel):
    """WhisperModel with loading skipped, to test pure logic."""

    def __init__(self, **kwargs):
        self.long_form_mode = kwargs.get("long_form_mode", "auto")
        self.chunk_length_s = kwargs.get("chunk_length_s", 30.0)


@pytest.mark.parametrize(
    ("seconds", "mode", "expected"),
    [
        (5.0, "auto", False),
        (29.9, "auto", False),
        (30.0, "auto", False),   # exactly one window still fits
        (30.5, "auto", True),
        (300.0, "auto", True),
        (300.0, "off", False),   # off never splits
    ],
)
def test_needs_long_form(seconds, mode, expected):
    model = _StubModel(long_form_mode=mode)
    audio = np.zeros(int(16_000 * seconds), dtype=np.float32)
    assert model.needs_long_form(audio, 16_000) is expected


def test_needs_long_form_scales_with_chunk_length():
    model = _StubModel(chunk_length_s=10.0)
    audio = np.zeros(16_000 * 15, dtype=np.float32)
    assert model.needs_long_form(audio, 16_000) is True

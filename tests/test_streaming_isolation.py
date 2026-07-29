"""Concurrent streams must not interleave.

A single ``TextIteratorStreamer`` used to be built in ``WhisperModel.__init__``
and shared by every call. Streamers are single-use and stateful, so two
simultaneous streaming requests consumed each other's tokens. These tests pin
the fix without loading model weights, by driving the real streaming code with
a stub model and tokenizer.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from whisper_asr.whisper_model import WhisperModel

SR = 16_000


class _StubTokenizer:
    """Minimal tokenizer: TextIteratorStreamer only needs decode()."""

    def decode(self, token_ids, **kwargs):
        return "".join(str(int(t)) for t in token_ids)

    def batch_decode(self, sequences, **kwargs):
        return ["".join(str(int(t)) for t in sequences[0])]


class _StubProcessor:
    def __init__(self):
        self.tokenizer = _StubTokenizer()

    def __call__(self, audio, sampling_rate, return_tensors=None, **kwargs):
        import torch

        class _Inputs(dict):
            input_features = torch.zeros(1, 80, 3000)

            def get(self, key, default=None):
                return default

        return _Inputs()


class _StubGenerativeModel:
    """Feeds a fixed token sequence into whatever streamer it is handed.

    ``tokens_by_call`` lets each concurrent caller get a distinguishable
    sequence, so cross-talk is detectable.
    """

    def __init__(self, sequences: list[list[int]], delay_s: float = 0.005):
        self._sequences = sequences
        self._delay_s = delay_s
        self._call_index = 0
        self._lock = threading.Lock()

    def generate(self, input_features=None, streamer=None, **kwargs):
        import time

        import torch

        with self._lock:
            sequence = self._sequences[self._call_index % len(self._sequences)]
            self._call_index += 1

        if streamer is not None:
            # The streamer is built with skip_prompt=True, so its first put()
            # is discarded as prompt. Real generate() feeds the decoder prompt
            # there; mimic that so the sequence arrives intact.
            streamer.put(torch.tensor([[0]]))
            for token in sequence:
                time.sleep(self._delay_s)
                streamer.put(torch.tensor([[token]]))
            streamer.end()
        return torch.tensor([sequence])


def _build(sequences: list[list[int]]) -> WhisperModel:
    """A WhisperModel with loading bypassed and stubs in place."""
    model = WhisperModel.__new__(WhisperModel)
    model.device = "cpu"
    model.gen_kwargs = {}
    model.long_form_mode = "auto"
    model.chunk_length_s = 30.0
    model.stride_length_s = 5.0
    model._torch_dtype = __import__("torch").float32
    model._processor = _StubProcessor()
    model._model = _StubGenerativeModel(sequences)
    return model


def test_streamer_is_not_shared_across_instances():
    """The shared-streamer attribute is gone entirely."""
    model = _build([[1, 2, 3]])
    assert not hasattr(model, "_streamer")


def test_single_stream_yields_its_tokens():
    model = _build([[1, 2, 3, 4]])
    audio = np.zeros(SR, dtype=np.float32)
    assert "".join(model.transcribe_direct_stream(audio, SR)) == "1234"


def test_concurrent_streams_do_not_interleave():
    """Two streams running at once must each receive only their own tokens."""
    # Distinct digit ranges make any cross-talk obvious.
    first_tokens = [1] * 12
    second_tokens = [9] * 12
    model = _build([first_tokens, second_tokens])

    audio = np.zeros(SR, dtype=np.float32)
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def run(key: str) -> None:
        try:
            results[key] = "".join(model.transcribe_direct_stream(audio, SR))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(key,)) for key in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert set(results) == {"a", "b"}

    # Each result must be single-valued: all 1s or all 9s, never a mix.
    for key, text in results.items():
        assert len(set(text)) == 1, f"stream {key} interleaved: {text!r}"
    assert {results["a"], results["b"]} == {"1" * 12, "9" * 12}


@pytest.mark.parametrize("stream_count", [3, 5])
def test_many_concurrent_streams_stay_separate(stream_count):
    sequences = [[digit] * 8 for digit in range(1, stream_count + 1)]
    model = _build(sequences)
    audio = np.zeros(SR, dtype=np.float32)

    results: list[str] = []
    lock = threading.Lock()

    def run() -> None:
        text = "".join(model.transcribe_direct_stream(audio, SR))
        with lock:
            results.append(text)

    threads = [threading.Thread(target=run) for _ in range(stream_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(results) == stream_count
    for text in results:
        assert len(set(text)) == 1, f"interleaved: {text!r}"
    assert len(set(results)) == stream_count, f"streams collided: {results}"

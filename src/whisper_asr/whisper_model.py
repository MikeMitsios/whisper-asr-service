"""Whisper model wrapper with eager loading and two inference backends."""

from __future__ import annotations

import re
from collections.abc import Generator
from threading import Thread
from typing import Literal

import numpy as np
import torch
from optimum.quanto import Calibration, freeze, quantize
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Pipeline,
    TextIteratorStreamer,
    pipeline,
)
from transformers.utils import logging

from whisper_asr.load_data import EMILIA_LANGUAGES, load_emilia_samples

logging.set_verbosity_error()

# Model precisions we support. Deliberately no float8 entry: there is no
# torch.float8_* dtype that Whisper's generate() can run in, so accepting the
# name would only let a config silently run something other than what it asked
# for. Activation quantization to float8 is a separate thing -- see
# quantized_config's `activations` key.
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Quantization types accepted by optimum-quanto, for weights and activations.
_QUANT_TYPES = ("qint2", "qint4", "qint8", "qfloat8")

LONG_FORM_MODES = ("auto", "off")

# Whisper's encoder takes a fixed 30s window. Past that, generate() runs the
# paper's sequential algorithm: it reads the last predicted timestamp to pick
# where the next window starts, and feeds the previous window's text back in as
# a decoder prompt. The thresholds drive temperature fallback -- a window whose
# output looks degenerate (repetition loops show up as a low compression ratio,
# low confidence as a low mean logprob) is retried hotter.
# Temperature fallback. A window whose output looks degenerate -- repetition
# loops show up as a low compression ratio, low confidence as a low mean
# logprob -- is retried at a higher temperature. Cheap insurance: temperature
# 0.0 is tried first and almost always accepted.
_FALLBACK_GEN_KWARGS = {
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
}

_LONG_FORM_GEN_KWARGS = {
    "return_timestamps": True,
    "condition_on_prev_tokens": True,
    **_FALLBACK_GEN_KWARGS,
}

# Feature-extractor kwargs that stop the 30s truncation. Without these the
# extractor silently cuts long audio and generate() never sees the rest.
_LONG_FORM_INPUT_KWARGS = {
    "truncation": False,
    "padding": "longest",
    "return_attention_mask": True,
}


_TIMESTAMP_TOKEN = re.compile(r"<\|(\d+\.\d+)\|>")


def last_timestamp(decoded_with_specials: str) -> float | None:
    """Return the largest ``<|x.xx|>`` timestamp in a decoded sequence.

    Whisper emits timestamp tokens alongside text when asked. The last one
    marks how far into the window the model believes it transcribed, which is
    where the next window should start.

    Args:
        decoded_with_specials: Text decoded with ``skip_special_tokens=False``.

    Returns:
        The timestamp in seconds, or ``None`` if the sequence has none.
    """
    matches = _TIMESTAMP_TOKEN.findall(decoded_with_specials)
    if not matches:
        return None
    return max(float(m) for m in matches)


def strip_overlap(previous: str, new: str, max_words: int = 40) -> str:
    """Drop the leading words of *new* that already end *previous*.

    Consecutive windows share audio, so their transcripts share text at the
    seam. This removes the longest such repeat, comparing case- and
    punctuation-insensitively so "home." and "home" still match.

    Args:
        previous: Text accumulated so far.
        new: Text produced by the next window.
        max_words: Longest repeat to look for.

    Returns:
        *new* with its duplicated prefix removed.
    """
    prev_words = previous.split()
    new_words = new.split()
    if not prev_words or not new_words:
        return new

    def norm(words: list[str]) -> list[str]:
        return [w.lower().strip(".,!?;:\"'") for w in words]

    limit = min(len(prev_words), len(new_words), max_words)
    normalized_prev = norm(prev_words)
    normalized_new = norm(new_words)
    for n in range(limit, 0, -1):
        if normalized_prev[-n:] == normalized_new[:n]:
            return " ".join(new_words[n:])
    return new


def resolve_dtype(name: str | None) -> torch.dtype:
    """Map a config dtype name to a ``torch.dtype``.

    Args:
        name: Dtype name from config, or ``None`` to accept the default.

    Returns:
        The matching ``torch.dtype``. ``None`` yields ``torch.float32``.

    Raises:
        ValueError: If *name* is not a supported dtype.
    """
    if name is None:
        return torch.float32
    if name not in _DTYPE_MAP:
        expected = ", ".join(repr(k) for k in _DTYPE_MAP)
        raise ValueError(f"Unsupported torch_dtype {name!r}. Expected one of: {expected}.")
    return _DTYPE_MAP[name]


class WhisperModel:
    """Loads a Whisper model at init and exposes pipeline / direct inference."""

    def __init__(
        self,
        model_id: str,
        device: str,
        backend: Literal["pipeline", "direct"] = "pipeline",
        gen_kwargs: dict | None = None,
        quantized_config: dict | None = None,
        torch_dtype: str | None = None,
        save_model_id: str | None = None,
        enable_calibration: bool = False,
        compile_model: bool = False,
        long_form_mode: Literal["auto", "off"] = "auto",
        chunk_length_s: float = 30.0,
        stride_length_s: float = 5.0,
        batch_size: int = 8,
    ) -> None:
        if long_form_mode not in LONG_FORM_MODES:
            expected = ", ".join(repr(m) for m in LONG_FORM_MODES)
            raise ValueError(
                f"Unsupported long_form_mode {long_form_mode!r}. Expected one of: {expected}."
            )
        if stride_length_s * 2 >= chunk_length_s:
            raise ValueError(
                f"stride_length_s ({stride_length_s}) must be under half of "
                f"chunk_length_s ({chunk_length_s}), otherwise windows never advance."
            )

        self.model_id = model_id
        self.device = device
        self.backend = backend
        self.gen_kwargs = gen_kwargs or {}
        self._torch_dtype = resolve_dtype(torch_dtype)
        self.save_model_id = save_model_id
        self.enable_calibration = enable_calibration
        self.long_form_mode = long_form_mode
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.batch_size = batch_size

        # Always load model + processor (needed by direct and streaming).
        # Additionally load pipeline if that is the chosen backend.
        self._model, self._processor = self._build_model_and_processor()

        if quantized_config:
            self._quantize_model(quantized_config)

        # Save before compiling: torch.compile returns an OptimizedModule
        # wrapper, and saving through it writes a wrapped checkpoint.
        if self.save_model_id:
            self._model.save_pretrained(self.save_model_id)
            self._processor.save_pretrained(self.save_model_id)

        # Compilation is opt-in. It costs a long warm-up on first inference and
        # buys little to nothing on CPU.
        if compile_model:
            self._model = torch.compile(self._model)

        self._pipeline: Pipeline | None = None
        if self.backend == "pipeline":
            self._pipeline = self._build_pipeline()

    # -- properties ----------------------------------------------------------

    @property
    def model(self) -> AutoModelForSpeechSeq2Seq:
        """The underlying HuggingFace model."""
        return self._model

    @property
    def processor(self) -> AutoProcessor:
        """The underlying HuggingFace processor."""
        return self._processor

    @property
    def torch_dtype(self) -> torch.dtype:
        """The resolved model precision."""
        return self._torch_dtype

    # -- builders ------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        # `torch_dtype` rather than `dtype`: the `dtype` alias only exists from
        # transformers 4.56, and `torch_dtype` is accepted across the whole
        # 4.x line.
        kwargs = {
            "task": "automatic-speech-recognition",
            "model": self.model_id,
            "torch_dtype": self._torch_dtype,
            "device": 0 if self.device == "cuda" else -1,
            "generate_kwargs": self.gen_kwargs,
        }
        if self.long_form_mode == "auto":
            # Chunked + batched long-form: windows are transcribed
            # independently, so they batch, and the overlap lets the pipeline
            # stitch the seams. Faster than sequential, slightly less accurate.
            kwargs["chunk_length_s"] = self.chunk_length_s
            kwargs["stride_length_s"] = (self.stride_length_s, self.stride_length_s)
            kwargs["batch_size"] = self.batch_size
        return pipeline(**kwargs)

    def _build_model_and_processor(
        self,
    ) -> tuple[AutoModelForSpeechSeq2Seq, AutoProcessor]:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype,
        )
        model.to(self.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    # -- public inference methods --------------------------------------------

    def transcribe_pipeline(self, audio: np.ndarray, sampling_rate: int) -> str:
        """Run inference via the HuggingFace pipeline API."""
        result = self._pipeline({"array": audio, "sampling_rate": sampling_rate})
        text = result.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        return text.strip()

    def needs_long_form(self, audio: np.ndarray, sampling_rate: int) -> bool:
        """Whether *audio* is longer than one encoder window and should be split."""
        if self.long_form_mode == "off":
            return False
        return len(audio) > int(self.chunk_length_s * sampling_rate)

    def transcribe_direct(self, audio: np.ndarray, sampling_rate: int) -> str:
        """Run inference via processor + model.generate.

        Audio over one window is handled by transformers' sequential long-form
        path, which needs the feature extractor to stop truncating and needs
        timestamps enabled to know where each window ends.
        """
        long_form = self.needs_long_form(audio, sampling_rate)

        inputs = self._processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            **(_LONG_FORM_INPUT_KWARGS if long_form else {}),
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self._torch_dtype)

        # Caller gen_kwargs last so an explicit setting always wins.
        generate_kwargs: dict = {"use_cache": True}
        if long_form:
            generate_kwargs.update(_LONG_FORM_GEN_KWARGS)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask.to(device=self.device)
        generate_kwargs.update(self.gen_kwargs)

        generated = self._model.generate(input_features=input_features, **generate_kwargs)
        return self._processor.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()

    def transcribe_direct_stream(
        self,
        audio: np.ndarray,
        sampling_rate: int,
    ) -> Generator[str, None, None]:
        """
        Stream transcription text as it is generated.

        Granularity depends on length, and the reason is worth knowing:

        * **Up to one encoder window (30s)** -- true token-by-token streaming
          via ``TextIteratorStreamer``. This is the common case for a streaming
          endpoint, and the text is identical to ``transcribe_direct``.
        * **Longer** -- the audio is cut into overlapping windows and each
          window's transcript is emitted when it completes, with the duplicated
          text at the seam removed. A ten-minute upload produces output
          continuously rather than nothing until the end.

        Long audio cannot stream per token, for two compounding reasons.
        Windows must be anchored on the model's own predicted timestamps rather
        than cut blindly every 30s -- a blind cut lands mid-utterance and the
        model responds by predicting EOS almost immediately (one window in
        testing returned 7 words for a full 30s of speech). Reading that
        timestamp means decoding the window's output, which only exists once
        generation finishes. Temperature fallback compounds it: it retries a
        degenerate window, and ``TextIteratorStreamer`` closes after the first
        pass, so it would hand back exactly the output the retries exist to
        replace. Correct text at window granularity beats corrupt text at token
        granularity.

        Yields:
            Decoded text fragments (str), in order.
        """
        if not self.needs_long_form(audio, sampling_rate):
            yield from self._stream_window(audio, sampling_rate)
            return

        window = int(self.chunk_length_s * sampling_rate)
        # Floor on how far a window may advance, so a short or timestamp-less
        # decode cannot stall the loop.
        min_advance = max(int(sampling_rate), window // 4)
        fallback_advance = window - int(self.stride_length_s * sampling_rate)

        position = 0
        accumulated = ""
        while position < len(audio):
            text, end_s = self._transcribe_window(
                audio[position : position + window],
                sampling_rate,
                robust=True,
            )

            if text:
                if accumulated:
                    text = strip_overlap(accumulated, text)
                if text:
                    chunk = f" {text}" if accumulated else text
                    accumulated += chunk
                    yield chunk

            # Advance to where the model said it stopped. Clamp so we always
            # make progress and never skip audio.
            if end_s is None:
                advance = fallback_advance
            else:
                advance = int(end_s * sampling_rate)
            position += max(min_advance, min(advance, window))

    def _transcribe_window(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        robust: bool = False,
    ) -> tuple[str, float | None]:
        """Transcribe one window in a single blocking call.

        Args:
            audio: Waveform for this window.
            sampling_rate: Sample rate of *audio*.
            robust: Enable temperature fallback and timestamps, so a window
                that decodes degenerately is retried rather than returned, and
                the caller learns where the model stopped.

        Returns:
            ``(text, last_timestamp_seconds)``. The timestamp is ``None`` when
            *robust* is off or the model emitted none.
        """
        inputs = self._processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self._torch_dtype)

        generate_kwargs: dict = {"use_cache": True}
        if robust:
            generate_kwargs.update(_FALLBACK_GEN_KWARGS)
            generate_kwargs["return_timestamps"] = True
        generate_kwargs.update(self.gen_kwargs)

        generated = self._model.generate(input_features=input_features, **generate_kwargs)
        tokenizer = self._processor.tokenizer
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        end_s = None
        if robust:
            with_specials = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
            end_s = last_timestamp(with_specials)

        return text, end_s

    def _stream_window(
        self,
        audio: np.ndarray,
        sampling_rate: int,
    ) -> Generator[str, None, None]:
        """Stream one window of at most ``chunk_length_s``, token by token.

        A streamer is created per call -- it is single-use and stateful, so
        sharing one across calls would interleave their token streams.

        No temperature fallback here: it needs several generate passes and the
        streamer closes after the first. See ``transcribe_direct_stream``.
        """
        inputs = self._processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self._torch_dtype)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            "input_features": input_features,
            **self.gen_kwargs,
            "use_cache": True,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        try:
            for token_text in streamer:
                if token_text:
                    yield token_text
        finally:
            thread.join()

    # -- quantization --------------------------------------------------------

    def _quantize_model(self, quantized_config: dict) -> None:
        """Quantize the loaded model in-place using optimum-quanto.

        Args:
            quantized_config: Dict with optional keys ``weights`` and
                ``activations``, each holding a quanto type name
                (``"qint2"``, ``"qint4"``, ``"qint8"``, ``"qfloat8"``) or
                ``None``.

        Raises:
            ValueError: If either key holds an unrecognised type name.
        """
        for key in ("weights", "activations"):
            value = quantized_config.get(key)
            if value is not None and value not in _QUANT_TYPES:
                expected = ", ".join(repr(t) for t in _QUANT_TYPES)
                raise ValueError(
                    f"Unsupported quantized_config.{key} {value!r}. Expected one of: {expected}."
                )

        quantize(self._model, **quantized_config)
        if self.enable_calibration and quantized_config.get("activations") is not None:
            self._calibrate_model()
        freeze(self._model)
        print("[WhisperModel] Quantization complete.")

    def _calibrate_model(self, num_samples: int = 20) -> None:
        """Estimate activation ranges by running representative samples.

        Activation quantization needs a dynamic range to map floats onto ints,
        and that range depends on the input distribution. Without this pass the
        scales are unset and activations saturate.
        """
        language = str(self.gen_kwargs.get("language", "en")).lower()
        if language not in EMILIA_LANGUAGES:
            language = "en"

        try:
            calibration_samples = load_emilia_samples(language=language, num_samples=num_samples)
        except Exception as exc:
            print(f"[WhisperModel] Calibration skipped: {exc}")
            return

        if not calibration_samples:
            print("[WhisperModel] Calibration skipped: no Emilia samples loaded.")
            return

        decoder_start = self._model.config.decoder_start_token_id
        if decoder_start is None:
            print("[WhisperModel] Calibration skipped: decoder start token is missing.")
            return

        with Calibration(momentum=0.9):
            for sample in calibration_samples:
                inputs = self._processor(
                    sample["audio"],
                    sampling_rate=sample["sr"],
                    return_tensors="pt",
                )
                input_features = inputs.input_features.to(
                    device=self.device,
                    dtype=self._torch_dtype,
                )
                decoder_input_ids = torch.tensor(
                    [[decoder_start]],
                    device=self.device,
                )
                with torch.no_grad():
                    self._model(
                        input_features=input_features,
                        decoder_input_ids=decoder_input_ids,
                    )

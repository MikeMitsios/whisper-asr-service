"""Config-driven transcription orchestrator."""

from __future__ import annotations

from collections.abc import Generator
from typing import Literal

import numpy as np

from whisper_asr.audio_utils import resample_audio
from whisper_asr.whisper_model import WhisperModel


class Transcriber:
    """Reads config, builds a WhisperModel, and orchestrates transcription."""

    def __init__(self, config: dict) -> None:
        self._backend: Literal["pipeline", "direct"] = config.get(
            "default_backend", "pipeline",
        )
        self._target_sr: int = config.get("target_sample_rate", 16000)
        self._model = WhisperModel(
            model_id=config["model_id"],
            device=config["device"],
            backend=self._backend,
            gen_kwargs=config.get("gen_kwargs"),
            quantized_config=config.get("quantized_config"),
            torch_dtype=config.get("torch_dtype"),
            save_model_id=config.get("save_model_id"),
            enable_calibration=config.get("enable_calibration", False),
            compile_model=config.get("compile_model", False),
            long_form_mode=config.get("long_form_mode", "auto"),
            chunk_length_s=config.get("chunk_length_s", 30.0),
            stride_length_s=config.get("stride_length_s", 5.0),
            batch_size=config.get("batch_size", 8),
        )

    @property
    def backend(self) -> Literal["pipeline", "direct"]:
        """The configured inference backend."""
        return self._backend

    @property
    def torch_model(self) -> object:
        """The underlying HuggingFace model, for size and dtype inspection."""
        return self._model.model

    def transcribe(self, audio: np.ndarray, sampling_rate: int) -> str:
        """Resample if needed, then delegate to the configured backend."""
        if sampling_rate != self._target_sr:
            audio = resample_audio(audio, sampling_rate, self._target_sr)
            sampling_rate = self._target_sr

        if self._backend == "pipeline":
            return self._model.transcribe_pipeline(audio, sampling_rate)
        return self._model.transcribe_direct(audio, sampling_rate)

    def transcribe_stream(
        self, audio: np.ndarray, sampling_rate: int,
    ) -> Generator[str, None, None]:
        """Resample if needed, then stream tokens via the direct backend."""
        if sampling_rate != self._target_sr:
            audio = resample_audio(audio, sampling_rate, self._target_sr)
            sampling_rate = self._target_sr

        yield from self._model.transcribe_direct_stream(audio, sampling_rate)

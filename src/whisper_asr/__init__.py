"""Whisper ASR package with pipeline and direct model backends.

Public names resolve lazily via :pep:`562` so that importing the package -- or
just ``whisper_asr.config`` -- does not pull in ``torch``, ``transformers``,
``datasets`` and ``optimum-quanto``. Those imports cost several seconds, which
matters both for the API's startup path and for keeping the test suite fast.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public name -> submodule that defines it.
_EXPORTS = {
    "load_config": "whisper_asr.config",
    "Transcriber": "whisper_asr.transcriber",
    "WhisperModel": "whisper_asr.whisper_model",
    "Evaluator": "whisper_asr.evaluator",
    "load_emilia_samples": "whisper_asr.load_data",
    "compare_sizes": "whisper_asr.quantization",
    "format_size": "whisper_asr.quantization",
    "get_dir_size_bytes": "whisper_asr.quantization",
    "get_model_memory_bytes": "whisper_asr.quantization",
    "load_original_model": "whisper_asr.quantization",
    "print_param_dtypes": "whisper_asr.quantization",
    "quantize_model": "whisper_asr.quantization",
    "save_model": "whisper_asr.quantization",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import the owning submodule on first access to a public name."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)

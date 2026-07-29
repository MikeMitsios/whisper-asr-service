"""YAML configuration loading and validation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_KEYS = ("model_id", "device")

# Misspelled keys that shipped in earlier configs, mapped to their real names.
_RENAMED_KEYS = {"enable_callibration": "enable_calibration"}

_DEFAULTS = {
    "default_backend": "pipeline",
    "sample_audios_folder": "sample_audios",
    # Long-form audio. Whisper's encoder window is 30s; "auto" splits anything
    # longer, "off" reverts to the old behaviour of truncating at one window.
    "long_form_mode": "auto",
    "chunk_length_s": 30.0,
    "stride_length_s": 5.0,
    "batch_size": 8,
    # Request limits, enforced by the API before inference.
    "max_audio_seconds": 600,
    "max_upload_mb": 50,
    # One model instance means concurrent generate() calls contend for it.
    "max_concurrent_inferences": 1,
    "cors_allow_origins": ["*"],
}

# Keys whose value must be one of a fixed set. Anything else raises rather than
# silently falling back -- a mistyped config should fail loudly at startup, not
# quietly run something other than what was asked for.
_ALLOWED_VALUES: dict[str, tuple[str, ...]] = {
    "default_backend": ("pipeline", "direct"),
    "device": ("cpu", "cuda"),
    "long_form_mode": ("auto", "off"),
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dictionary.

    Required keys: model_id, device.
    Optional keys get default values if not provided.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The parsed configuration as a plain dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required key is missing or a key holds an
            unrecognised value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    missing = [k for k in _REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    _apply_renamed_keys(config)

    for key, default in _DEFAULTS.items():
        config.setdefault(key, default)

    _validate_allowed_values(config)

    return config


def _apply_renamed_keys(config: dict[str, Any]) -> None:
    """Migrate deprecated key spellings in place, warning for each one."""
    for old, new in _RENAMED_KEYS.items():
        if old not in config:
            continue
        value = config.pop(old)
        config.setdefault(new, value)
        warnings.warn(
            f"Config key {old!r} is deprecated; use {new!r} instead.",
            DeprecationWarning,
            stacklevel=3,
        )


def _validate_allowed_values(config: dict[str, Any]) -> None:
    """Raise if any constrained key holds a value outside its allowed set."""
    for key, allowed in _ALLOWED_VALUES.items():
        if key in config and config[key] not in allowed:
            expected = ", ".join(repr(v) for v in allowed)
            raise ValueError(
                f"Invalid config value for {key!r}: {config[key]!r}. "
                f"Expected one of: {expected}."
            )

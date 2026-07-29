"""Config loading, defaults, validation, and deprecated key migration."""

from __future__ import annotations

import pytest
import yaml

from whisper_asr.config import load_config

MINIMAL = {"model_id": "openai/whisper-tiny", "device": "cpu"}


def write_config(tmp_path, **overrides) -> str:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({**MINIMAL, **overrides}), encoding="utf-8")
    return str(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "nope.yml")


@pytest.mark.parametrize("missing", ["model_id", "device"])
def test_missing_required_key_raises(tmp_path, missing):
    payload = {k: v for k, v in MINIMAL.items() if k != missing}
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Missing required config keys: {missing}"):
        load_config(path)


def test_defaults_are_applied(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config["default_backend"] == "pipeline"
    assert config["long_form_mode"] == "auto"
    assert config["chunk_length_s"] == 30.0
    assert config["stride_length_s"] == 5.0
    assert config["max_audio_seconds"] == 600
    assert config["max_upload_mb"] == 50
    assert config["max_concurrent_inferences"] == 1
    assert config["cors_allow_origins"] == ["*"]


def test_explicit_values_survive_defaults(tmp_path):
    config = load_config(write_config(tmp_path, max_audio_seconds=30, chunk_length_s=15))
    assert config["max_audio_seconds"] == 30
    assert config["chunk_length_s"] == 15


@pytest.mark.parametrize(
    ("key", "bad", "allowed"),
    [
        ("default_backend", "sideways", "'pipeline', 'direct'"),
        ("device", "tpu", "'cpu', 'cuda'"),
        ("long_form_mode", "maybe", "'auto', 'off'"),
    ],
)
def test_invalid_enum_value_raises(tmp_path, key, bad, allowed):
    """A mistyped value must fail loudly rather than silently defaulting."""
    with pytest.raises(ValueError, match="Invalid config value") as excinfo:
        load_config(write_config(tmp_path, **{key: bad}))
    assert allowed in str(excinfo.value)


def test_deprecated_calibration_key_is_migrated(tmp_path):
    """The old misspelling still works, with a warning."""
    path = write_config(tmp_path, enable_callibration=True)
    with pytest.warns(DeprecationWarning, match="enable_callibration"):
        config = load_config(path)
    assert config["enable_calibration"] is True
    assert "enable_callibration" not in config


def test_correct_calibration_key_does_not_warn(tmp_path, recwarn):
    load_config(write_config(tmp_path, enable_calibration=True))
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_shipped_configs_load():
    """Both configs in the repo must be valid."""
    for path in ("configs/default.yml", "configs/cpu.yml"):
        config = load_config(path)
        assert config["model_id"]
        assert config["device"] in ("cpu", "cuda")

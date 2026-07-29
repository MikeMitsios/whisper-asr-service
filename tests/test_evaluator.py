"""Evaluation metrics, text normalization, and CSV schema."""

from __future__ import annotations

import numpy as np
import pytest

from whisper_asr.evaluator import Evaluator
from whisper_asr.text_normalization import build_normalizer

SR = 16_000


class _ScriptedTranscriber:
    """Returns a queued hypothesis per call, so metrics are deterministic."""

    def __init__(self, hypotheses: list[str]) -> None:
        self._hypotheses = list(hypotheses)
        self.backend = "direct"

    def transcribe(self, audio: np.ndarray, sampling_rate: int) -> str:
        return self._hypotheses.pop(0)


def _samples(pairs: list[tuple[str, str]], seconds: float = 2.0):
    """Build (audio, sr, reference) tuples with matching hypotheses."""
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    return [(audio, SR, reference) for reference, _ in pairs]


# -- normalization -----------------------------------------------------------


def test_english_normalizer_ignores_case_and_punctuation():
    normalize = build_normalizer("en")
    assert normalize("Hello, World!").strip() == normalize("hello world").strip()


def test_english_normalizer_expands_abbreviations():
    """The full normalizer also harmonises honorifics and contractions."""
    normalize = build_normalizer("en")
    assert normalize("Mr. Smith arrived.").strip() == normalize("mister smith arrived").strip()


def test_basic_normalizer_used_for_non_english():
    normalize = build_normalizer("de")
    assert normalize("Guten TAG!").strip() == normalize("guten tag").strip()


def test_normalizer_tolerates_missing_spelling_map():
    """A tokenizer that cannot be loaded must degrade, not raise."""
    normalize = build_normalizer("en", model_id="does/not/exist-xyz")
    assert normalize("HELLO.").strip() == "hello"


# -- metrics -----------------------------------------------------------------


def test_perfect_transcription_scores_zero_error():
    # At least five words: sacrebleu scores 4-grams, so a shorter sentence
    # cannot reach 100 even when it matches exactly.
    pairs = [("hello there my old friend", "hello there my old friend")]
    evaluator = Evaluator(_ScriptedTranscriber([h for _, h in pairs]))
    metrics = evaluator.evaluate(_samples(pairs))
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0
    assert metrics["bleu"] == pytest.approx(100.0)


def test_normalized_wer_forgives_case_and_punctuation():
    """The headline number: raw WER punishes formatting, normalized does not."""
    pairs = [("Hello, World! Nice day.", "hello world nice day")]
    evaluator = Evaluator(
        _ScriptedTranscriber([h for _, h in pairs]),
        normalizer=build_normalizer("en"),
    )
    metrics = evaluator.evaluate(_samples(pairs))
    assert metrics["wer"] > 0.0
    assert metrics["wer_normalized"] == 0.0


def test_normalized_metrics_absent_without_a_normalizer():
    pairs = [("hello", "hello")]
    metrics = Evaluator(_ScriptedTranscriber(["hello"])).evaluate(_samples(pairs))
    assert "wer_normalized" not in metrics
    assert "cer_normalized" not in metrics


def test_empty_reference_after_normalization_is_dropped():
    """A reference that normalizes to nothing would divide by zero in jiwer.

    "!!!" normalizes to the empty string; note that "..." does not -- it keeps a
    single period -- so the guard has to test the normalized form, not the raw.
    """
    pairs = [("!!!", "anything"), ("hello world", "hello world")]
    evaluator = Evaluator(
        _ScriptedTranscriber([h for _, h in pairs]),
        normalizer=build_normalizer("en"),
    )
    metrics = evaluator.evaluate(_samples(pairs))
    assert metrics["wer_normalized"] == 0.0


def test_latency_percentiles_and_rtf_are_reported():
    pairs = [("a b", "a b"), ("c d", "c d"), ("e f", "e f")]
    evaluator = Evaluator(_ScriptedTranscriber([h for _, h in pairs]))
    metrics = evaluator.evaluate(_samples(pairs, seconds=4.0))

    for key in ("avg_time_s", "p50_time_s", "p95_time_s", "total_time_s", "rtf"):
        assert key in metrics, key
        assert metrics[key] >= 0.0
    assert metrics["p95_time_s"] >= metrics["p50_time_s"]
    assert metrics["total_time_s"] >= metrics["avg_time_s"]
    # 3 samples x 4s of audio, transcribed instantly, so RTF must be tiny.
    assert metrics["rtf"] < 1.0


def test_empty_sample_list_raises():
    with pytest.raises(ValueError, match="No samples to evaluate"):
        Evaluator(_ScriptedTranscriber([])).evaluate([])


# -- CSV schema --------------------------------------------------------------


def test_every_metric_key_has_a_csv_column():
    """A metric the header does not know about would be silently dropped."""
    import scripts.evaluation as evaluation

    pairs = [("hello world", "hello world")]
    evaluator = Evaluator(
        _ScriptedTranscriber(["hello world"]),
        normalizer=build_normalizer("en"),
    )
    metrics = evaluator.evaluate(_samples(pairs))
    metrics["model_size_mb"] = 1.0

    unknown = set(metrics) - set(evaluation.CSV_HEADER)
    assert not unknown, f"metrics missing from CSV_HEADER: {sorted(unknown)}"


def test_save_metrics_rejects_unknown_keys(tmp_path):
    import scripts.evaluation as evaluation

    with pytest.raises(ValueError, match="Metrics not in CSV_HEADER"):
        evaluation.save_metrics(tmp_path / "out.csv", "model", 1, {"nonsense_metric": 1.0})


def test_save_metrics_writes_header_once(tmp_path):
    import scripts.evaluation as evaluation

    path = tmp_path / "out.csv"
    evaluation.save_metrics(path, "model-a", 5, {"wer": 0.1})
    evaluation.save_metrics(path, "model-b", 5, {"wer": 0.2})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("model_name,num_samples")
    assert lines[1].startswith("model-a")
    assert lines[2].startswith("model-b")


def test_shipped_results_csv_has_no_mislabelled_dtype():
    """float8 was never a supported dtype; those rows actually ran float32."""
    from pathlib import Path

    text = Path("evaluation_results.csv").read_text(encoding="utf-8")
    assert "float8_quantized" not in text
    assert "_float8," not in text

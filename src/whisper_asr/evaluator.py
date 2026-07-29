"""ASR evaluation: quality metrics, latency percentiles, and real-time factor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import sacrebleu
from jiwer import cer, wer
from tqdm import tqdm

from whisper_asr.text_normalization import Normalizer
from whisper_asr.transcriber import Transcriber


@dataclass
class EvalResult:
    """Container for per-sample evaluation data."""

    references: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    audio_lengths: list[float] = field(default_factory=list)


class Evaluator:
    """Evaluate ASR quality and speed given a Transcriber."""

    def __init__(self, transcriber: Transcriber, normalizer: Normalizer | None = None) -> None:
        """
        Args:
            transcriber: The transcriber under test.
            normalizer: Text normalizer used for the normalized metrics. When
                omitted, only raw metrics are reported.
        """
        self._transcriber = transcriber
        self._normalizer = normalizer

    def evaluate(
        self,
        samples: list[tuple[np.ndarray, int, str]],
    ) -> dict[str, float]:
        """Run transcription on *samples* and return aggregated metrics.

        Args:
            samples: List of ``(audio_array, sampling_rate, ground_truth)`` tuples.

        Returns:
            Dict with keys ``wer``, ``cer``, ``bleu``, ``wer_normalized``,
            ``cer_normalized``, ``avg_time_s``, ``p50_time_s``, ``p95_time_s``,
            ``total_time_s`` and ``rtf``.

        Raises:
            ValueError: If *samples* is empty.
        """
        if not samples:
            raise ValueError("No samples to evaluate.")

        result = EvalResult()

        for audio, sr, reference in tqdm(samples):
            audio_duration = len(audio) / sr

            start = time.perf_counter()
            hypothesis = self._transcriber.transcribe(audio, sr)
            elapsed = time.perf_counter() - start

            result.references.append(reference)
            result.hypotheses.append(hypothesis)
            result.durations.append(elapsed)
            result.audio_lengths.append(audio_duration)

        return self._compute_metrics(result)

    def _compute_metrics(self, result: EvalResult) -> dict[str, float]:
        total_time = sum(result.durations)
        total_audio = sum(result.audio_lengths)
        bleu = sacrebleu.corpus_bleu(result.hypotheses, [result.references])

        metrics = {
            "wer": wer(result.references, result.hypotheses),
            "cer": cer(result.references, result.hypotheses),
            "bleu": bleu.score,
            "avg_time_s": total_time / len(result.durations),
            # Percentiles as well as the mean: a serving metric needs the tail.
            "p50_time_s": float(np.percentile(result.durations, 50)),
            "p95_time_s": float(np.percentile(result.durations, 95)),
            "total_time_s": total_time,
            # RTF < 1 means faster than real time.
            "rtf": total_time / total_audio if total_audio > 0 else float("inf"),
        }
        metrics.update(self._normalized_metrics(result))
        return metrics

    def _normalized_metrics(self, result: EvalResult) -> dict[str, float]:
        """WER and CER after Whisper's text normalization.

        Pairs whose reference normalizes to nothing are dropped -- jiwer divides
        by the reference word count, so an empty reference raises.
        """
        if self._normalizer is None:
            return {}

        pairs = [
            (ref, hyp)
            for ref, hyp in (
                (self._normalizer(raw_ref), self._normalizer(raw_hyp))
                for raw_ref, raw_hyp in zip(
                    result.references, result.hypotheses, strict=True
                )
            )
            if ref.strip()
        ]
        if not pairs:
            return {}

        references = [ref for ref, _ in pairs]
        hypotheses = [hyp for _, hyp in pairs]
        return {
            "wer_normalized": wer(references, hypotheses),
            "cer_normalized": cer(references, hypotheses),
        }

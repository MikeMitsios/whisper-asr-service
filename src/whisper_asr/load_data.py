"""Helpers for loading samples from the Emilia-Dataset."""

import itertools
import json

from datasets import Audio, load_dataset

from whisper_asr.audio_utils import load_audio_from_mp3_bytes, resample_audio

EMILIA_LANGUAGES = {"en": "EN", "zh": "ZH", "de": "DE", "fr": "FR", "ja": "JA", "ko": "KO"}


def load_emilia_samples(
    language: str,
    num_samples: int,
    *,
    sampling_rate: int | None = 16000,
    subset: str | None = None,
) -> list[dict]:
    """Stream Emilia-Dataset and return sample dicts.

    Each dict contains: ``audio``, ``sr``, ``ground_truth``, ``mp3_bytes``.
    Samples without mp3 data are skipped automatically.

    Parameters
    ----------
    sampling_rate : int | None
        If provided, every audio sample is resampled to this rate before
        being returned.  The ``sr`` field in each dict will reflect the
        new rate.
    """
    if subset:
        path = f"{subset}/*.tar"
    else:
        code = EMILIA_LANGUAGES[language]
        path = f"Emilia/{code}/*.tar"

    dataset = load_dataset(
        "amphion/Emilia-Dataset",
        data_files={"data": path},
        split="data",
        streaming=True,
    )
    dataset = dataset.cast_column("mp3", Audio(decode=False))

    samples = []
    for sample in itertools.islice(iter(dataset), num_samples):
        mp3_bytes = sample.get("mp3", {}).get("bytes")
        if mp3_bytes is None:
            continue

        meta = sample.get("json", {})
        if isinstance(meta, (str, bytes)):
            try:
                meta = json.loads(meta) if isinstance(meta, str) else json.loads(meta.decode())
            except Exception:
                meta = {}

        audio, sr = load_audio_from_mp3_bytes(mp3_bytes)

        if sampling_rate is not None and sr != sampling_rate:
            audio = resample_audio(audio, sr, sampling_rate)
            sr = sampling_rate

        samples.append({
            "audio": audio,
            "sr": sr,
            "ground_truth": meta.get("text", ""),
            "mp3_bytes": mp3_bytes,
        })

    return samples

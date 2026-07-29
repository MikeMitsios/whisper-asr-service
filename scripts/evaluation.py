"""Evaluate Whisper ASR on Emilia-Dataset and append metrics to a CSV.

Usage:
    python scripts/evaluation.py --num-samples 50 --language en
    python scripts/evaluation.py --config configs/cpu.yml --num-samples 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from whisper_asr.config import load_config
from whisper_asr.evaluator import Evaluator
from whisper_asr.load_data import EMILIA_LANGUAGES, load_emilia_samples
from whisper_asr.quantization import (
    format_size,
    get_model_memory_bytes,
    print_param_dtypes,
)
from whisper_asr.text_normalization import build_normalizer
from whisper_asr.transcriber import Transcriber

project_root = Path(__file__).resolve().parent.parent

# New columns are appended, never inserted, so rows written by earlier versions
# stay parseable.
CSV_HEADER = [
    "model_name",
    "num_samples",
    "wer",
    "cer",
    "bleu",
    "avg_time_s",
    "total_time_s",
    "rtf",
    "wer_normalized",
    "cer_normalized",
    "p50_time_s",
    "p95_time_s",
    "model_size_mb",
]


def build_model_name(config: dict) -> str:
    """Build a descriptive model name from config (includes quantization info)."""
    name = config["model_id"]
    qcfg = config.get("quantized_config")
    torch_dtype = config.get("torch_dtype", "float32")
    name += f"_{torch_dtype}"
    if qcfg and qcfg.get("weights"):
        name += f"_quantized_{qcfg['weights']}_{qcfg['activations']}"
    return name


def save_metrics(path: Path, model_name: str, num_samples: int, metrics: dict) -> None:
    """Append one row of metrics to *path* (creates file with header if needed)."""
    write_header = not path.exists()
    row = {"model_name": model_name, "num_samples": num_samples, **metrics}
    unknown = set(row) - set(CSV_HEADER)
    if unknown:
        raise ValueError(f"Metrics not in CSV_HEADER: {', '.join(sorted(unknown))}")

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Whisper ASR on Emilia-Dataset")
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "default.yml"),
        help="Path to the YAML config (default: configs/default.yml)",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--language", choices=list(EMILIA_LANGUAGES.keys()), default="en")
    parser.add_argument("--backend", choices=["pipeline", "direct"], default=None)
    parser.add_argument("--output", default="evaluation_results.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.backend:
        config["default_backend"] = args.backend

    model_name = build_model_name(config)
    print(f"Model: {model_name}")
    print(f"Backend: {config['default_backend']}")

    transcriber = Transcriber(config)

    print("\n  Model parameter dtypes:")
    print_param_dtypes(transcriber.torch_model)
    model_bytes = get_model_memory_bytes(transcriber.torch_model)
    print(f"\n  In-memory size: {format_size(model_bytes)}")

    print(f"\nLoading {args.num_samples} samples (language={args.language}) ...")
    raw = load_emilia_samples(args.language, args.num_samples)
    samples = [(s["audio"], s["sr"], s["ground_truth"]) for s in raw if s["ground_truth"]]
    print(f"Collected {len(samples)} valid samples.")

    if not samples:
        print("No valid samples found. Exiting.")
        return

    normalizer = build_normalizer(args.language, model_id=config["model_id"])
    evaluator = Evaluator(transcriber, normalizer=normalizer)
    metrics = evaluator.evaluate(samples)

    # Recorded so the size/latency trade-off of a quantization run is on the
    # same row as its quality numbers.
    metrics["model_size_mb"] = round(model_bytes / (1024 * 1024), 2)

    print("\n--- Evaluation Results ---")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    output_path = Path(args.output)
    save_metrics(output_path, model_name, len(samples), metrics)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

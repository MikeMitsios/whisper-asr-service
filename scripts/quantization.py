"""Quantize a Whisper model and report the size reduction.

Usage:
    python scripts/quantization.py
    python scripts/quantization.py --model-id openai/whisper-large-v3 --weights int8
    python scripts/quantization.py --save --output-dir ./models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoProcessor

from whisper_asr.quantization import (
    compare_sizes,
    format_size,
    get_dir_size_bytes,
    get_model_memory_bytes,
    load_original_model,
    print_param_dtypes,
    quantize_model,
    save_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize a Whisper model and compare sizes")
    parser.add_argument(
        "--model-id",
        default="openai/whisper-large-v3",
        help="HuggingFace model ID (default: openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--weights",
        default="int4",
        choices=["int2", "int4", "int8"],
        help="Weight quantization type (default: int4)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load the model on (default: cpu)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write both models to disk and compare on-disk sizes as well. "
        "This needs several GB of free space.",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Root directory for saved models, with --save (default: ./models)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    original_dir = output_root / "original"
    quantized_dir = output_root / "quantized"

    model = load_original_model(args.model_id, device=args.device)
    processor = AutoProcessor.from_pretrained(args.model_id)

    print("\n  Original model parameter dtypes:")
    print_param_dtypes(model)
    mem_original = get_model_memory_bytes(model)
    print(f"\n  In-memory size (original): {format_size(mem_original)}")

    if args.save:
        print(f"\n  Saving original model to {original_dir} ...")
        save_model(model, processor, original_dir)
        disk_original = get_dir_size_bytes(original_dir)
        print(f"  Saved. Disk size: {format_size(disk_original)}")

    model = quantize_model(model, weights=args.weights)

    print("\n  Quantized model parameter dtypes:")
    print_param_dtypes(model)
    mem_quantized = get_model_memory_bytes(model)
    print(f"\n  In-memory size (quantized): {format_size(mem_quantized)}")

    print("\n  In-memory comparison:")
    smaller = compare_sizes(mem_original, mem_quantized)

    if args.save:
        print(f"\n  Saving quantized model to {quantized_dir} ...")
        save_model(model, processor, quantized_dir)
        disk_quantized = get_dir_size_bytes(quantized_dir)
        print(f"  Saved. Disk size: {format_size(disk_quantized)}")
        print("\n  On-disk comparison:")
        smaller = compare_sizes(disk_original, disk_quantized) and smaller

    sys.exit(0 if smaller else 1)


if __name__ == "__main__":
    main()

"""Quantization helpers for Whisper models (optimum-quanto)."""

from __future__ import annotations

from pathlib import Path

import torch
from optimum.quanto import freeze, qint2, qint4, qint8, quantize
from optimum.quanto.tensor import QTensor
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# ---------------------------------------------------------------------------
# Size / memory helpers
# ---------------------------------------------------------------------------

def get_dir_size_bytes(path: Path) -> int:
    """Return total size in bytes of all files under *path*."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def _tensor_memory_bytes(t: torch.Tensor) -> int:
    """Return the true in-memory size of a tensor, accounting for quantized storage.

    Quanto's quantized tensors (QBytesTensor for int8/float8, QBitsTensor for
    int4/int2) are ``torch.Tensor`` wrapper-subclasses whose ``.dtype`` /
    ``.element_size()`` reflect the *dequantized* output, not the actual
    storage.  We use ``__tensor_flatten__`` to reach the real inner tensors
    (``_data``, ``_scale``, ``_shift``, ...) and sum their sizes instead.
    """
    if isinstance(t, QTensor):
        inner_names, _ = t.__tensor_flatten__()
        total = 0
        for name in inner_names:
            inner = getattr(t, name)
            total += _tensor_memory_bytes(inner)  # recurse for nested subclasses
        return total
    return t.nelement() * t.element_size()


def get_model_memory_bytes(model: torch.nn.Module) -> int:
    """Estimate in-memory size from parameter + buffer tensors."""
    total = 0
    for p in model.parameters():
        total += _tensor_memory_bytes(p)
    for b in model.buffers():
        total += _tensor_memory_bytes(b)
    return total


# ---------------------------------------------------------------------------
# Dtype inspection
# ---------------------------------------------------------------------------

def _param_dtype_str(param: torch.Tensor) -> str:
    """Return a human-readable dtype string, aware of all quanto QTensor variants."""
    if isinstance(param, QTensor):
        inner_names, _ = param.__tensor_flatten__()
        parts = ", ".join(f"{n}={getattr(param, n).dtype}" for n in inner_names)
        return f"{param.qtype} ({parts})"
    return str(param.dtype)


def print_param_dtypes(model: torch.nn.Module, limit: int = 10) -> None:
    """Print the dtype of the first *limit* parameters."""
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f"  {name}: {_param_dtype_str(param)}")
        if idx + 1 >= limit:
            print(f"  ... ({sum(1 for _ in model.parameters()) - limit} more parameters)")
            break


# ---------------------------------------------------------------------------
# Core quantization logic
# ---------------------------------------------------------------------------

_WEIGHT_TYPES = {"int2": qint2, "int4": qint4, "int8": qint8}


def load_original_model(
    model_id: str,
    device: str = "cpu",
) -> AutoModelForSpeechSeq2Seq:
    """Load the original (fp32) Whisper model."""
    print(f"\n  Loading original model: {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
    )
    return model


def save_model(
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    save_dir: Path,
) -> Path:
    """Save model and processor to *save_dir* and return the path."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    return save_dir


def quantize_model(
    model: AutoModelForSpeechSeq2Seq,
    weights: str = "int4",
) -> AutoModelForSpeechSeq2Seq:
    """Quantize the model's weights in-place using optimum-quanto.

    Args:
        model: The model to quantize.
        weights: Weight quantization type -- ``"int2"``, ``"int4"`` or
            ``"int8"``.

    Returns:
        The same model object, quantized and frozen.

    Raises:
        ValueError: If *weights* is not a recognised quantization type.
    """
    if weights not in _WEIGHT_TYPES:
        expected = ", ".join(repr(k) for k in _WEIGHT_TYPES)
        raise ValueError(f"Unsupported weights type {weights!r}. Expected one of: {expected}.")

    print(f"\n  Quantizing model weights to {weights} ...")
    quantize(model, weights=_WEIGHT_TYPES[weights], activations=None)
    freeze(model)
    return model


def compare_sizes(original_bytes: int, quantized_bytes: int, label: str = "Model size") -> bool:
    """Print a comparison table and return True if quantized is smaller."""
    reduction = original_bytes - quantized_bytes
    pct = (reduction / original_bytes) * 100 if original_bytes else 0

    print("\n" + "=" * 60)
    print(f"  {label} comparison")
    print("=" * 60)
    print(f"  Original  : {format_size(original_bytes)}")
    print(f"  Quantized : {format_size(quantized_bytes)}")
    print(f"  Reduction : {format_size(reduction)} ({pct:.1f}%)")
    print("=" * 60)

    is_smaller = quantized_bytes < original_bytes
    if is_smaller:
        print("  SUCCESS - quantized model is smaller than the original.")
    else:
        print("  FAIL - quantized model is NOT smaller. Check configuration.")
    print("=" * 60 + "\n")
    return is_smaller

"""Shared validation and audio-loading helpers for ASR routers."""

import numpy as np
from fastapi import HTTPException, Request, UploadFile

from whisper_asr.audio_utils import load_audio_from_bytes, resample_audio

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mpga", ".webm"}

_CANONICAL_SR = 16_000
_BYTES_PER_MB = 1024 * 1024


def check_file_extension(filename: str) -> bool:
    """Return *True* if ``filename`` has a supported audio extension."""
    if not filename:
        return False
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def to_canonical_pcm16(audio: np.ndarray, sampling_rate: int) -> bytes:
    """Convert float32 audio to canonical 16-bit PCM bytes at 16 kHz.

    Used for cache-key hashing: the same samples hash the same way whatever
    container and sample rate they arrived in, so a WAV and a lossless FLAC of
    one recording share a cache entry.

    Two files that *sound* identical but hold different samples -- a lossy
    re-encode, or the same float input quantized by two different encoders --
    hash differently. That is correct: they are different audio, and the
    transcript may legitimately differ.
    """
    if sampling_rate != _CANONICAL_SR:
        audio = resample_audio(audio, sampling_rate, _CANONICAL_SR)
    pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return pcm16.tobytes()


async def read_and_decode_audio(
    file: UploadFile,
    request: Request,
) -> tuple[np.ndarray, int]:
    """
    Validate, read, and decode an uploaded audio file.

    Enforces the configured upload-size and duration limits so a large file
    cannot exhaust memory or occupy the model indefinitely.

    Raises:
        HTTPException: 400 for an invalid extension, empty body, or a decode
            failure; 413 when the upload or its duration exceeds the limit.

    Returns:
        Tuple of (audio_array, sampling_rate).
    """
    config = request.app.state.config
    max_upload_bytes = int(config["max_upload_mb"] * _BYTES_PER_MB)
    max_audio_seconds = config["max_audio_seconds"]

    if not check_file_extension(file.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Reject on the declared length first so an oversized body is refused
    # before it is buffered.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload too large: {int(declared) / _BYTES_PER_MB:.1f} MB "
                f"exceeds the {config['max_upload_mb']} MB limit."
            ),
        )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}") from e

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Check again on the real byte count: content-length covers the whole
    # multipart body and can be absent or wrong.
    if len(contents) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload too large: {len(contents) / _BYTES_PER_MB:.1f} MB "
                f"exceeds the {config['max_upload_mb']} MB limit."
            ),
        )

    try:
        audio_array, sampling_rate = load_audio_from_bytes(contents)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio file: {e}",
        ) from e

    duration_s = len(audio_array) / sampling_rate if sampling_rate else 0.0
    if duration_s > max_audio_seconds:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio too long: {duration_s:.1f}s exceeds the "
                f"{max_audio_seconds}s limit."
            ),
        )

    return audio_array, sampling_rate

"""Audio loading and resampling utilities."""

import io

import numpy as np
import soundfile as sf

TARGET_SAMPLING_RATE: int = 16000

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torch
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_audio_from_bytes(bytes_: bytes) -> tuple[np.ndarray, int]:
    """
    Load audio from raw bytes. Supports WAV, MP3, FLAC, OGG, M4A.

    Returns:
        Tuple of (audio_array, sampling_rate). Audio is float32 in [-1, 1].
    """
    buffer = io.BytesIO(bytes_)
    audio_array, sampling_rate = sf.read(buffer, dtype="float32", always_2d=False)
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    return audio_array, sampling_rate


def load_audio_from_mp3_bytes(bytes_: bytes) -> tuple[np.ndarray, int]:
    """
    Load audio from MP3 bytes (e.g., Emilia dataset samples).
    Uses soundfile; for MP3 may require librosa fallback on some systems.

    Returns:
        Tuple of (audio_array, sampling_rate). Audio is float32 in [-1, 1].
    """
    try:
        return load_audio_from_bytes(bytes_)
    except Exception:
        if HAS_LIBROSA:
            buffer = io.BytesIO(bytes_)
            audio_array, sampling_rate = librosa.load(buffer, sr=None, mono=True)
            return audio_array.astype(np.float32), int(sampling_rate)
        raise


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample audio to ``target_sr`` Hz (default 16 kHz). Returns float32 numpy array."""
    if orig_sr == target_sr:
        return audio
    if HAS_TORCHAUDIO:
        tensor = torch.from_numpy(audio).float()
        resampled = torchaudio.functional.resample(
            tensor, orig_freq=orig_sr, new_freq=target_sr
        )
        return resampled.numpy()
    if HAS_LIBROSA:
        return librosa.resample(
            audio.astype(np.float64),
            orig_sr=orig_sr,
            target_sr=target_sr,
        ).astype(np.float32)
    if HAS_SCIPY:
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = scipy_signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
    raise RuntimeError("Need torchaudio, librosa, or scipy for resampling")

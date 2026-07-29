"""Text normalization for ASR evaluation.

Raw WER punishes a model for punctuation, casing and spelling variants that a
human reader would not count as errors -- "The dog's tail." versus "the dogs
tail" scores three errors out of three words. Whisper's own reported numbers use
these normalizers, so reporting normalized WER alongside raw is what makes a
result comparable to published figures.
"""

from __future__ import annotations

from collections.abc import Callable

from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)

Normalizer = Callable[[str], str]


def build_normalizer(language: str = "en", model_id: str | None = None) -> Normalizer:
    """Return the appropriate Whisper text normalizer for *language*.

    English gets the full normalizer -- number words, contractions, British and
    American spelling, honorifics. Everything else gets the basic one, which
    lowercases and strips punctuation and symbols.

    Args:
        language: Two-letter language code.
        model_id: Model whose tokenizer supplies the English spelling map. Only
            consulted for English; falls back to an empty map if unavailable.

    Returns:
        A callable mapping raw text to normalized text.
    """
    if language.lower() not in {"en", "english"}:
        return BasicTextNormalizer()

    return EnglishTextNormalizer(_english_spelling_map(model_id))


def _english_spelling_map(model_id: str | None) -> dict[str, str]:
    """Load the tokenizer's British-to-American spelling map, or an empty one.

    Loading a tokenizer needs the network on a cold cache, so a failure here
    degrades to normalization without spelling harmonisation rather than
    breaking the whole evaluation.
    """
    if model_id is None:
        return {}
    try:
        from transformers import WhisperTokenizer

        tokenizer = WhisperTokenizer.from_pretrained(model_id)
    except Exception as exc:
        print(f"[normalizer] Spelling map unavailable ({exc}); continuing without it.")
        return {}
    return dict(getattr(tokenizer, "english_spelling_normalizer", {}) or {})

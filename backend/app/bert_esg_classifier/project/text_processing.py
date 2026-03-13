import re
from typing import List

import nltk


def _ensure_nltk_punkt() -> None:
    """
    Ensure that the NLTK sentence tokenizer data is available.
    This is called lazily so importing the module does not fail in environments
    where the resource has not yet been downloaded.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _clean_text(text: str) -> str:
    """
    Basic cleaning for raw ESG report text.

    - Normalize whitespace
    - Strip leading/trailing spaces
    """
    if not text:
        return ""

    # Replace all whitespace (newlines, tabs) with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Clean raw report text and split it into non-empty sentences.

    Args:
        text: Raw ESG report text.

    Returns:
        List of cleaned, non-empty sentences.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    _ensure_nltk_punkt()

    # Use NLTK's pretrained Punkt sentence tokenizer for robust splitting
    sentences = nltk.sent_tokenize(cleaned)

    # Strip whitespace and drop empty sentences
    sentences = [s.strip() for s in sentences if s and s.strip()]
    return sentences


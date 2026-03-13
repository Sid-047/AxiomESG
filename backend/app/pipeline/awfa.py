from __future__ import annotations

import re
from typing import Dict, List, Tuple


def _normalize(text: str) -> str:
    """
    Lightweight normalization used by the deterministic AWFA weighting.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _weight(sentence: str, category: str) -> float:
    """
    Heuristic sentence weighting based on length and simple ESG keywords.
    """
    base = 0.4
    length_bonus = min(len(sentence) / 200.0, 0.6)
    keyword_bonus = 0.0
    lowered = sentence.lower()
    if category == "E":
        keywords = ["emission", "carbon", "climate", "energy", "water", "waste"]
    elif category == "S":
        keywords = ["diversity", "inclusion", "safety", "labor", "community", "privacy"]
    else:
        keywords = ["governance", "board", "ethics", "compliance", "audit", "risk"]
    for kw in keywords:
        if kw in lowered:
            keyword_bonus += 0.1
    return round(min(base + length_bonus + keyword_bonus, 1.0), 3)


def apply_awfa(category_sentences: Dict[str, List[str]]) -> List[Tuple[str, str, float]]:
    """
    Deterministic AWFA v0 used by the main FastAPI pipeline.

    Args:
        category_sentences: mapping
            { "E": [...], "S": [...], "G": [...] }

    Returns:
        List of (category, sentence, weight) sorted by descending weight.
    """
    seen = set()
    weighted: List[Tuple[str, str, float]] = []
    for category, sentences in category_sentences.items():
        for sentence in sentences:
            key = _normalize(sentence)
            if not key or key in seen:
                continue
            seen.add(key)
            weight = _weight(sentence, category)
            weighted.append((category, sentence, weight))
    weighted.sort(key=lambda x: (-x[2], x[1]))
    return weighted
"""
Real AWFA (Axiom Weighting & Filtering Algorithm) — Research Variant
====================================================================

Implements TF-IDF-based uniqueness scoring, ESG keyword-density confidence
scoring, combined weighting, and Jaccard token-set deduplication.

This is the V4 variant — the main research contribution for demonstrating
that a principled evidence-selection algorithm outperforms naive heuristics.

API matches the existing strategy registry signature:
    (category_sentences: Dict[str, List[str]]) -> List[Tuple[str, str, float]]
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# ESG keyword dictionaries (category-specific)
# ---------------------------------------------------------------------------

ESG_KEYWORDS: Dict[str, List[str]] = {
    "E": [
        "emission", "emissions", "carbon", "co2", "greenhouse", "ghg",
        "climate", "energy", "renewable", "solar", "wind", "water",
        "waste", "recycling", "biodiversity", "pollution", "deforestation",
        "fossil", "methane", "scope 1", "scope 2", "scope 3", "tco2e",
        "intensity", "footprint", "sequestration", "offset",
    ],
    "S": [
        "diversity", "inclusion", "equity", "safety", "health", "labor",
        "employee", "workforce", "community", "human rights", "training",
        "turnover", "engagement", "wellbeing", "incident", "fatality",
        "trir", "ltifr", "gender", "minority", "wages", "benefits",
        "volunteer", "philanthropy", "privacy",
    ],
    "G": [
        "board", "governance", "ethics", "compliance", "audit", "risk",
        "shareholder", "transparency", "anti-corruption", "bribery",
        "whistleblower", "independence", "compensation", "oversight",
        "fiduciary", "disclosure", "policy", "regulation", "accountability",
        "committee", "charter",
    ],
}


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization, stripping punctuation."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _token_set(text: str) -> Set[str]:
    return set(_tokenize(text))


# ---------------------------------------------------------------------------
# TF-IDF-like Uniqueness Score
# ---------------------------------------------------------------------------

def _compute_uniqueness_scores(sentences: List[str]) -> List[float]:
    """
    Compute a uniqueness score for each sentence based on inverse document
    frequency of its tokens across the candidate pool.

    Higher score = sentence contains rarer tokens relative to the pool.
    """
    if not sentences:
        return []

    # Document frequency: how many sentences contain each token
    n = len(sentences)
    token_sets = [_token_set(s) for s in sentences]
    df: Counter = Counter()
    for ts in token_sets:
        for token in ts:
            df[token] += 1

    scores = []
    for ts in token_sets:
        if not ts:
            scores.append(0.0)
            continue
        # Average IDF across tokens in this sentence
        idf_sum = sum(math.log((n + 1) / (df[t] + 1)) + 1.0 for t in ts)
        score = idf_sum / len(ts)
        scores.append(score)

    # Normalize to [0, 1]
    if scores:
        s_min = min(scores)
        s_max = max(scores)
        if s_max > s_min:
            scores = [(s - s_min) / (s_max - s_min) for s in scores]
        else:
            scores = [0.5] * len(scores)

    return scores


# ---------------------------------------------------------------------------
# ESG Keyword-Density Confidence Score
# ---------------------------------------------------------------------------

def _compute_confidence_scores(
    sentences: List[str], categories: List[str]
) -> List[float]:
    """
    Compute confidence score based on ESG keyword density.
    Score is the fraction of tokens that are ESG keywords for the sentence's
    assigned category.
    """
    scores = []
    for sentence, category in zip(sentences, categories):
        tokens = _tokenize(sentence)
        if not tokens:
            scores.append(0.0)
            continue
        keywords = ESG_KEYWORDS.get(category, [])
        # Check for keyword presence (keywords can be multi-word)
        text_lower = sentence.lower()
        hit_count = sum(1 for kw in keywords if kw in text_lower)
        # Also count individual token hits
        token_hits = sum(1 for t in tokens if any(t in kw or kw in t for kw in keywords))
        # Combined density
        density = (hit_count * 2 + token_hits) / (len(tokens) + len(keywords))
        scores.append(min(density, 1.0))

    # Normalize to [0, 1]
    if scores:
        s_max = max(scores) if max(scores) > 0 else 1.0
        scores = [s / s_max for s in scores]

    return scores


# ---------------------------------------------------------------------------
# Jaccard Deduplication
# ---------------------------------------------------------------------------

def _jaccard_dedup(
    items: List[Tuple[str, str, float]],
    threshold: float = 0.8,
) -> Tuple[List[Tuple[str, str, float]], int]:
    """
    Remove near-duplicate sentences based on Jaccard token similarity.

    Args:
        items: List of (category, sentence, weight) sorted by descending weight
        threshold: Jaccard similarity threshold for considering two sentences as duplicates

    Returns:
        (deduplicated list, number of items removed)
    """
    kept: List[Tuple[str, str, float]] = []
    kept_token_sets: List[Set[str]] = []
    removed = 0

    for cat, sent, w in items:
        ts = _token_set(sent)
        is_dup = False
        for existing_ts in kept_token_sets:
            if not ts and not existing_ts:
                is_dup = True
                break
            if not ts or not existing_ts:
                continue
            intersection = ts & existing_ts
            union = ts | existing_ts
            jaccard = len(intersection) / len(union)
            if jaccard >= threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append((cat, sent, w))
            kept_token_sets.append(ts)

    return kept, removed


# ---------------------------------------------------------------------------
# Main Real AWFA function
# ---------------------------------------------------------------------------

def apply_real_awfa(
    category_sentences: Dict[str, List[str]],
    uniqueness_weight: float = 0.5,
    confidence_weight: float = 0.5,
    dedup_threshold: float = 0.8,
) -> List[Tuple[str, str, float]]:
    """
    Real AWFA: TF-IDF uniqueness + keyword density confidence + Jaccard dedup.

    This is the research-grade evidence selection algorithm.

    Args:
        category_sentences: {"E": [...], "S": [...], "G": [...]}
        uniqueness_weight: Weight for TF-IDF uniqueness score (default 0.5)
        confidence_weight: Weight for keyword density score (default 0.5)
        dedup_threshold: Jaccard similarity threshold for dedup (default 0.8)

    Returns:
        Sorted, deduplicated list of (category, sentence, weight) tuples.
    """
    # Flatten
    sentences: List[str] = []
    categories: List[str] = []
    for cat in ("E", "S", "G"):
        for s in category_sentences.get(cat, []):
            sentences.append(s)
            categories.append(cat)

    if not sentences:
        return []

    # Compute component scores
    uniqueness = _compute_uniqueness_scores(sentences)
    confidence = _compute_confidence_scores(sentences, categories)

    # Combine scores
    combined: List[Tuple[str, str, float]] = []
    for i, (sent, cat) in enumerate(zip(sentences, categories)):
        score = uniqueness_weight * uniqueness[i] + confidence_weight * confidence[i]
        score = round(min(max(score, 0.0), 1.0), 4)
        combined.append((cat, sent, score))

    # Sort by descending weight
    combined.sort(key=lambda x: (-x[2], x[1]))

    # Deduplicate
    deduped, removed_count = _jaccard_dedup(combined, threshold=dedup_threshold)

    return deduped


def get_dedup_stats(
    category_sentences: Dict[str, List[str]],
    dedup_threshold: float = 0.8,
) -> Dict[str, int]:
    """Return stats about deduplication for reporting."""
    total_input = sum(len(v) for v in category_sentences.values())

    sentences = []
    categories = []
    for cat in ("E", "S", "G"):
        for s in category_sentences.get(cat, []):
            sentences.append(s)
            categories.append(cat)

    uniqueness = _compute_uniqueness_scores(sentences)
    confidence = _compute_confidence_scores(sentences, categories)

    combined = []
    for i, (sent, cat) in enumerate(zip(sentences, categories)):
        score = 0.5 * uniqueness[i] + 0.5 * confidence[i]
        combined.append((cat, sent, round(score, 4)))

    combined.sort(key=lambda x: (-x[2], x[1]))
    _, removed = _jaccard_dedup(combined, threshold=dedup_threshold)

    return {
        "total_input": total_input,
        "total_after_dedup": total_input - removed,
        "removed_by_dedup": removed,
        "dedup_rate": round(removed / max(total_input, 1), 4),
    }

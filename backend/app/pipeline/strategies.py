"""
Strategy registry for ESG sentence weighting algorithms.

Each strategy accepts category_sentences: Dict[str, List[str]]
and returns List[Tuple[str, str, float]] sorted by descending weight.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

STRATEGY_META: Dict[str, Dict[str, str]] = {
    "heuristic": {
        "label": "Heuristic AWFA",
        "description": "Keyword-length heuristic weighting with deterministic dedup. No ML model required.",
    },
    "bert_mean": {
        "label": "BERT + Mean Fusion",
        "description": "BERT ESG classifier with empirical-prior weighted mean fusion.",
    },
    "bert_static": {
        "label": "BERT + Static Fusion",
        "description": "BERT ESG classifier with fixed E/S/G weights (0.5 / 0.3 / 0.2).",
    },
    "bert_awfa_v1": {
        "label": "BERT + AWFA v1",
        "description": "BERT ESG classifier with attention context network fusion.",
    },
    "bert_awfa_v2": {
        "label": "BERT + AWFA v2",
        "description": "BERT ESG classifier with multi-head attention + interaction MLP fusion.",
    },
}

WeightFn = Callable[[Dict[str, List[str]]], List[Tuple[str, str, float]]]

_registry: Dict[str, WeightFn] = {}


def register(name: str) -> Callable[[WeightFn], WeightFn]:
    """Decorator to register a weighting strategy."""
    def decorator(fn: WeightFn) -> WeightFn:
        _registry[name] = fn
        return fn
    return decorator


def get_strategy(name: str) -> WeightFn:
    """Look up a registered strategy by key. Raises ValueError for unknown keys."""
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return _registry[name]


def list_strategies() -> List[Dict[str, str]]:
    """Return metadata for all registered strategies."""
    result = []
    for key in STRATEGY_META:
        if key in _registry:
            meta = STRATEGY_META[key].copy()
            meta["key"] = key
            result.append(meta)
    return result


# Register heuristic strategy at import time
from app.pipeline.awfa import apply_awfa  # noqa: E402

register("heuristic")(apply_awfa)

from typing import Dict, List, Tuple

import torch
from torch import Tensor


def _vector_norm(vec: Tensor) -> float:
    """Return the L2 norm of a 1D tensor as a Python float."""
    return torch.norm(vec).item() if vec.numel() > 0 else 0.0


def _sentence_ranking(weights: Tensor, sentences: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
    """
    Build a simple importance ranking of sentences given their weights.

    Returns:
        List of (index, weight, sentence) sorted by descending weight.
    """
    if weights.numel() == 0 or not sentences:
        return []

    top_k = min(top_k, len(sentences))
    values, indices = torch.topk(weights, k=top_k)

    ranking: List[Tuple[int, float, str]] = []
    for idx, w in zip(indices.tolist(), values.tolist()):
        ranking.append((idx, float(w), sentences[idx]))
    return ranking


def evaluate_fusion_methods(
    sentences: List[str],
    mean_result: Tuple[Tensor, Tensor],
    static_result: Tuple[Tensor, Tensor],
    awfa_v1_result: Tuple[Tensor, Tensor],
    awfa_v2_result: Tuple[Tensor, Tensor],
) -> None:
    """
    Compare the outputs of all four fusion methods and print a
    human-readable comparison table.

    Each result tuple is:
        (fused_vector, weights)
    where:
        fused_vector: document representation
        weights:      sentence- or signal-level weights
    """
    method_stats: Dict[str, Dict[str, object]] = {}

    # Mean Fusion
    mean_vec, mean_weights = mean_result
    method_stats["Mean Fusion"] = {
        "avg_weight": float(mean_weights.mean().item()) if mean_weights.numel() else 0.0,
        "norm": _vector_norm(mean_vec),
        "weight_pattern": "uniform",
        "ranking": _sentence_ranking(mean_weights, sentences),
    }

    # Static Fusion
    static_vec, static_weights = static_result
    method_stats["Static Fusion"] = {
        "avg_weight": float(static_weights.mean().item()) if static_weights.numel() else 0.0,
        "norm": _vector_norm(static_vec),
        "weight_pattern": "ESG-static",
        "ranking": _sentence_ranking(static_weights, sentences),
    }

    # AWFA v1 (weights are over signals; we summarize them)
    awfa_v1_vec, awfa_v1_weights = awfa_v1_result
    method_stats["AWFA v1"] = {
        "avg_weight": float(awfa_v1_weights.mean().item()) if awfa_v1_weights.numel() else 0.0,
        "norm": _vector_norm(awfa_v1_vec.mean(dim=0) if awfa_v1_vec.ndim == 2 else awfa_v1_vec),
        "weight_pattern": "adaptive-signals",
        "ranking": [],  # sentence ranking is not directly defined for v1
    }

    # AWFA v2 (weights are over signals; we summarize them)
    awfa_v2_vec, awfa_v2_weights = awfa_v2_result
    method_stats["AWFA v2"] = {
        "avg_weight": float(awfa_v2_weights.mean().item()) if awfa_v2_weights.numel() else 0.0,
        "norm": _vector_norm(awfa_v2_vec.mean(dim=0) if awfa_v2_vec.ndim == 2 else awfa_v2_vec),
        "weight_pattern": "adaptive+confidence",
        "ranking": [],  # sentence ranking is driven by internal scores
    }

    # Print comparison table
    header = f"{'Method':<12} {'Avg Weight':>12} {'Vector Norm':>14} {'Weight Pattern':>18}"
    print(header)
    print("-" * len(header))

    for method, stats in method_stats.items():
        print(
            f"{method:<12} "
            f"{stats['avg_weight']:>12.4f} "
            f"{stats['norm']:>14.4f} "
            f"{stats['weight_pattern']:>18}"
        )

    print("\nSentence importance (top examples per method):\n")

    for method, stats in method_stats.items():
        ranking = stats["ranking"]
        if not ranking:
            continue
        print(f"{method}:")
        for idx, w, sent in ranking:
            print(f"  #{idx:03d}  weight={w:.4f}  text={sent[:120]!r}")
        print()


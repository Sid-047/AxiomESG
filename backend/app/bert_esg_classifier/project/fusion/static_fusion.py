from typing import List, Tuple

import torch
from torch import Tensor


class StaticFusion:
    """
    Static ESG-weighted fusion.

    Each sentence receives an importance weight computed as the dot
    product between its ESG probability vector and a fixed set of
    ESG weights:

        Environmental = 0.5
        Social        = 0.3
        Governance    = 0.2

    The final document representation is a weighted average of
    sentence embeddings using these importance scores.
    """

    def __init__(self) -> None:
        # Fixed ESG weights: [E, S, G]
        self.static_weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)

    def __call__(self, signals: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            signals:
                [0] embeddings: Tensor [num_sentences, feature_dim]
                [1] probabilities: Tensor [num_sentences, 3]
                Remaining entries are ignored by this strategy.

        Returns:
            fused_vector: Tensor [feature_dim] — ESG-weighted embedding.
            weights: Tensor [num_sentences] — normalized sentence weights.
        """
        if len(signals) < 2:
            raise ValueError(
                "StaticFusion expects at least [embeddings, probabilities] in signals."
            )

        embeddings, probabilities = signals[0], signals[1]

        if embeddings.numel() == 0:
            return (
                torch.zeros(embeddings.size(-1), dtype=embeddings.dtype),
                torch.empty(0, dtype=embeddings.dtype),
            )

        # Ensure ESG weights are on the same device / dtype as probabilities
        static_weights = self.static_weights.to(
            device=probabilities.device, dtype=probabilities.dtype
        )

        # sentence_weight = dot(probabilities, static_weights)
        # Shape: [num_sentences]
        raw_weights = torch.matmul(probabilities, static_weights)

        # Avoid division by zero if all weights are zero
        if torch.all(raw_weights == 0):
            normalized = torch.full_like(
                raw_weights, 1.0 / float(raw_weights.numel())
            )
        else:
            normalized = raw_weights / raw_weights.sum()

        # Weighted average of embeddings across sentences
        fused = torch.matmul(normalized, embeddings)  # [feature_dim]

        return fused, normalized


from typing import List, Tuple

import torch
from torch import Tensor


class MeanFusion:
    """
    Empirical-weighted mean fusion over sentence embeddings.

    Sentence importance is computed from ESG probabilities using an
    empirical class-prior distribution estimated from `final_esg_dataset.xlsx`.

    Empirical priors (normalized over E/S/G, excluding "Other"):
      - Environmental: 3058
      - Social:        3110
      - Governance:    4211
    """

    def __call__(self, signals: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            signals:
                [0] embeddings: Tensor [num_sentences, feature_dim]
                [1] probabilities: Tensor [num_sentences, 3]

        Returns:
            fused_vector: Tensor [feature_dim] — weighted document embedding.
            weights: Tensor [num_sentences] — normalized sentence importance weights.
        """
        if len(signals) < 2:
            raise ValueError("MeanFusion expects [embeddings, probabilities] in signals.")

        embeddings, probabilities = signals[0], signals[1]

        if embeddings.numel() == 0:
            # Empty document: return zero vector and empty weights
            return (
                torch.zeros(embeddings.size(-1), dtype=embeddings.dtype),
                torch.empty(0, dtype=embeddings.dtype),
            )

        if probabilities.shape[-1] != 3:
            raise ValueError(
                f"Expected 3-way ESG probabilities, got {probabilities.shape[-1]} classes."
            )

        # Empirical priors derived from dataset label distribution.
        priors = torch.tensor([3058.0, 3110.0, 4211.0], device=probabilities.device, dtype=probabilities.dtype)
        priors = priors / priors.sum()

        # sentence_weight = dot(probabilities, empirical_priors)
        raw_weights = torch.matmul(probabilities, priors)  # [num_sentences]

        if torch.all(raw_weights == 0):
            weights = torch.full_like(raw_weights, 1.0 / float(raw_weights.numel()))
        else:
            weights = raw_weights / raw_weights.sum()

        fused = torch.matmul(weights, embeddings)  # [feature_dim]

        return fused, weights


from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AWFAv1(nn.Module):
    """
    Attention-Weighted Fusion Architecture (v1).

    Learns attention weights over multiple input signals to produce
    a fused representation for each sentence.
    """

    def __init__(self, num_signals: int, feature_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.num_signals = num_signals
        self.feature_dim = feature_dim

        # Encodes global context from all signals
        self.context_network = nn.Sequential(
            nn.Linear(num_signals * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Generates attention weights for each signal
        self.attention_layer = nn.Linear(hidden_dim, num_signals)

    def forward(self, signals: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            signals: List of tensors, each of shape [batch, feature_dim].
                     All signals must share the same batch size and feature_dim.

        Returns:
            fused:   Tensor [batch, feature_dim] — fused signal representation.
            weights: Tensor [batch, num_signals] — attention weights per signal.
        """
        if len(signals) != self.num_signals:
            raise ValueError(
                f"Expected {self.num_signals} signals, got {len(signals)} instead."
            )

        # Stack signals → shape: [batch, num_signals, feature_dim]
        stacked_signals = torch.stack(signals, dim=1)

        # Concatenate signals along the feature dimension → [batch, num_signals * feature_dim]
        context_vector = torch.cat(signals, dim=-1)

        # Extract contextual features
        context_features = self.context_network(context_vector)

        # Generate attention logits and normalize to weights
        attention_logits = self.attention_layer(context_features)  # [batch, num_signals]
        weights = F.softmax(attention_logits, dim=-1)

        # Weighted fusion of signals
        fused = torch.sum(weights.unsqueeze(-1) * stacked_signals, dim=1)

        return fused, weights


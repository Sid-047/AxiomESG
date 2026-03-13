from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AWFAv2(nn.Module):
    """
    Extended Attention-Weighted Fusion Architecture (v2).

    Matches the reference training notebook: Attention-based adaptive fusion
    using MultiheadAttention over embedded signals, followed by an interaction
    MLP and a weight generator to produce signal weights.
    """

    def __init__(
        self,
        num_signals: int,
        feature_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()

        self.num_signals = num_signals
        self.feature_dim = feature_dim

        self.signal_embed = nn.Linear(feature_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.weight_generator = nn.Sequential(
            nn.Linear(num_signals * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_signals),
        )

    def forward(self, signals: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            signals: List of tensors, each of shape [batch, feature_dim].

        Returns:
            fused:   Tensor [batch, hidden_dim] — fused representation.
            weights: Tensor [batch, num_signals] — attention weights per signal.
        """
        if len(signals) != self.num_signals:
            raise ValueError(
                f"Expected {self.num_signals} signals, got {len(signals)} instead."
            )

        # x: [batch, num_signals, feature_dim]
        x = torch.stack(signals, dim=1)

        # Embed signals: [batch, num_signals, hidden_dim]
        x = self.signal_embed(x)

        # Self-attention across signals
        attn_output, _ = self.attention(x, x, x)  # [batch, num_signals, hidden_dim]

        # Interaction modeling
        interaction = self.interaction(attn_output)  # [batch, num_signals, hidden_dim]

        # Flatten context across signals
        context = interaction.reshape(interaction.shape[0], -1)  # [batch, num_signals * hidden_dim]

        # Generate signal weights
        logits = self.weight_generator(context)  # [batch, num_signals]
        weights = F.softmax(logits, dim=-1)

        # Fuse attended signal representations
        fused = torch.sum(weights.unsqueeze(-1) * attn_output, dim=1)  # [batch, hidden_dim]

        return fused, weights


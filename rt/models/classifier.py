"""
High-Reward Binary Classifier.

Trained to predict whether the current timestep has a reward above the median
of the offline dataset. Used during backward generation to select the best
(R_t, s_t) candidates (highest P(H=1|R_t, s_t)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HighRewardClassifier(nn.Module):
    """
    Binary classifier: P(H=1 | R_t, s_t)

    H=1 means the reward at this timestep is above the dataset median.

    Input: concatenation of [R_t (scalar), s_t (state_dim)]
    Output: logit for P(H=1)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        input_dim = 1 + state_dim  # RTG (scalar) + state

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rtg: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rtg:    (B, 1) or (B,) return-to-go
            states: (B, state_dim)
        Returns:
            logits: (B, 1)
        """
        if rtg.dim() == 1:
            rtg = rtg.unsqueeze(-1)
        x = torch.cat([rtg, states], dim=-1)
        return self.net(x)

    def predict_proba(self, rtg: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Returns P(H=1|R_t, s_t) as a probability in [0, 1]. Shape: (B,)"""
        logits = self.forward(rtg, states)
        return torch.sigmoid(logits).squeeze(-1)

    def loss(
        self,
        rtg: torch.Tensor,
        states: torch.Tensor,
        labels: torch.Tensor,  # (B,) float binary labels
    ) -> torch.Tensor:
        """Binary cross-entropy loss."""
        logits = self.forward(rtg, states).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, labels)

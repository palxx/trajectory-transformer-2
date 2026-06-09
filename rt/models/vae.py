"""
Transition VAE for reliability estimation.

Encodes (s, a, s') → latent z, then decodes back.
The reconstruction error on a given transition approximates the
world-model error DIST(P(·|s,a), P̂(·|s,a)) used in Eq. 3.

VAE Loss (Eq. 7):
  L(ψ, φ) = -E[log p(x|z)] + D_KL(q(z|x) || p(z))
  where x = (s, a, s')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransitionVAE(nn.Module):
    """
    Variational Autoencoder over (s, a, s') transitions.

    Encoder: (s, a, s') → (μ, log σ²)
    Decoder: z → (ŝ, â, ŝ')
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim + state_dim  # (s, a, s')

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, state_dim+action_dim+state_dim) → (μ, logvar)"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon: reconstruction of (s, a, s')
            mu, logvar: latent distribution parameters
            z: sampled latent
        """
        x = torch.cat([states, actions, next_states], dim=-1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes ELBO loss.
        Returns: total_loss, recon_loss, kl_loss
        """
        x = torch.cat([states, actions, next_states], dim=-1)
        recon, mu, logvar, _ = self.forward(states, actions, next_states)

        # Reconstruction loss (MSE = negative log-likelihood under Gaussian)
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def reconstruction_error(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-sample reconstruction error (MSE, summed over input dims).
        Shape: (B,)
        Used to approximate world-model error for reliability estimation.
        """
        x = torch.cat([states, actions, next_states], dim=-1)
        mu, logvar = self.encode(x)
        z = mu  # use mean (no noise) for deterministic error estimate
        recon = self.decode(z)
        # Per-sample mean squared error
        return F.mse_loss(recon, x, reduction="none").mean(dim=-1)

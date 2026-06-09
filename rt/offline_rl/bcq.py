"""
Batch-Constrained Q-learning (BCQ) for offline RL.

Reference: Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration" (ICML 2019)

Key components:
  - VAE to generate candidate actions conditioned on state
  - Perturbation network ξ to perturb candidates
  - Twin Q-networks with conservative selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from copy import deepcopy


class BCQ_VAE(nn.Module):
    """VAE for conditional action generation: p(a|s)"""

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 750):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Encoder: (s, a) → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (s, z) → a
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(state, action)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(state, z)
        return recon, mu, logvar

    def sample(self, state: torch.Tensor) -> torch.Tensor:
        """Sample action by decoding random z ~ N(0,I)."""
        z = torch.randn(state.shape[0], self.latent_dim, device=state.device)
        z = z.clamp(-0.5, 0.5)
        return self.decode(state, z)

    def loss(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        recon, mu, logvar = self.forward(state, action)
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.5 * kl_loss


class PerturbationNetwork(nn.Module):
    """
    Perturbation network ξ: (s, a) → Δa in [-φ, φ]
    Adds a small perturbation to a candidate action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 400, phi: float = 0.05):
        super().__init__()
        self.phi = phi
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return action + self.phi * self.net(x)


class BCQ_QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 400):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class BCQ:
    """
    Batch-Constrained Q-learning agent.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 750,
        lr: float = 1e-4,
        gamma: float = 0.99,
        phi: float = 0.05,
        lmbda: float = 0.75,
        polyak: float = 0.005,
        n_action_samples: int = 10,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.phi = phi
        self.lmbda = lmbda
        self.polyak = polyak
        self.n_action_samples = n_action_samples
        self.device = device
        self.action_dim = action_dim

        latent_dim = action_dim * 2

        # Networks
        self.vae = BCQ_VAE(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.perturb = PerturbationNetwork(state_dim, action_dim, hidden_dim=400, phi=phi).to(device)
        self.perturb_target = deepcopy(self.perturb).to(device)
        self.qf = BCQ_QNetwork(state_dim, action_dim, hidden_dim=400).to(device)
        self.qf_target = deepcopy(self.qf).to(device)

        # Optimizers
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.perturb_opt = torch.optim.Adam(self.perturb.parameters(), lr=lr)
        self.qf_opt = torch.optim.Adam(self.qf.parameters(), lr=lr)

        # Freeze targets
        for p in self.qf_target.parameters():
            p.requires_grad = False
        for p in self.perturb_target.parameters():
            p.requires_grad = False

    def update(self, batch: dict) -> Dict[str, float]:
        s = batch["observations"].to(self.device)
        a = batch["actions"].to(self.device)
        r = batch["rewards"].to(self.device).unsqueeze(-1)
        ns = batch["next_observations"].to(self.device)
        done = batch["terminals"].to(self.device).unsqueeze(-1)
        B = s.shape[0]

        # ---- VAE update ----
        vae_loss = self.vae.loss(s, a)
        self.vae_opt.zero_grad()
        vae_loss.backward()
        self.vae_opt.step()

        # ---- Q update ----
        with torch.no_grad():
            # Sample n_action_samples candidate actions for next states
            ns_rep = ns.unsqueeze(1).repeat(1, self.n_action_samples, 1).view(B * self.n_action_samples, -1)
            # Generate candidate actions via VAE
            cand_a = self.vae.sample(ns_rep)
            # Perturb
            cand_a = self.perturb_target(ns_rep, cand_a)
            # Evaluate with target Q
            q1_next, q2_next = self.qf_target(ns_rep, cand_a)
            q_next = self.lmbda * torch.min(q1_next, q2_next) + (1.0 - self.lmbda) * torch.max(q1_next, q2_next)
            # Max over candidates
            q_next = q_next.view(B, self.n_action_samples, 1).max(dim=1).values  # (B, 1)
            q_backup = r + self.gamma * (1.0 - done) * q_next

        q1, q2 = self.qf(s, a)
        qf_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.qf_opt.zero_grad()
        qf_loss.backward()
        self.qf_opt.step()

        # ---- Perturbation network update ----
        cand_a_gen = self.vae.sample(s)
        perturbed_a = self.perturb(s, cand_a_gen)
        q1_perturb, _ = self.qf(s, perturbed_a)
        perturb_loss = -q1_perturb.mean()

        self.perturb_opt.zero_grad()
        perturb_loss.backward()
        self.perturb_opt.step()

        # ---- Soft target updates ----
        with torch.no_grad():
            for p, pt in zip(self.qf.parameters(), self.qf_target.parameters()):
                pt.data.mul_(1.0 - self.polyak).add_(self.polyak * p.data)
            for p, pt in zip(self.perturb.parameters(), self.perturb_target.parameters()):
                pt.data.mul_(1.0 - self.polyak).add_(self.polyak * p.data)

        return {
            "vae_loss": vae_loss.item(),
            "qf_loss": qf_loss.item(),
            "perturb_loss": perturb_loss.item(),
        }

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        s = torch.from_numpy(state).float().to(self.device)
        s_rep = s.unsqueeze(0).repeat(self.n_action_samples, 1)
        cand_a = self.vae.sample(s_rep)
        cand_a = self.perturb(s_rep, cand_a)
        q1, q2 = self.qf(s_rep, cand_a)
        ind = (self.lmbda * q1 + (1.0 - self.lmbda) * q2).argmax(dim=0).item()
        return cand_a[ind].cpu().numpy()

    def save(self, path: str):
        torch.save({
            "vae": self.vae.state_dict(),
            "perturb": self.perturb.state_dict(),
            "perturb_target": self.perturb_target.state_dict(),
            "qf": self.qf.state_dict(),
            "qf_target": self.qf_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ckpt["vae"])
        self.perturb.load_state_dict(ckpt["perturb"])
        self.perturb_target.load_state_dict(ckpt["perturb_target"])
        self.qf.load_state_dict(ckpt["qf"])
        self.qf_target.load_state_dict(ckpt["qf_target"])

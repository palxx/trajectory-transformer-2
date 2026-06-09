"""
Implicit Q-Learning (IQL) for offline RL.

Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning" (ICLR 2022)

Update rules:
  V(s) : L_V = E[L_τ(Q(s,a) - V(s))]        τ=0.7 (expectile regression)
  Q(s,a): L_Q = (r + γ * V(s') - Q(s,a))²
  π(a|s): L_π = exp(β * A(s,a)) * log π(a|s)  where A = Q - V (advantage-weighted regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dim, 1, n_layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)  # (B, 1)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = build_mlp(state_dim + action_dim, hidden_dim, 1, n_layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)  # (B, 1)


class TwinQNetwork(nn.Module):
    """Two Q-networks for conservative estimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim, n_layers)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim, n_layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(states, actions), self.q2(states, actions)

    def min(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(states, actions)
        return torch.min(q1, q2)


class GaussianPolicy(nn.Module):
    """Gaussian policy for IQL (deterministic mean + fixed log_std)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.backbone = build_mlp(state_dim, hidden_dim, hidden_dim, n_layers - 1)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.backbone(states))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)  # (B, 1)

    def sample(self, states: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.rsample()


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Asymmetric L2 loss (expectile regression).
    L_τ(u) = |τ - 1(u<0)| * u²
    """
    weight = torch.where(diff >= 0, torch.full_like(diff, tau), torch.full_like(diff, 1.0 - tau))
    return (weight * diff.pow(2)).mean()


class IQL:
    """
    Implicit Q-Learning agent.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau_expectile: float = 0.7,
        beta: float = 3.0,
        polyak: float = 0.005,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau_expectile
        self.beta = beta
        self.polyak = polyak
        self.device = device

        # Networks
        self.vf = ValueNetwork(state_dim, hidden_dim, n_layers).to(device)
        self.qf = TwinQNetwork(state_dim, action_dim, hidden_dim, n_layers).to(device)
        self.qf_target = deepcopy(self.qf).to(device)
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim, n_layers).to(device)

        # Optimizers
        self.vf_opt = torch.optim.Adam(self.vf.parameters(), lr=lr)
        self.qf_opt = torch.optim.Adam(self.qf.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Freeze target
        for p in self.qf_target.parameters():
            p.requires_grad = False

    def update(self, batch: dict) -> Dict[str, float]:
        s = batch["observations"].to(self.device)
        a = batch["actions"].to(self.device)
        r = batch["rewards"].to(self.device).unsqueeze(-1)
        ns = batch["next_observations"].to(self.device)
        done = batch["terminals"].to(self.device).unsqueeze(-1)

        # ---- Value function update ----
        with torch.no_grad():
            q_target = self.qf_target.min(s, a)

        v_pred = self.vf(s)
        vf_loss = expectile_loss(q_target - v_pred, self.tau)

        self.vf_opt.zero_grad()
        vf_loss.backward()
        self.vf_opt.step()

        # ---- Q function update ----
        with torch.no_grad():
            v_next = self.vf(ns)
            q_backup = r + self.gamma * (1.0 - done) * v_next

        q1, q2 = self.qf(s, a)
        qf_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.qf_opt.zero_grad()
        qf_loss.backward()
        self.qf_opt.step()

        # ---- Policy update ----
        with torch.no_grad():
            v = self.vf(s)
            q = self.qf_target.min(s, a)
            advantage = q - v
            # Clamp advantage for numerical stability
            exp_adv = torch.exp(self.beta * advantage).clamp(max=100.0)

        log_pi = self.actor.log_prob(s, a)
        actor_loss = -(exp_adv * log_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Soft target update ----
        with torch.no_grad():
            for p, pt in zip(self.qf.parameters(), self.qf_target.parameters()):
                pt.data.mul_(1.0 - self.polyak)
                pt.data.add_(self.polyak * p.data)

        return {
            "vf_loss": vf_loss.item(),
            "qf_loss": qf_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        s = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        mean, _ = self.actor.forward(s)
        return mean.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "vf": self.vf.state_dict(),
            "qf": self.qf.state_dict(),
            "qf_target": self.qf_target.state_dict(),
            "actor": self.actor.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.vf.load_state_dict(ckpt["vf"])
        self.qf.load_state_dict(ckpt["qf"])
        self.qf_target.load_state_dict(ckpt["qf_target"])
        self.actor.load_state_dict(ckpt["actor"])

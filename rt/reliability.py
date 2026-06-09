"""
Reliability estimation for RT (Eq. 3-4 from paper).

Cumulative Reliability (Eq. 3):
    Γ(s_t, τ_{<t}) = Σ_{i=1}^{t} attn_weight_i * vae_error_i

where:
  - attn_weight_i = softmax-normalized attention weight from state at timestep t
                    attending to position i
  - vae_error_i   = VAE reconstruction error on transition (s_i, a_i, s_{i+1})

Threshold (Eq. 8):
    α = max reconstruction error on offline dataset

Truncation (Eq. 4):
    U_t = 0  if Γ(s_t, τ_{<t}) ≤ α   (reliable, continue)
          1  if Γ(s_t, τ_{<t}) > α   (unreliable, stop)

Pessimistic Reward (Eq. 6):
    R̂(s_t, a_t) = 0                                    if U_t = 1
                  r(s_t, a_t) - β * Γ(s_t, τ_{<t}) / α  otherwise
"""

import torch
import numpy as np
from typing import Optional, Tuple

from rt.models.transformer import RTTransformer
from rt.models.vae import TransitionVAE


class ReliabilityEstimator:
    """
    Encapsulates reliability estimation logic.

    Given a sequence context, computes per-step reliability scores
    and determines whether to continue or truncate generation.
    """

    def __init__(
        self,
        vae: TransitionVAE,
        alpha: float,              # reliability threshold (max VAE error on D_env)
        beta: float = 1.0,        # pessimism coefficient
        device: str = "cpu",
    ):
        self.vae = vae
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.vae.eval()

    @torch.no_grad()
    def compute_vae_errors(
        self,
        states: torch.Tensor,       # (B, T, state_dim)
        actions: torch.Tensor,      # (B, T, action_dim)
        next_states: torch.Tensor,  # (B, T, state_dim)
    ) -> torch.Tensor:
        """
        Compute VAE reconstruction error for each transition.
        Returns: (B, T) per-transition errors.
        """
        B, T, _ = states.shape
        s = states.reshape(B * T, -1)
        a = actions.reshape(B * T, -1)
        ns = next_states.reshape(B * T, -1)
        errors = self.vae.reconstruction_error(s, a, ns)   # (B*T,)
        return errors.reshape(B, T)

    @torch.no_grad()
    def compute_cumulative_reliability(
        self,
        transformer: RTTransformer,
        states: torch.Tensor,       # (B, T, state_dim) — context so far
        actions: torch.Tensor,      # (B, T, action_dim)
        rewards: torch.Tensor,      # (B, T, 1)
        rtg: torch.Tensor,          # (B, T, 1)
        timesteps: torch.Tensor,    # (B, T)
        next_states: torch.Tensor,  # (B, T, state_dim)  s_{t+1} for each t
    ) -> torch.Tensor:
        """
        Compute Γ(s_t, τ_{<t}) for the LAST timestep t in the sequence.

        Returns:
            gamma_vals: (B,) cumulative reliability scores for last position
        """
        B, T, _ = states.shape

        # 1. Run transformer forward to collect attention weights
        #    We need attention weights from last block's last attention layer
        transformer.eval()
        _ = transformer.forward(rtg, states, actions, rewards, timesteps)

        # 2. Get attention weights: (B, 4T, 4T), averaged over heads
        attn_weights_full = transformer.get_last_attention_weights()  # (B, 4T, 4T)
        if attn_weights_full is None:
            return torch.zeros(B, device=states.device)

        # 3. Get attention weights of the last state token attending to previous positions
        #    State token at timestep t is at position 4*(t-1)+1 in 0-indexed = 4*(T-1)+1
        last_state_pos = 4 * (T - 1) + 1
        # attn_weights[b, last_state_pos, :] are attention weights over all positions
        attn_row = attn_weights_full[:, last_state_pos, :]  # (B, 4T)

        # 4. Compute VAE errors for all transitions in context
        #    For each timestep i (1-indexed), error uses (s_i, a_i, s_{i+1})
        #    We have next_states as s_{i+1} for i=0..T-2, last next_state is predicted
        vae_errors = self.compute_vae_errors(
            states[:, :-1],       # s_0 .. s_{T-2}   (B, T-1, state_dim)
            actions[:, :-1],      # a_0 .. a_{T-2}
            next_states[:, :-1],  # s_1 .. s_{T-1}
        )  # (B, T-1)

        # 5. Map VAE errors to token positions
        #    Error for timestep i is associated with state token at position 4*i+1
        #    We aggregate: Γ = Σ_i attn_weight[state_i_pos] * vae_error_i
        gamma_vals = torch.zeros(B, device=states.device)
        for i in range(T - 1):
            state_pos = 4 * i + 1  # position of state token at timestep i
            if state_pos < attn_row.shape[1]:
                gamma_vals += attn_row[:, state_pos] * vae_errors[:, i]

        return gamma_vals  # (B,)

    def is_reliable(self, gamma: torch.Tensor) -> torch.Tensor:
        """U_t = 0 (reliable) if Γ ≤ α. Returns bool tensor (B,)."""
        return gamma <= self.alpha

    def pessimistic_reward(
        self,
        raw_reward: torch.Tensor,  # (B,)
        gamma: torch.Tensor,       # (B,) cumulative reliability
        reliable: torch.Tensor,    # (B,) bool
    ) -> torch.Tensor:
        """
        Eq. 6: pessimistic reward
          R̂ = 0                              if unreliable
               r - β * Γ / α               otherwise
        """
        penalty = self.beta * gamma / (self.alpha + 1e-8)
        adj_reward = raw_reward - penalty
        return torch.where(reliable, adj_reward, torch.zeros_like(adj_reward))


@torch.no_grad()
def compute_max_vae_error(
    vae: TransitionVAE,
    trajectories: list,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> float:
    """
    Compute maximum VAE reconstruction error on all transitions in D_env.
    This becomes the threshold α (Eq. 8).
    """
    from rt.data import normalize_states

    vae.eval()
    max_error = 0.0

    # Collect all transitions
    s_list, a_list, ns_list = [], [], []
    for traj in trajectories:
        obs = traj["observations"].astype(np.float32)
        obs = normalize_states(obs, state_mean, state_std)
        acts = traj["actions"].astype(np.float32)
        T = len(obs)
        if T < 2:
            continue
        s_list.append(obs[:-1])
        a_list.append(acts[:-1])
        ns_list.append(obs[1:])

    all_s = np.concatenate(s_list, axis=0)
    all_a = np.concatenate(a_list, axis=0)
    all_ns = np.concatenate(ns_list, axis=0)

    N = len(all_s)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        s = torch.from_numpy(all_s[start:end]).to(device)
        a = torch.from_numpy(all_a[start:end]).to(device)
        ns = torch.from_numpy(all_ns[start:end]).to(device)
        errors = vae.reconstruction_error(s, a, ns)  # (B,)
        batch_max = errors.max().item()
        if batch_max > max_error:
            max_error = batch_max

    return max_error

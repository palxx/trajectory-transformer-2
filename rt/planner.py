"""
RTBeamPlanner: beam-search MPC controller using the forward RT transformer
as a learned world model (Trajectory-Transformer-style online planning).

Unlike the IQL/BCQ agents, this planner never trains a policy network — at
every real environment step it "imagines" several candidate futures with the
forward RTTransformer, scores them by predicted discounted reward, and
executes the first action of the best-scoring imagined rollout.

Per imagined step, three `generate_step` calls are used (mirroring the
prediction convention documented in rt/models/transformer.py):
  1. predict a_t from (R_t, s_t)            -> pred_action
  2. predict r_t, R_{t+1} from (.., a_t)     -> pred_reward, pred_rtg_next
  3. predict s_{t+1} from (.., R_{t+1})      -> pred_state_next
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

from rt.models.transformer import RTTransformer


@dataclass
class _Beam:
    """One imagined rollout branch. Tensor shapes: (1, T, dim)."""
    rtg: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    timesteps: torch.Tensor
    first_action: np.ndarray
    score: float = 0.0


class RTBeamPlanner:
    """Beam-search MPC controller wrapping a trained forward RTTransformer."""

    def __init__(
        self,
        forward_rt: RTTransformer,
        state_dim: int,
        action_dim: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        gamma: float = 0.99,
        context_len: int = 20,
        planning_horizon: int = 5,
        beam_width: int = 4,
        n_candidates: int = 8,
        action_noise_std: float = 0.1,
        device: str = "cpu",
    ):
        self.forward_rt = forward_rt.to(device).eval()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.context_len = context_len
        self.planning_horizon = planning_horizon
        self.beam_width = beam_width
        self.n_candidates = n_candidates
        self.action_noise_std = action_noise_std
        self.device = device

    def reset(self, init_state: np.ndarray, target_return: float, max_ep_len: int = 1000) -> None:
        """Start a new episode. `init_state` must already be normalized."""
        self.max_ep_len = max_ep_len
        self.t = 0
        self.rtg_real = float(target_return)

        device = self.device
        self.rtg = torch.tensor([[[self.rtg_real]]], dtype=torch.float32, device=device)
        self.states = torch.tensor(init_state, dtype=torch.float32, device=device).view(1, 1, self.state_dim)
        self.actions = torch.zeros(1, 1, self.action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(1, 1, 1, dtype=torch.float32, device=device)
        self.timesteps = torch.tensor([[self.t]], dtype=torch.long, device=device)

    @torch.no_grad()
    def _expand_beam(self, beam: _Beam, step_idx: int) -> List[_Beam]:
        rtg, states, actions, rewards, timesteps = (
            beam.rtg, beam.states, beam.actions, beam.rewards, beam.timesteps,
        )

        # Pass 1: predict action a_t from (R_t, s_t). Causal masking means the
        # zero placeholders at actions[:, -1] / rewards[:, -1] don't affect this.
        _, pred_action, _, _ = self.forward_rt.generate_step(rtg, states, actions, rewards, timesteps)

        # Build candidate actions: deterministic prediction + noisy variants.
        candidates = [pred_action]
        for _ in range(self.n_candidates - 1):
            noise_scale = pred_action.abs().mean() * self.action_noise_std + 1e-3
            candidates.append(pred_action + torch.randn_like(pred_action) * noise_scale)

        children: List[_Beam] = []
        for candidate in candidates:
            # Pass 2: fill in a_t, predict r_t and R_{t+1}.
            actions_filled = actions.clone()
            actions_filled[:, -1, :] = candidate
            _, _, pred_reward, pred_rtg_next = self.forward_rt.generate_step(
                rtg, states, actions_filled, rewards, timesteps
            )

            rewards_filled = rewards.clone()
            rewards_filled[:, -1, :] = pred_reward

            # Pass 3: append new timestep with R_{t+1}, predict s_{t+1}.
            rtg_ext = torch.cat([rtg, pred_rtg_next.unsqueeze(1)], dim=1)
            states_ext = torch.cat([states, torch.zeros_like(states[:, :1])], dim=1)
            actions_ext = torch.cat([actions_filled, torch.zeros_like(actions[:, :1])], dim=1)
            rewards_ext = torch.cat([rewards_filled, torch.zeros_like(rewards[:, :1])], dim=1)
            ts_ext = torch.cat([timesteps, timesteps[:, -1:] + 1], dim=1)

            pred_state_next, _, _, _ = self.forward_rt.generate_step(
                rtg_ext, states_ext, actions_ext, rewards_ext, ts_ext
            )
            states_ext[:, -1, :] = pred_state_next

            ctx = self.context_len
            score = beam.score + (self.gamma ** step_idx) * pred_reward.item()
            first_action = (
                candidate.squeeze(0).cpu().numpy() if step_idx == 0 else beam.first_action
            )
            children.append(_Beam(
                rtg=rtg_ext[:, -ctx:], states=states_ext[:, -ctx:], actions=actions_ext[:, -ctx:],
                rewards=rewards_ext[:, -ctx:], timesteps=ts_ext[:, -ctx:],
                first_action=first_action, score=score,
            ))

        return children

    @torch.no_grad()
    def act(self) -> np.ndarray:
        """Run beam search from the current real history; return the real
        (clipped) action to execute next."""
        root = _Beam(
            rtg=self.rtg, states=self.states, actions=self.actions,
            rewards=self.rewards, timesteps=self.timesteps,
            first_action=None, score=0.0,
        )
        beams = [root]
        for step_idx in range(self.planning_horizon):
            children: List[_Beam] = []
            for beam in beams:
                children.extend(self._expand_beam(beam, step_idx))
            children.sort(key=lambda b: b.score, reverse=True)
            beams = children[: self.beam_width]

        action = beams[0].first_action
        return np.clip(action, self.action_low, self.action_high)

    def append_real_transition(self, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        """After env.step(): record the real (action, reward, next_state).
        `next_state` must already be normalized."""
        device = self.device
        self.actions[:, -1, :] = torch.tensor(action, dtype=torch.float32, device=device)
        self.rewards[:, -1, :] = float(reward)

        self.rtg_real -= float(reward)
        self.t += 1

        new_rtg = torch.tensor([[[self.rtg_real]]], dtype=torch.float32, device=device)
        new_state = torch.tensor(next_state, dtype=torch.float32, device=device).view(1, 1, self.state_dim)
        new_action = torch.zeros(1, 1, self.action_dim, dtype=torch.float32, device=device)
        new_reward = torch.zeros(1, 1, 1, dtype=torch.float32, device=device)
        new_ts = torch.tensor([[min(self.t, self.max_ep_len - 1)]], dtype=torch.long, device=device)

        ctx = self.context_len
        self.rtg = torch.cat([self.rtg, new_rtg], dim=1)[:, -ctx:]
        self.states = torch.cat([self.states, new_state], dim=1)[:, -ctx:]
        self.actions = torch.cat([self.actions, new_action], dim=1)[:, -ctx:]
        self.rewards = torch.cat([self.rewards, new_reward], dim=1)[:, -ctx:]
        self.timesteps = torch.cat([self.timesteps, new_ts], dim=1)[:, -ctx:]

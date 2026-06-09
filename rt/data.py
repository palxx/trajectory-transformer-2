"""
Data loading and preprocessing utilities for RT offline RL.

Handles:
  - Loading D4RL datasets
  - Computing return-to-go (RTG)
  - State normalization
  - Creating trajectory segments for transformer training
  - PyTorch Dataset/DataLoader wrappers
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


def load_d4rl_dataset(env_name: str) -> Tuple[dict, object]:
    """
    Load a D4RL dataset.
    Returns: (dataset_dict, env)
    """
    import gym
    import d4rl  # noqa: F401 - registers environments
    env = gym.make(env_name)
    dataset = env.get_dataset()
    return dataset, env


def split_into_trajectories(dataset: dict) -> List[dict]:
    """
    Split a flat D4RL dataset dictionary into a list of episode trajectories.
    Each trajectory dict has keys: observations, actions, rewards, terminals, timeouts
    """
    obs = dataset["observations"]
    acts = dataset["actions"]
    rews = dataset["rewards"]
    terminals = dataset.get("terminals", np.zeros(len(rews), dtype=bool))
    timeouts = dataset.get("timeouts", np.zeros(len(rews), dtype=bool))

    trajectories = []
    traj = {"observations": [], "actions": [], "rewards": [], "terminals": [], "timeouts": []}

    for i in range(len(rews)):
        traj["observations"].append(obs[i])
        traj["actions"].append(acts[i])
        traj["rewards"].append(rews[i])
        traj["terminals"].append(terminals[i])
        traj["timeouts"].append(timeouts[i])

        if terminals[i] or timeouts[i]:
            trajectories.append({
                k: np.array(v) for k, v in traj.items()
            })
            traj = {"observations": [], "actions": [], "rewards": [], "terminals": [], "timeouts": []}

    # Include any incomplete trailing trajectory
    if len(traj["observations"]) > 0:
        trajectories.append({k: np.array(v) for k, v in traj.items()})

    return trajectories


def compute_rtg(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Compute discounted return-to-go for a reward sequence."""
    T = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


def compute_state_stats(trajectories: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of states over all trajectories."""
    all_states = np.concatenate([traj["observations"] for traj in trajectories], axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0) + 1e-6
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def denormalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return states * std + mean


class TrajectoryDataset(Dataset):
    """
    Dataset of fixed-length trajectory segments for transformer training.

    Each item is a dict with tensors:
      rtg:       (context_len, 1)
      states:    (context_len, state_dim)
      actions:   (context_len, action_dim)
      rewards:   (context_len, 1)
      timesteps: (context_len,) int
    """

    def __init__(
        self,
        trajectories: List[dict],
        context_len: int,
        gamma: float = 0.99,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ):
        self.context_len = context_len
        self.gamma = gamma
        self.state_mean = state_mean
        self.state_std = state_std

        # Pre-process all trajectories
        self.segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
        # Each element: (rtg, states, actions, rewards, traj_len)

        self._processed_trajs = []
        for traj in trajectories:
            obs = traj["observations"].astype(np.float32)
            acts = traj["actions"].astype(np.float32)
            rews = traj["rewards"].astype(np.float32)
            rtg = compute_rtg(rews, gamma)  # (T,)

            if state_mean is not None and state_std is not None:
                obs = normalize_states(obs, state_mean, state_std)

            self._processed_trajs.append({
                "observations": obs,
                "actions": acts,
                "rewards": rews,
                "rtg": rtg,
                "length": len(rews),
            })

        # Build index: (traj_idx, start_t)
        self.index = []
        for i, traj in enumerate(self._processed_trajs):
            T = traj["length"]
            for start in range(0, T):
                self.index.append((i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        traj_idx, start = self.index[idx]
        traj = self._processed_trajs[traj_idx]
        T = traj["length"]

        end = min(start + self.context_len, T)
        seg_len = end - start

        # Pad if segment shorter than context_len
        pad = self.context_len - seg_len

        obs = traj["observations"][start:end]
        acts = traj["actions"][start:end]
        rews = traj["rewards"][start:end]
        rtg = traj["rtg"][start:end]

        if pad > 0:
            obs = np.concatenate([obs, np.zeros((pad, obs.shape[-1]), dtype=np.float32)], axis=0)
            acts = np.concatenate([acts, np.zeros((pad, acts.shape[-1]), dtype=np.float32)], axis=0)
            rews = np.concatenate([rews, np.zeros(pad, dtype=np.float32)], axis=0)
            rtg = np.concatenate([rtg, np.zeros(pad, dtype=np.float32)], axis=0)

        timesteps = np.arange(start, start + self.context_len, dtype=np.int64)

        return {
            "rtg": torch.from_numpy(rtg).float().unsqueeze(-1),           # (T, 1)
            "states": torch.from_numpy(obs).float(),                        # (T, state_dim)
            "actions": torch.from_numpy(acts).float(),                      # (T, action_dim)
            "rewards": torch.from_numpy(rews).float().unsqueeze(-1),       # (T, 1)
            "timesteps": torch.from_numpy(timesteps).long(),                # (T,)
            "mask": torch.cat([
                torch.ones(seg_len, dtype=torch.bool),
                torch.zeros(pad, dtype=torch.bool)
            ]),  # (T,) True = real data
        }


class TransitionDataset(Dataset):
    """
    Dataset of (s, a, s') transitions for VAE training.
    Also supports classifier training (includes reward labels).
    """

    def __init__(
        self,
        trajectories: List[dict],
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ):
        states_list, actions_list, next_states_list, rewards_list = [], [], [], []

        for traj in trajectories:
            obs = traj["observations"].astype(np.float32)
            acts = traj["actions"].astype(np.float32)
            rews = traj["rewards"].astype(np.float32)

            if state_mean is not None and state_std is not None:
                obs = normalize_states(obs, state_mean, state_std)

            T = len(obs)
            if T < 2:
                continue

            states_list.append(obs[:-1])
            actions_list.append(acts[:-1])
            next_states_list.append(obs[1:])
            rewards_list.append(rews[:-1])

        self.states = torch.from_numpy(np.concatenate(states_list, axis=0)).float()
        self.actions = torch.from_numpy(np.concatenate(actions_list, axis=0)).float()
        self.next_states = torch.from_numpy(np.concatenate(next_states_list, axis=0)).float()
        self.rewards = torch.from_numpy(np.concatenate(rewards_list, axis=0)).float()

        # Binary label: 1 if reward >= median
        median_reward = self.rewards.median().item()
        self.high_reward_labels = (self.rewards >= median_reward).float()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "next_states": self.next_states[idx],
            "rewards": self.rewards[idx],
            "high_reward": self.high_reward_labels[idx],
        }


class RLDataset(Dataset):
    """
    Dataset of (s, a, r, s', done) transitions for offline RL training.
    Supports combined datasets (D_env + D_model).
    """

    def __init__(self, transitions: dict):
        """
        transitions: dict with keys:
          observations, actions, rewards, next_observations, terminals
        Each value: np.ndarray of shape (N, ...)
        """
        self.observations = torch.from_numpy(transitions["observations"].astype(np.float32))
        self.actions = torch.from_numpy(transitions["actions"].astype(np.float32))
        self.rewards = torch.from_numpy(transitions["rewards"].astype(np.float32))
        self.next_observations = torch.from_numpy(transitions["next_observations"].astype(np.float32))
        self.terminals = torch.from_numpy(transitions["terminals"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> dict:
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "terminals": self.terminals[idx],
        }


def trajectories_to_rl_transitions(
    trajectories: List[dict],
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
) -> dict:
    """Convert list of trajectory dicts to flat RL transition dict."""
    obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []

    for traj in trajectories:
        obs = traj["observations"].astype(np.float32)
        acts = traj["actions"].astype(np.float32)
        rews = traj["rewards"].astype(np.float32)
        terms = traj.get("terminals", np.zeros(len(rews), dtype=np.float32)).astype(np.float32)

        if state_mean is not None and state_std is not None:
            obs = normalize_states(obs, state_mean, state_std)

        T = len(obs)
        if T < 2:
            continue

        obs_list.append(obs[:-1])
        act_list.append(acts[:-1])
        rew_list.append(rews[:-1])
        next_obs_list.append(obs[1:])
        done_list.append(terms[:-1])

    return {
        "observations": np.concatenate(obs_list, axis=0),
        "actions": np.concatenate(act_list, axis=0),
        "rewards": np.concatenate(rew_list, axis=0),
        "next_observations": np.concatenate(next_obs_list, axis=0),
        "terminals": np.concatenate(done_list, axis=0),
    }

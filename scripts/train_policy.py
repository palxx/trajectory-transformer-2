"""
Script 3: Train offline RL policy (IQL or BCQ) on D = D_env ∪ D_model.

Usage:
    python scripts/train_policy.py --env hopper-medium-v2 --algo iql --device cuda
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rt.data import (
    load_d4rl_dataset, split_into_trajectories,
    trajectories_to_rl_transitions, RLDataset,
)
from rt.offline_rl.iql import IQL
from rt.offline_rl.bcq import BCQ


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="hopper-medium-v2")
    p.add_argument("--algo", type=str, default="iql", choices=["iql", "bcq"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--train_steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    # IQL
    p.add_argument("--iql_tau", type=float, default=0.7)
    p.add_argument("--iql_beta", type=float, default=3.0)
    # BCQ
    p.add_argument("--bcq_phi", type=float, default=0.05)
    p.add_argument("--bcq_lmbda", type=float, default=0.75)
    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_freq", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def evaluate_policy(agent, env, n_episodes: int, state_mean: np.ndarray, state_std: np.ndarray) -> float:
    """Evaluate policy for n_episodes, return mean normalized score."""
    import gym
    total_reward = 0.0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_norm = (obs - state_mean) / state_std
            action = agent.select_action(obs_norm)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
    return total_reward / n_episodes


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load environment
    import gym
    import d4rl  # noqa
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load normalization stats
    state_mean = np.load(os.path.join(args.checkpoint_dir, "state_mean.npy"))
    state_std = np.load(os.path.join(args.checkpoint_dir, "state_std.npy"))

    # Load D_env
    env_path = os.path.join(args.data_dir, f"{args.env}_d_env.pkl")
    if os.path.exists(env_path):
        with open(env_path, "rb") as f:
            env_trajectories = pickle.load(f)
    else:
        print("D_env pickle not found, loading from D4RL directly...")
        dataset_raw, _ = load_d4rl_dataset(args.env)  # noqa
        from rt.data import load_d4rl_dataset, split_into_trajectories
        _, env2 = load_d4rl_dataset(args.env)
        dataset_raw2 = env2.get_dataset()
        env_trajectories = split_into_trajectories(dataset_raw2)

    env_transitions = trajectories_to_rl_transitions(env_trajectories, state_mean, state_std)

    # Load D_model if available
    model_path = os.path.join(args.data_dir, f"{args.env}_d_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_trajectories = pickle.load(f)
        model_transitions = trajectories_to_rl_transitions(model_trajectories, state_mean, state_std)
        # Combine D_env + D_model
        combined = {
            k: np.concatenate([env_transitions[k], model_transitions[k]], axis=0)
            for k in env_transitions.keys()
        }
        print(f"Combined dataset: {len(combined['observations'])} transitions "
              f"(env={len(env_transitions['observations'])}, model={len(model_transitions['observations'])})")
    else:
        combined = env_transitions
        print(f"No D_model found, using only D_env: {len(combined['observations'])} transitions")

    dataset = RLDataset(combined)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Build agent
    if args.algo == "iql":
        agent = IQL(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            lr=args.lr, gamma=args.gamma,
            tau_expectile=args.iql_tau, beta=args.iql_beta,
            device=args.device,
        )
    else:
        agent = BCQ(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=750, lr=args.lr, gamma=args.gamma,
            phi=args.bcq_phi, lmbda=args.bcq_lmbda,
            device=args.device,
        )

    print(f"\nTraining {args.algo.upper()} for {args.train_steps} steps...")
    loader_iter = iter(loader)
    logs = {k: [] for k in ["vf_loss", "qf_loss", "actor_loss", "vae_loss", "perturb_loss"]}

    best_score = -1e9
    for step in tqdm(range(1, args.train_steps + 1), desc=f"Training {args.algo.upper()}"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        info = agent.update(batch)
        for k, v in info.items():
            if k in logs:
                logs[k].append(v)

        if step % args.log_freq == 0:
            log_str = f"[{args.algo.upper()}] Step {step}/{args.train_steps}"
            for k, vs in logs.items():
                if vs:
                    log_str += f" | {k}={np.mean(vs[-args.log_freq:]):.4f}"
            print(log_str)

        if step % args.eval_freq == 0:
            score = evaluate_policy(agent, env, args.eval_episodes, state_mean, state_std)
            try:
                norm_score = env.get_normalized_score(score) * 100
            except Exception:
                norm_score = score
            print(f"  Eval @ step {step}: raw={score:.2f}, normalized={norm_score:.2f}")

            if norm_score > best_score:
                best_score = norm_score
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.algo}_best.pt")
                agent.save(ckpt_path)
                print(f"  New best! Saved to {ckpt_path}")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, f"{args.algo}_final.pt")
    agent.save(final_path)
    print(f"\nFinal agent saved to {final_path}")
    print(f"Best normalized score: {best_score:.2f}")


if __name__ == "__main__":
    main()

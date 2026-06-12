"""
Script: Online beam-search planning with the trained forward RT transformer.

Uses RTBeamPlanner (rt/planner.py) as a Trajectory-Transformer-style MPC
controller: at each environment step, the forward RT transformer "imagines"
several candidate futures, scores them by predicted discounted reward, and
the first action of the best-scoring imagined rollout is executed.

Usage:
    python scripts/plan.py --env hopper-medium-v2 \
        --checkpoint_dir experiments/hopper-medium-v2_iql_seed0_1781071789 \
        --n_episodes 5 --device cuda
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rt.data import load_minari_dataset, minari_to_trajectories, get_normalized_score, normalize_states, compute_rtg
from rt.models.transformer import RTTransformer
from rt.planner import RTBeamPlanner


def get_args():
    p = argparse.ArgumentParser(description="RT beam-search planner evaluation")
    p.add_argument("--env", type=str, default="hopper-medium-v2")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="planner_results")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--target_return", type=float, default=None,
                   help="If unset, use the max episode return observed in the dataset.")
    p.add_argument("--max_ep_len", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.99)

    # Beam search
    p.add_argument("--planning_horizon", type=int, default=5)
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--n_candidates", type=int, default=8)
    p.add_argument("--action_noise_std", type=float, default=0.1)

    # RT architecture (must match the checkpoint)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--context_len", type=int, default=20)

    return p.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print(f"Loading dataset: {args.env}")
    minari_dataset, env = load_minari_dataset(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    state_mean = np.load(os.path.join(args.checkpoint_dir, "state_mean.npy"))
    state_std = np.load(os.path.join(args.checkpoint_dir, "state_std.npy"))

    target_return = args.target_return
    if target_return is None:
        trajectories = minari_to_trajectories(minari_dataset)
        target_return = float(max(compute_rtg(t["rewards"], args.gamma)[0] for t in trajectories))
        print(f"  Using data-driven target_return = {target_return:.2f}")

    print("Loading Forward RT...")
    forward_rt = RTTransformer(
        state_dim=state_dim, action_dim=action_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, context_len=args.context_len,
    )
    forward_rt.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "forward_rt.pt"), map_location=device))
    forward_rt.eval()

    planner = RTBeamPlanner(
        forward_rt=forward_rt,
        state_dim=state_dim, action_dim=action_dim,
        state_mean=state_mean, state_std=state_std,
        action_low=env.action_space.low, action_high=env.action_space.high,
        gamma=args.gamma, context_len=args.context_len,
        planning_horizon=args.planning_horizon, beam_width=args.beam_width,
        n_candidates=args.n_candidates, action_noise_std=args.action_noise_std,
        device=device,
    )

    raw_returns = []
    norm_scores = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        planner.reset(normalize_states(obs, state_mean, state_std), target_return, max_ep_len=args.max_ep_len)

        ep_return = 0.0
        for t in range(args.max_ep_len):
            action = planner.act()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            planner.append_real_transition(action, reward, normalize_states(next_obs, state_mean, state_std))
            ep_return += reward
            obs = next_obs
            if terminated or truncated:
                break

        norm_score = float(get_normalized_score(minari_dataset, ep_return))
        raw_returns.append(float(ep_return))
        norm_scores.append(norm_score)
        print(f"  Episode {ep + 1}/{args.n_episodes}: raw_return={ep_return:.2f}, normalized={norm_score:.2f}")

    results = {
        "env": args.env,
        "checkpoint_dir": args.checkpoint_dir,
        "target_return": target_return,
        "planning_horizon": args.planning_horizon,
        "beam_width": args.beam_width,
        "n_candidates": args.n_candidates,
        "action_noise_std": args.action_noise_std,
        "raw_returns": raw_returns,
        "normalized_scores": norm_scores,
        "raw_return_mean": float(np.mean(raw_returns)),
        "raw_return_std": float(np.std(raw_returns)),
        "normalized_score_mean": float(np.mean(norm_scores)),
        "normalized_score_std": float(np.std(norm_scores)),
    }

    print(f"\n{'=' * 60}")
    print(f"RT Beam Planner results for {args.env}")
    print(f"  Raw return:       {results['raw_return_mean']:.2f} +/- {results['raw_return_std']:.2f}")
    print(f"  Normalized score: {results['normalized_score_mean']:.2f} +/- {results['normalized_score_std']:.2f}")
    print(f"{'=' * 60}")

    output_path = os.path.join(args.output_dir, f"{args.env}_plan_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

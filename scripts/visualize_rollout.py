"""
Script: Visually compare three rollouts as animated GIFs:
  1. A real episode replayed from the offline dataset (action-replay)
  2. A rollout from the RT beam-search planner (rt/planner.py)
  3. A rollout from the trained IQL policy (iql_best.pt)

Note on (1): Minari/D4RL MuJoCo observations exclude qpos[0] (x-position), so
exact env.set_state() replay of a recorded episode isn't possible. Instead we
reset the environment with the same seed used for (2) and (3) and step
through the recorded action sequence. This gives a visually representative
(but not frame-exact) reference trajectory, while keeping all three rollouts
comparable since they share the same starting state.

Usage:
    python scripts/visualize_rollout.py --env hopper-medium-v2 \
        --checkpoint_dir experiments/hopper-medium-v2_iql_seed0_1781071789 \
        --max_steps 200 --device cuda
"""

import argparse
import os
import sys
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rt.data import load_minari_dataset, minari_to_trajectories, normalize_states, compute_rtg
from rt.models.transformer import RTTransformer
from rt.planner import RTBeamPlanner
from rt.offline_rl.iql import IQL


def get_args():
    p = argparse.ArgumentParser(description="Visualize and compare RT rollouts")
    p.add_argument("--env", type=str, default="hopper-medium-v2")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="visualizations")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument("--target_return", type=float, default=None,
                   help="If unset, use the max episode return observed in the dataset.")
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

    # IQL architecture (must match the checkpoint)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_rl_layers", type=int, default=2)

    return p.parse_args()


def save_gif(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    imageio.mimsave(path, frames, duration=1.0 / fps)


def rollout_dataset_episode(
    env, trajectory: dict, seed: int, max_steps: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """Reset env(seed) and replay the recorded action sequence, capturing frames."""
    frames = []
    env.reset(seed=seed)
    frames.append(env.render())

    actions = trajectory["actions"]
    n = len(actions) if max_steps is None else min(max_steps, len(actions))
    total_r = 0.0
    for t in range(n):
        a = np.clip(actions[t], env.action_space.low, env.action_space.high)
        _, r, terminated, truncated, _ = env.step(a)
        frames.append(env.render())
        total_r += r
        if terminated or truncated:
            break
    return frames, total_r


def rollout_planner_episode(
    env, planner: RTBeamPlanner, seed: int, max_steps: int,
    state_mean: np.ndarray, state_std: np.ndarray, target_return: float,
) -> Tuple[List[np.ndarray], float]:
    frames = []
    obs, _ = env.reset(seed=seed)
    frames.append(env.render())
    planner.reset(normalize_states(obs, state_mean, state_std), target_return, max_ep_len=max_steps)

    total_r = 0.0
    for t in range(max_steps):
        action = planner.act()
        next_obs, r, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        planner.append_real_transition(action, r, normalize_states(next_obs, state_mean, state_std))
        total_r += r
        if terminated or truncated:
            break
    return frames, total_r


def rollout_iql_episode(
    env, agent: IQL, seed: int, max_steps: int,
    state_mean: np.ndarray, state_std: np.ndarray,
) -> Tuple[List[np.ndarray], float]:
    frames = []
    obs, _ = env.reset(seed=seed)
    frames.append(env.render())

    total_r = 0.0
    for t in range(max_steps):
        obs_norm = normalize_states(obs, state_mean, state_std)
        action = agent.select_action(obs_norm)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, r, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_r += r
        if terminated or truncated:
            break
    return frames, total_r


def combine_side_by_side(*frame_lists: List[np.ndarray]) -> List[np.ndarray]:
    """Pad each list to the same length (repeating its last frame), then
    stack horizontally per timestep."""
    n = max(len(frames) for frames in frame_lists)
    padded = []
    for frames in frame_lists:
        if len(frames) < n:
            frames = frames + [frames[-1]] * (n - len(frames))
        padded.append(frames)
    return [np.hstack(group) for group in zip(*padded)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print(f"Loading dataset: {args.env}")
    minari_dataset, _ = load_minari_dataset(args.env)
    env = minari_dataset.recover_environment(render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    trajectories = minari_to_trajectories(minari_dataset)
    traj = random.Random(args.seed).choice(trajectories)
    print(f"  Selected dataset episode with {len(traj['observations'])} steps")

    state_mean = np.load(os.path.join(args.checkpoint_dir, "state_mean.npy"))
    state_std = np.load(os.path.join(args.checkpoint_dir, "state_std.npy"))

    target_return = args.target_return
    if target_return is None:
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

    print("Loading IQL agent...")
    agent = IQL(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=args.hidden_dim, n_layers=args.n_rl_layers, device=device,
    )
    agent.load(os.path.join(args.checkpoint_dir, "iql_best.pt"))

    print("\nRolling out dataset replay...")
    frames_data, ret_data = rollout_dataset_episode(env, traj, args.seed, args.max_steps)
    print(f"  Dataset replay return: {ret_data:.2f}")

    print("Rolling out RT beam planner...")
    frames_plan, ret_plan = rollout_planner_episode(
        env, planner, args.seed, args.max_steps, state_mean, state_std, target_return,
    )
    print(f"  Planner return: {ret_plan:.2f}")

    print("Rolling out IQL policy...")
    frames_iql, ret_iql = rollout_iql_episode(env, agent, args.seed, args.max_steps, state_mean, state_std)
    print(f"  IQL return: {ret_iql:.2f}")

    prefix = os.path.join(args.output_dir, args.env)
    print("\nSaving GIFs...")
    save_gif(frames_data, f"{prefix}_dataset_seed{args.seed}.gif", fps=args.fps)
    save_gif(frames_plan, f"{prefix}_planner_seed{args.seed}.gif", fps=args.fps)
    save_gif(frames_iql, f"{prefix}_iql_seed{args.seed}.gif", fps=args.fps)

    combined = combine_side_by_side(frames_data, frames_plan, frames_iql)
    combined_path = f"{prefix}_comparison_seed{args.seed}.gif"
    save_gif(combined, combined_path, fps=args.fps)
    print(f"Saved comparison GIF (dataset | planner | iql) to {combined_path}")

    print(f"\n{'=' * 60}")
    print(f"Returns -> dataset: {ret_data:.2f}  planner: {ret_plan:.2f}  iql: {ret_iql:.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

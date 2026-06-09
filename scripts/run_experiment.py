"""
Script 4: End-to-end RT offline RL pipeline.

Runs the full Algorithm 1:
  1. Load D_env
  2. Train forward + backward RT transformer
  3. Train VAE, compute reliability threshold α
  4. Train classifier
  5. Augment dataset (generate D_model)
  6. Train offline RL policy (IQL/BCQ) on D = D_env ∪ D_model
  7. Evaluate final policy

Usage:
    python scripts/run_experiment.py --env hopper-medium-v2 --algo iql --device cuda
    python scripts/run_experiment.py --env walker2d-medium-v2 --algo iql --quick  # for quick test
"""

import argparse
import os
import sys
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rt.config import RTConfig
from rt.data import (
    load_d4rl_dataset, split_into_trajectories, compute_state_stats,
    TrajectoryDataset, TransitionDataset, RLDataset, trajectories_to_rl_transitions,
)
from rt.models.transformer import RTTransformer
from rt.models.vae import TransitionVAE
from rt.models.classifier import HighRewardClassifier
from rt.reliability import ReliabilityEstimator, compute_max_vae_error
from rt.augmentor import DataAugmentor
from rt.offline_rl.iql import IQL
from rt.offline_rl.bcq import BCQ

import torch.nn.functional as F


def get_args():
    p = argparse.ArgumentParser(description="RT: End-to-end offline RL pipeline")
    p.add_argument("--env", type=str, default="hopper-medium-v2")
    p.add_argument("--algo", type=str, default="iql", choices=["iql", "bcq"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_dir", type=str, default="experiments")

    # Transformer
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--rt_steps", type=int, default=100_000)
    p.add_argument("--rt_lr", type=float, default=1e-4)
    p.add_argument("--rt_batch_size", type=int, default=64)

    # VAE
    p.add_argument("--vae_latent_dim", type=int, default=32)
    p.add_argument("--vae_hidden_dim", type=int, default=256)
    p.add_argument("--vae_steps", type=int, default=50_000)
    p.add_argument("--vae_lr", type=float, default=1e-4)
    p.add_argument("--vae_batch_size", type=int, default=256)

    # Classifier
    p.add_argument("--cls_hidden_dim", type=int, default=256)
    p.add_argument("--cls_steps", type=int, default=20_000)
    p.add_argument("--cls_lr", type=float, default=1e-4)
    p.add_argument("--cls_batch_size", type=int, default=256)

    # Augmentation
    p.add_argument("--n_generations", type=int, default=50_000)
    p.add_argument("--generation_horizon", type=int, default=5)
    p.add_argument("--candidate_K", type=int, default=10)
    p.add_argument("--beta_pessimism", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)

    # RL training
    p.add_argument("--rl_steps", type=int, default=1_000_000)
    p.add_argument("--rl_batch_size", type=int, default=256)
    p.add_argument("--rl_lr", type=float, default=3e-4)
    p.add_argument("--iql_tau", type=float, default=0.7)
    p.add_argument("--iql_beta", type=float, default=3.0)
    p.add_argument("--bcq_phi", type=float, default=0.05)
    p.add_argument("--bcq_lmbda", type=float, default=0.75)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_rl_layers", type=int, default=2)

    # Eval
    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_freq", type=int, default=1000)

    # Quick mode (tiny config for testing)
    p.add_argument("--quick", action="store_true", help="Run with tiny config for quick testing")

    return p.parse_args()


def compute_rt_loss(model, batch, device):
    rtg = batch["rtg"].to(device)
    states = batch["states"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    timesteps = batch["timesteps"].to(device)
    mask = batch["mask"].to(device)

    pred_s, pred_a, pred_r, pred_rtg = model(rtg, states, actions, rewards, timesteps)

    mask_f = mask.float()
    mask_3d = mask_f.unsqueeze(-1)
    eps = 1e-8

    loss_s = (F.mse_loss(pred_s, states, reduction="none") * mask_3d).sum() / (mask_3d.sum() * states.shape[-1] + eps)
    loss_a = (F.mse_loss(pred_a, actions, reduction="none") * mask_3d).sum() / (mask_3d.sum() * actions.shape[-1] + eps)
    loss_r = (F.mse_loss(pred_r, rewards, reduction="none") * mask_3d).sum() / (mask_3d.sum() + eps)

    rtg_target = rtg[:, 1:, :]
    mask_rtg = mask_3d[:, 1:, :]
    loss_rtg = (F.mse_loss(pred_rtg[:, :-1, :], rtg_target, reduction="none") * mask_rtg).sum() / (mask_rtg.sum() + eps)

    return loss_s + loss_a + loss_r + loss_rtg


def train_model(model, dataset, n_steps, lr, batch_size, device, name="Model",
                log_freq=1000, warmup_steps=10_000, grad_clip=1.0):
    model.train().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(warmup_steps, 1), 1.0)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    loader_iter = iter(loader)
    losses = []
    for step in tqdm(range(1, n_steps + 1), desc=f"Training {name}"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        loss = compute_rt_loss(model, batch, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % log_freq == 0:
            print(f"[{name}] step={step}/{n_steps} loss={np.mean(losses[-log_freq:]):.4f}")
    return model


def train_vae_model(vae, dataset, n_steps, lr, batch_size, device, log_freq=2000):
    vae.train().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    loader_iter = iter(loader)
    losses = []
    for step in tqdm(range(1, n_steps + 1), desc="Training VAE"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        s, a, ns = batch["states"].to(device), batch["actions"].to(device), batch["next_states"].to(device)
        total, _, _ = vae.loss(s, a, ns)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        losses.append(total.item())
        if step % log_freq == 0:
            print(f"[VAE] step={step}/{n_steps} loss={np.mean(losses[-log_freq:]):.4f}")
    return vae


def train_classifier_model(cls, dataset, n_steps, lr, batch_size, device, log_freq=2000):
    cls.train().to(device)
    optimizer = torch.optim.Adam(cls.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    loader_iter = iter(loader)
    losses = []
    for step in tqdm(range(1, n_steps + 1), desc="Training Classifier"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        s = batch["states"].to(device)
        r = batch["rewards"].to(device).unsqueeze(-1)
        label = batch["high_reward"].to(device)
        loss = cls.loss(r, s, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % log_freq == 0:
            print(f"[Classifier] step={step}/{n_steps} loss={np.mean(losses[-log_freq:]):.4f}")
    return cls


def evaluate_policy(agent, env, n_episodes, state_mean, state_std):
    total = 0.0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_n = (obs - state_mean) / state_std
            a = agent.select_action(obs_n)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            obs, r, done, _ = env.step(a)
            ep_r += r
        total += ep_r
    return total / n_episodes


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.quick:
        print("=== QUICK MODE (reduced config) ===")
        args.rt_steps = 500
        args.vae_steps = 500
        args.cls_steps = 200
        args.n_generations = 20
        args.rl_steps = 500
        args.eval_freq = 200
        args.log_freq = 100
        args.d_model = 32
        args.n_heads = 2
        args.n_layers = 2
        args.d_ff = 64

    exp_name = f"{args.env}_{args.algo}_seed{args.seed}_{int(time.time())}"
    exp_dir = os.path.join(args.exp_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    device = args.device
    print(f"Device: {device}")

    # =========================================================================
    # Step 1: Load dataset
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Loading dataset {args.env}")
    print(f"{'='*60}")
    import gym
    import d4rl  # noqa
    raw_dataset, env = load_d4rl_dataset(args.env)
    env_eval = gym.make(args.env)
    trajectories = split_into_trajectories(raw_dataset)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"  Trajectories: {len(trajectories)}, state_dim: {state_dim}, action_dim: {action_dim}")

    state_mean, state_std = compute_state_stats(trajectories)
    np.save(os.path.join(exp_dir, "state_mean.npy"), state_mean)
    np.save(os.path.join(exp_dir, "state_std.npy"), state_std)

    # =========================================================================
    # Step 2: Train Forward + Backward RT
    # =========================================================================
    print(f"\n{'='*60}")
    print("Step 2: Training RT Transformers")
    print(f"{'='*60}")

    fwd_rt = RTTransformer(
        state_dim=state_dim, action_dim=action_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, context_len=args.context_len,
    )
    fwd_dataset = TrajectoryDataset(trajectories, args.context_len, args.gamma, state_mean, state_std)
    fwd_rt = train_model(fwd_rt, fwd_dataset, args.rt_steps, args.rt_lr, args.rt_batch_size,
                         device, name="FWD-RT", log_freq=args.log_freq)
    torch.save(fwd_rt.state_dict(), os.path.join(exp_dir, "forward_rt.pt"))

    bwd_rt = RTTransformer(
        state_dim=state_dim, action_dim=action_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, context_len=args.context_len,
    )
    rev_trajs = [{k: v[::-1].copy() for k, v in t.items()} for t in trajectories]
    bwd_dataset = TrajectoryDataset(rev_trajs, args.context_len, args.gamma, state_mean, state_std)
    bwd_rt = train_model(bwd_rt, bwd_dataset, args.rt_steps, args.rt_lr, args.rt_batch_size,
                         device, name="BWD-RT", log_freq=args.log_freq)
    torch.save(bwd_rt.state_dict(), os.path.join(exp_dir, "backward_rt.pt"))

    # =========================================================================
    # Step 3: Train VAE + compute α
    # =========================================================================
    print(f"\n{'='*60}")
    print("Step 3: Training VAE + Computing Reliability Threshold")
    print(f"{'='*60}")

    vae = TransitionVAE(state_dim=state_dim, action_dim=action_dim,
                        latent_dim=args.vae_latent_dim, hidden_dim=args.vae_hidden_dim)
    trans_dataset = TransitionDataset(trajectories, state_mean=state_mean, state_std=state_std)
    vae = train_vae_model(vae, trans_dataset, args.vae_steps, args.vae_lr, args.vae_batch_size, device)
    torch.save(vae.state_dict(), os.path.join(exp_dir, "vae.pt"))

    vae.eval()
    alpha = compute_max_vae_error(vae, trajectories, state_mean, state_std, device=device)
    print(f"  Reliability threshold α = {alpha:.6f}")
    np.save(os.path.join(exp_dir, "alpha.npy"), np.array(alpha))

    # =========================================================================
    # Step 4: Train High-Reward Classifier
    # =========================================================================
    print(f"\n{'='*60}")
    print("Step 4: Training High-Reward Classifier")
    print(f"{'='*60}")

    classifier = HighRewardClassifier(state_dim=state_dim, hidden_dim=args.cls_hidden_dim)
    classifier = train_classifier_model(classifier, trans_dataset, args.cls_steps,
                                        args.cls_lr, args.cls_batch_size, device)
    torch.save(classifier.state_dict(), os.path.join(exp_dir, "classifier.pt"))

    # =========================================================================
    # Step 5: Augment Dataset
    # =========================================================================
    print(f"\n{'='*60}")
    print("Step 5: Augmenting Dataset")
    print(f"{'='*60}")

    rel_estimator = ReliabilityEstimator(vae=vae, alpha=alpha, beta=args.beta_pessimism, device=device)
    augmentor = DataAugmentor(
        forward_rt=fwd_rt, backward_rt=bwd_rt,
        vae=vae, classifier=classifier,
        reliability_estimator=rel_estimator,
        state_dim=state_dim, action_dim=action_dim,
        state_mean=state_mean, state_std=state_std,
        gamma=args.gamma, generation_horizon=args.generation_horizon,
        candidate_K=args.candidate_K, context_len=args.context_len,
        device=device,
    )

    d_model = augmentor.augment_dataset(trajectories, n_generations=args.n_generations)
    print(f"  Generated {len(d_model)} synthetic trajectories")

    d_model_path = os.path.join(exp_dir, "d_model.pkl")
    with open(d_model_path, "wb") as f:
        pickle.dump(d_model, f)

    # =========================================================================
    # Step 6: Train Offline RL Policy on D = D_env ∪ D_model
    # =========================================================================
    print(f"\n{'='*60}")
    print("Step 6: Training Offline RL Policy")
    print(f"{'='*60}")

    env_transitions = trajectories_to_rl_transitions(trajectories, state_mean, state_std)
    if d_model:
        model_transitions = trajectories_to_rl_transitions(d_model, state_mean, state_std)
        combined = {k: np.concatenate([env_transitions[k], model_transitions[k]], axis=0)
                    for k in env_transitions}
    else:
        combined = env_transitions

    print(f"  Total transitions: {len(combined['observations'])}")
    rl_dataset = RLDataset(combined)
    rl_loader = DataLoader(rl_dataset, batch_size=args.rl_batch_size, shuffle=True,
                           drop_last=True, num_workers=0)

    if args.algo == "iql":
        agent = IQL(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=args.hidden_dim, n_layers=args.n_rl_layers,
            lr=args.rl_lr, gamma=args.gamma,
            tau_expectile=args.iql_tau, beta=args.iql_beta,
            device=device,
        )
    else:
        agent = BCQ(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=750, lr=args.rl_lr, gamma=args.gamma,
            phi=args.bcq_phi, lmbda=args.bcq_lmbda,
            device=device,
        )

    loader_iter = iter(rl_loader)
    rl_logs: dict = {}
    best_norm_score = -1e9
    scores_history = []

    for step in tqdm(range(1, args.rl_steps + 1), desc=f"Training {args.algo.upper()}"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(rl_loader)
            batch = next(loader_iter)

        info = agent.update(batch)
        for k, v in info.items():
            rl_logs.setdefault(k, []).append(v)

        if step % args.log_freq == 0:
            log_str = f"[{args.algo.upper()}] step={step}/{args.rl_steps}"
            for k, vs in rl_logs.items():
                log_str += f" | {k}={np.mean(vs[-args.log_freq:]):.4f}"
            print(log_str)

        if step % args.eval_freq == 0:
            raw_score = evaluate_policy(agent, env_eval, args.eval_episodes, state_mean, state_std)
            try:
                norm_score = env_eval.get_normalized_score(raw_score) * 100
            except Exception:
                norm_score = raw_score
            scores_history.append((step, norm_score))
            print(f"  Eval @ step {step}: raw={raw_score:.2f}, normalized={norm_score:.2f}")
            if norm_score > best_norm_score:
                best_norm_score = norm_score
                agent.save(os.path.join(exp_dir, f"{args.algo}_best.pt"))

    agent.save(os.path.join(exp_dir, f"{args.algo}_final.pt"))

    # =========================================================================
    # Step 7: Final Evaluation
    # =========================================================================
    print(f"\n{'='*60}")
    print("Step 7: Final Evaluation")
    print(f"{'='*60}")

    final_raw = evaluate_policy(agent, env_eval, args.eval_episodes * 2, state_mean, state_std)
    try:
        final_norm = env_eval.get_normalized_score(final_raw) * 100
    except Exception:
        final_norm = final_raw

    print(f"\n{'='*60}")
    print(f"RESULTS for {args.env} with {args.algo.upper()}")
    print(f"  Best normalized score: {best_norm_score:.2f}")
    print(f"  Final normalized score: {final_norm:.2f}")
    print(f"  Score history: {scores_history}")
    print(f"  Experiment dir: {exp_dir}")
    print(f"{'='*60}")

    # Save results
    results = {
        "env": args.env, "algo": args.algo, "seed": args.seed,
        "best_normalized_score": best_norm_score,
        "final_normalized_score": final_norm,
        "scores_history": scores_history,
    }
    import json
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

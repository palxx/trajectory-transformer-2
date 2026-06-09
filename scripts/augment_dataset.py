"""
Script 2: Augment offline dataset using backward RT generation.

Loads trained RT + VAE, runs the augmentation procedure (Algorithm 1, Steps 4-5),
and saves D_model to disk.

Usage:
    python scripts/augment_dataset.py --env hopper-medium-v2 --device cuda
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rt.config import RTConfig
from rt.data import (
    load_d4rl_dataset, split_into_trajectories, compute_state_stats,
    TransitionDataset,
)
from rt.models.transformer import RTTransformer
from rt.models.vae import TransitionVAE
from rt.models.classifier import HighRewardClassifier
from rt.reliability import ReliabilityEstimator
from rt.augmentor import DataAugmentor


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="hopper-medium-v2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--n_generations", type=int, default=50_000)
    p.add_argument("--generation_horizon", type=int, default=5)
    p.add_argument("--candidate_K", type=int, default=10)
    p.add_argument("--beta_pessimism", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--vae_latent_dim", type=int, default=32)
    p.add_argument("--vae_hidden_dim", type=int, default=256)
    p.add_argument("--cls_hidden_dim", type=int, default=256)
    p.add_argument("--cls_steps", type=int, default=20_000)
    p.add_argument("--cls_lr", type=float, default=1e-4)
    p.add_argument("--cls_batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_classifier(
    classifier: HighRewardClassifier,
    dataset: TransitionDataset,
    n_steps: int,
    lr: float,
    batch_size: int,
    device: str,
    log_freq: int = 2000,
) -> HighRewardClassifier:
    """Train the high-reward binary classifier."""
    classifier.train()
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
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

        # For classifier we use (reward_scalar, state) as input
        # We approximate RTG with raw reward for simplicity at classifier training
        loss = classifier.loss(r, s, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % log_freq == 0:
            import numpy as np
            print(f"[Classifier] Step {step}/{n_steps} | loss={np.mean(losses[-log_freq:]):.4f}")

    return classifier


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # Load dataset
    print(f"Loading dataset: {args.env}")
    dataset_raw, env = load_d4rl_dataset(args.env)
    trajectories = split_into_trajectories(dataset_raw)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load normalization stats
    state_mean = np.load(os.path.join(args.checkpoint_dir, "state_mean.npy"))
    state_std = np.load(os.path.join(args.checkpoint_dir, "state_std.npy"))
    alpha = float(np.load(os.path.join(args.checkpoint_dir, "alpha.npy")))
    print(f"  Reliability threshold α = {alpha:.6f}")

    # Load Forward RT
    print("Loading Forward RT...")
    fwd_rt = RTTransformer(
        state_dim=state_dim, action_dim=action_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, context_len=args.context_len,
    )
    fwd_rt.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "forward_rt.pt"), map_location=device))
    fwd_rt.eval()

    # Load Backward RT
    print("Loading Backward RT...")
    bwd_rt = RTTransformer(
        state_dim=state_dim, action_dim=action_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, context_len=args.context_len,
    )
    bwd_rt.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "backward_rt.pt"), map_location=device))
    bwd_rt.eval()

    # Load VAE
    print("Loading VAE...")
    vae = TransitionVAE(
        state_dim=state_dim, action_dim=action_dim,
        latent_dim=args.vae_latent_dim, hidden_dim=args.vae_hidden_dim,
    )
    vae.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "vae.pt"), map_location=device))
    vae.eval()

    # Train and/or load classifier
    cls_path = os.path.join(args.checkpoint_dir, "classifier.pt")
    classifier = HighRewardClassifier(state_dim=state_dim, hidden_dim=args.cls_hidden_dim)
    if os.path.exists(cls_path):
        print("Loading existing classifier...")
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
    else:
        print("Training Classifier...")
        trans_dataset = TransitionDataset(trajectories, state_mean=state_mean, state_std=state_std)
        classifier = train_classifier(
            classifier, trans_dataset, args.cls_steps, args.cls_lr,
            args.cls_batch_size, device,
        )
        torch.save(classifier.state_dict(), cls_path)
        print(f"Saved classifier to {cls_path}")
    classifier.eval()

    # Build reliability estimator
    rel_estimator = ReliabilityEstimator(vae=vae, alpha=alpha, beta=args.beta_pessimism, device=device)

    # Build augmentor
    augmentor = DataAugmentor(
        forward_rt=fwd_rt,
        backward_rt=bwd_rt,
        vae=vae,
        classifier=classifier,
        reliability_estimator=rel_estimator,
        state_dim=state_dim,
        action_dim=action_dim,
        state_mean=state_mean,
        state_std=state_std,
        gamma=args.gamma,
        generation_horizon=args.generation_horizon,
        candidate_K=args.candidate_K,
        context_len=args.context_len,
        device=device,
    )

    # Generate synthetic trajectories
    print(f"\nGenerating {args.n_generations} synthetic trajectories...")
    d_model = augmentor.augment_dataset(trajectories, n_generations=args.n_generations)
    print(f"  Generated {len(d_model)} synthetic trajectories")

    # Save D_model
    output_path = os.path.join(args.output_dir, f"{args.env}_d_model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(d_model, f)
    print(f"Saved D_model to {output_path}")

    # Also save D_env trajectories
    env_path = os.path.join(args.output_dir, f"{args.env}_d_env.pkl")
    with open(env_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved D_env to {env_path}")


if __name__ == "__main__":
    main()

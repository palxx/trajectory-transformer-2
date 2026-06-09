"""
Script 1: Train RT transformer (forward + backward) and VAE on D_env.

Usage:
    python scripts/train_dynamics.py --env hopper-medium-v2 --device cuda
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rt.config import RTConfig
from rt.data import (
    load_d4rl_dataset, split_into_trajectories,
    compute_state_stats, TrajectoryDataset,
)
from rt.models.transformer import RTTransformer
from rt.models.vae import TransitionVAE
from rt.data import TransitionDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="hopper-medium-v2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--rt_steps", type=int, default=100_000)
    p.add_argument("--rt_lr", type=float, default=1e-4)
    p.add_argument("--rt_batch_size", type=int, default=64)
    p.add_argument("--vae_steps", type=int, default=50_000)
    p.add_argument("--vae_lr", type=float, default=1e-4)
    p.add_argument("--vae_batch_size", type=int, default=256)
    p.add_argument("--vae_latent_dim", type=int, default=32)
    p.add_argument("--vae_hidden_dim", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_freq", type=int, default=1000)
    return p.parse_args()


def compute_rt_loss(
    model: RTTransformer,
    batch: dict,
    device: str,
) -> torch.Tensor:
    """MSE loss for all prediction heads."""
    rtg = batch["rtg"].to(device)
    states = batch["states"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    timesteps = batch["timesteps"].to(device)
    mask = batch["mask"].to(device)  # (B, T)

    pred_s, pred_a, pred_r, pred_rtg = model(rtg, states, actions, rewards, timesteps)

    # Losses over valid (non-padded) positions
    # pred_s[t] predicts s[t] (from R_t token)
    # pred_a[t] predicts a[t] (from s_t token)
    # pred_r[t] predicts r[t] (from a_t token)
    # pred_rtg[t] predicts R[t+1] (from r_t token) — valid for t < T-1

    mask_f = mask.float()
    mask_3d = mask_f.unsqueeze(-1)  # (B, T, 1)

    loss_s = (F.mse_loss(pred_s, states, reduction="none") * mask_3d).sum() / (mask_3d.sum() * states.shape[-1] + 1e-8)
    loss_a = (F.mse_loss(pred_a, actions, reduction="none") * mask_3d).sum() / (mask_3d.sum() * actions.shape[-1] + 1e-8)
    loss_r = (F.mse_loss(pred_r, rewards, reduction="none") * mask_3d).sum() / (mask_3d.sum() + 1e-8)

    # RTG prediction: pred_rtg[t] → R[t+1], valid for t < T-1
    rtg_target = rtg[:, 1:, :]  # (B, T-1, 1)
    mask_rtg = mask_3d[:, 1:, :]
    loss_rtg = (F.mse_loss(pred_rtg[:, :-1, :], rtg_target, reduction="none") * mask_rtg).sum() / (mask_rtg.sum() + 1e-8)

    return loss_s + loss_a + loss_r + loss_rtg


import torch.nn.functional as F


def train_transformer(
    model: RTTransformer,
    dataset: TrajectoryDataset,
    n_steps: int,
    lr: float,
    batch_size: int,
    device: str,
    log_freq: int = 1000,
    name: str = "RT",
    warmup_steps: int = 10_000,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
) -> RTTransformer:
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 1.0)
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
            print(f"[{name}] Step {step}/{n_steps} | loss={np.mean(losses[-log_freq:]):.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    return model


def train_vae(
    vae: TransitionVAE,
    dataset: TransitionDataset,
    n_steps: int,
    lr: float,
    batch_size: int,
    kl_weight: float,
    device: str,
    log_freq: int = 1000,
) -> TransitionVAE:
    vae.train()
    vae.to(device)
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

        s = batch["states"].to(device)
        a = batch["actions"].to(device)
        ns = batch["next_states"].to(device)

        total, recon, kl = vae.loss(s, a, ns, kl_weight=kl_weight)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        losses.append(total.item())
        if step % log_freq == 0:
            print(f"[VAE] Step {step}/{n_steps} | loss={np.mean(losses[-log_freq:]):.4f} | recon={recon.item():.4f} | kl={kl.item():.4f}")

    return vae


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = args.device

    print(f"Loading dataset: {args.env}")
    dataset_raw, env = load_d4rl_dataset(args.env)
    trajectories = split_into_trajectories(dataset_raw)
    print(f"  Loaded {len(trajectories)} trajectories")

    # State normalization
    state_mean, state_std = compute_state_stats(trajectories)
    np.save(os.path.join(args.checkpoint_dir, "state_mean.npy"), state_mean)
    np.save(os.path.join(args.checkpoint_dir, "state_std.npy"), state_std)
    print(f"  State dim: {state_mean.shape[0]}, Action dim: {env.action_space.shape[0]}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # -----------------------------------------------------------------------
    # Train Forward RT
    # -----------------------------------------------------------------------
    print("\n=== Training Forward RT Transformer ===")
    fwd_rt = RTTransformer(
        state_dim=state_dim, action_dim=action_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, context_len=args.context_len,
    )
    fwd_dataset = TrajectoryDataset(
        trajectories, context_len=args.context_len, gamma=args.gamma,
        state_mean=state_mean, state_std=state_std,
    )
    fwd_rt = train_transformer(
        fwd_rt, fwd_dataset, args.rt_steps, args.rt_lr, args.rt_batch_size,
        device, args.log_freq, name="FWD-RT",
    )
    torch.save(fwd_rt.state_dict(), os.path.join(args.checkpoint_dir, "forward_rt.pt"))
    print(f"Saved forward RT to {args.checkpoint_dir}/forward_rt.pt")

    # -----------------------------------------------------------------------
    # Train Backward RT (on reversed sequences)
    # -----------------------------------------------------------------------
    print("\n=== Training Backward RT Transformer ===")
    bwd_rt = RTTransformer(
        state_dim=state_dim, action_dim=action_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, context_len=args.context_len,
    )
    # For backward: reverse each trajectory before creating segments
    rev_trajectories = []
    for traj in trajectories:
        rev_traj = {k: v[::-1].copy() for k, v in traj.items()}
        rev_trajectories.append(rev_traj)

    bwd_dataset = TrajectoryDataset(
        rev_trajectories, context_len=args.context_len, gamma=args.gamma,
        state_mean=state_mean, state_std=state_std,
    )
    bwd_rt = train_transformer(
        bwd_rt, bwd_dataset, args.rt_steps, args.rt_lr, args.rt_batch_size,
        device, args.log_freq, name="BWD-RT",
    )
    torch.save(bwd_rt.state_dict(), os.path.join(args.checkpoint_dir, "backward_rt.pt"))
    print(f"Saved backward RT to {args.checkpoint_dir}/backward_rt.pt")

    # -----------------------------------------------------------------------
    # Train VAE
    # -----------------------------------------------------------------------
    print("\n=== Training Transition VAE ===")
    vae = TransitionVAE(
        state_dim=state_dim, action_dim=action_dim,
        latent_dim=args.vae_latent_dim, hidden_dim=args.vae_hidden_dim,
    )
    trans_dataset = TransitionDataset(trajectories, state_mean=state_mean, state_std=state_std)
    vae = train_vae(
        vae, trans_dataset, args.vae_steps, args.vae_lr, args.vae_batch_size,
        kl_weight=1.0, device=device, log_freq=args.log_freq,
    )
    torch.save(vae.state_dict(), os.path.join(args.checkpoint_dir, "vae.pt"))
    print(f"Saved VAE to {args.checkpoint_dir}/vae.pt")

    # -----------------------------------------------------------------------
    # Compute reliability threshold α = max VAE error on D_env
    # -----------------------------------------------------------------------
    print("\n=== Computing Reliability Threshold α ===")
    from rt.reliability import compute_max_vae_error
    alpha = compute_max_vae_error(vae, trajectories, state_mean, state_std, device=device)
    print(f"  α = {alpha:.6f}")
    np.save(os.path.join(args.checkpoint_dir, "alpha.npy"), np.array(alpha))
    print(f"Saved α to {args.checkpoint_dir}/alpha.npy")

    print("\nDone! All dynamics models trained and saved.")


if __name__ == "__main__":
    main()

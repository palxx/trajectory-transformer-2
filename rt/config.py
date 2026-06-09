"""
Configuration dataclass for all RT hyperparameters.
All hyperparameters are centralized here for easy tuning.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RTConfig:
    # -----------------------------------------------------------------------
    # Environment / Dataset
    # -----------------------------------------------------------------------
    env_name: str = "hopper-medium-v2"           # D4RL dataset name
    seed: int = 0
    device: str = "cuda"                          # "cuda" or "cpu"

    # -----------------------------------------------------------------------
    # Transformer architecture
    # -----------------------------------------------------------------------
    d_model: int = 128                            # embedding dimension
    n_heads: int = 4                              # number of attention heads
    n_layers: int = 3                             # number of transformer blocks
    d_ff: int = 512                               # feed-forward hidden dim
    dropout: float = 0.1
    max_ep_len: int = 1000                        # maximum episode length
    context_len: int = 20                         # context window (timesteps)

    # -----------------------------------------------------------------------
    # Transformer training
    # -----------------------------------------------------------------------
    rt_lr: float = 1e-4
    rt_batch_size: int = 64
    rt_train_steps: int = 100_000
    rt_warmup_steps: int = 10_000
    rt_weight_decay: float = 1e-4
    rt_grad_clip: float = 1.0

    # -----------------------------------------------------------------------
    # VAE architecture & training
    # -----------------------------------------------------------------------
    vae_latent_dim: int = 32
    vae_hidden_dim: int = 256
    vae_lr: float = 1e-4
    vae_batch_size: int = 256
    vae_train_steps: int = 50_000
    vae_kl_weight: float = 1.0

    # -----------------------------------------------------------------------
    # High-reward classifier
    # -----------------------------------------------------------------------
    cls_hidden_dim: int = 256
    cls_lr: float = 1e-4
    cls_batch_size: int = 256
    cls_train_steps: int = 20_000

    # -----------------------------------------------------------------------
    # Data augmentation
    # -----------------------------------------------------------------------
    n_generations: int = 50_000                   # number of synthetic trajectories
    generation_horizon: int = 5                   # H in paper (max backward steps)
    candidate_K: int = 10                         # number of RTG/state candidates per step
    beta_pessimism: float = 1.0                   # β for pessimistic reward penalty
    gamma: float = 0.99                           # discount factor

    # -----------------------------------------------------------------------
    # IQL hyperparameters
    # -----------------------------------------------------------------------
    iql_lr: float = 3e-4
    iql_batch_size: int = 256
    iql_train_steps: int = 1_000_000
    iql_tau: float = 0.7                          # expectile for value function
    iql_beta: float = 3.0                         # temperature for policy extraction
    iql_gamma: float = 0.99
    iql_target_update_freq: int = 1
    iql_polyak: float = 0.005                     # soft target update
    iql_hidden_dim: int = 256
    iql_n_layers: int = 2

    # -----------------------------------------------------------------------
    # BCQ hyperparameters
    # -----------------------------------------------------------------------
    bcq_lr: float = 1e-4
    bcq_batch_size: int = 100
    bcq_train_steps: int = 1_000_000
    bcq_phi: float = 0.05                         # max perturbation
    bcq_lmbda: float = 0.75                       # conservative Q mixing
    bcq_latent_dim_multiplier: int = 2            # latent = action_dim * 2
    bcq_hidden_dim: int = 750
    bcq_gamma: float = 0.99
    bcq_target_update_freq: int = 1
    bcq_polyak: float = 0.005

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    eval_freq: int = 10_000                       # evaluate every N gradient steps
    eval_episodes: int = 10
    normalize_reward: bool = True                 # normalize rewards using D4RL scores

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    log_freq: int = 1000

    # -----------------------------------------------------------------------
    # Offline RL algorithm selection
    # -----------------------------------------------------------------------
    offline_algo: str = "iql"                     # "iql" or "bcq"


@dataclass
class TrainingState:
    """Tracks mutable training state (step counts, best scores, etc.)."""
    rt_step: int = 0
    vae_step: int = 0
    cls_step: int = 0
    rl_step: int = 0
    best_normalized_score: float = -1e9

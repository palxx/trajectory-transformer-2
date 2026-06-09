"""
Dataset Augmentor for RT (Algorithm 1, Steps 4-5).

For each generation:
  1. Sample a trajectory τ from D_env
  2. Pick a random split point T' in τ
  3. Generate backward prefix τ̂ = {s_{T'-k}, a_{T'-k}, r_{T'-k}}_{k=0}^{T'}
     using backward RT + reliability estimation
  4. Merge: τ' = τ̂_{<T'} + τ_{≥T'}
  5. Add τ' to D_model

Backward generation:
  - Reverse the known suffix [T':end] to form the initial context for the backward model
  - Autoregressively extend backward until:
    a. Reached desired horizon H, OR
    b. Reliability check fails (U_t = 1)
"""

import numpy as np
import torch
import random
from typing import List, Optional, Tuple, Dict

from rt.models.transformer import RTTransformer
from rt.models.vae import TransitionVAE
from rt.models.classifier import HighRewardClassifier
from rt.reliability import ReliabilityEstimator
from rt.data import compute_rtg, normalize_states, denormalize_states


class DataAugmentor:
    """
    Generates synthetic trajectories using the backward RT transformer
    and appends them to the model buffer D_model.
    """

    def __init__(
        self,
        forward_rt: RTTransformer,
        backward_rt: RTTransformer,
        vae: TransitionVAE,
        classifier: HighRewardClassifier,
        reliability_estimator: ReliabilityEstimator,
        state_dim: int,
        action_dim: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        gamma: float = 0.99,
        generation_horizon: int = 5,
        candidate_K: int = 10,
        context_len: int = 20,
        device: str = "cpu",
    ):
        self.forward_rt = forward_rt.to(device)
        self.backward_rt = backward_rt.to(device)
        self.vae = vae.to(device)
        self.classifier = classifier.to(device)
        self.reliability = reliability_estimator
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.gamma = gamma
        self.H = generation_horizon
        self.K = candidate_K
        self.context_len = context_len
        self.device = device

        # Set all models to eval
        self.forward_rt.eval()
        self.backward_rt.eval()
        self.vae.eval()
        self.classifier.eval()

    @torch.no_grad()
    def _select_best_rtg_state(
        self,
        candidate_rtgs: torch.Tensor,   # (K, 1)
        candidate_states: torch.Tensor,  # (K, state_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use the high-reward classifier to pick the best (R_t, s_t) candidate.
        Returns: (rtg, state) each shape (1, ...)
        """
        probs = self.classifier.predict_proba(candidate_rtgs, candidate_states)  # (K,)
        best_idx = probs.argmax().item()
        return candidate_rtgs[best_idx:best_idx+1], candidate_states[best_idx:best_idx+1]

    @torch.no_grad()
    def _generate_backward_prefix(
        self,
        suffix_obs: np.ndarray,    # (T_suffix, state_dim) normalized
        suffix_acts: np.ndarray,   # (T_suffix, action_dim)
        suffix_rews: np.ndarray,   # (T_suffix,)
        suffix_rtg: np.ndarray,    # (T_suffix,)
        suffix_start_t: int,       # timestep index of first element in suffix
        horizon: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Generate a backward prefix of up to `horizon` steps.

        Returns lists of (obs, acts, rews, rtg) for generated steps,
        ordered from earliest to latest (i.e., prepend order).
        """
        device = self.device

        # Reverse the suffix to form initial context for backward model
        rev_obs = suffix_obs[::-1].copy()    # most recent first
        rev_acts = suffix_acts[::-1].copy()
        rev_rews = suffix_rews[::-1].copy()
        rev_rtg = suffix_rtg[::-1].copy()

        # Trim to context_len
        ctx_len = min(len(rev_obs), self.context_len)
        ctx_obs = rev_obs[:ctx_len]
        ctx_acts = rev_acts[:ctx_len]
        ctx_rews = rev_rews[:ctx_len]
        ctx_rtg = rev_rtg[:ctx_len]
        # Timesteps for reversed context: count downward from suffix_start_t - 1
        # The suffix starts at suffix_start_t, so the element just before suffix is at suffix_start_t-1
        # In the reversed context, position 0 = timestep suffix_start_t + T_suffix - 1 (most recent)
        T_suffix = len(suffix_obs)
        ctx_timesteps = np.arange(
            suffix_start_t + T_suffix - 1,
            suffix_start_t + T_suffix - 1 - ctx_len,
            -1,
            dtype=np.int64
        )

        gen_obs = []
        gen_acts = []
        gen_rews = []
        gen_rtg = []

        # Track context as tensors
        ctx_obs_t = torch.from_numpy(ctx_obs).float().to(device).unsqueeze(0)     # (1, ctx, state_dim)
        ctx_acts_t = torch.from_numpy(ctx_acts).float().to(device).unsqueeze(0)
        ctx_rews_t = torch.from_numpy(ctx_rews).float().to(device).unsqueeze(0).unsqueeze(-1)
        ctx_rtg_t = torch.from_numpy(ctx_rtg).float().to(device).unsqueeze(0).unsqueeze(-1)
        ctx_ts_t = torch.from_numpy(ctx_timesteps).long().to(device).unsqueeze(0)

        for step in range(horizon):
            # 1. Generate K candidate (R_t, s_t) using backward model
            candidate_rtgs = []
            candidate_states = []

            # Run backward model once to get one candidate
            pred_s, pred_a, pred_r, pred_R = self.backward_rt.generate_step(
                ctx_rtg_t, ctx_obs_t, ctx_acts_t, ctx_rews_t, ctx_ts_t
            )
            # pred_s is the "next" state in backward direction (i.e., earlier in time)
            # pred_R is the predicted RTG

            # Generate K candidates by adding noise to RTG and sampling states
            for k in range(self.K):
                noise_rtg = pred_R + torch.randn_like(pred_R) * (pred_R.abs().mean() * 0.1 + 0.01)
                candidate_rtgs.append(noise_rtg)
                candidate_states.append(pred_s)  # state prediction is deterministic

            candidate_rtgs = torch.cat(candidate_rtgs, dim=0)    # (K, 1)
            candidate_states = torch.cat(candidate_states, dim=0)  # (K, state_dim)

            # 2. Select best candidate using classifier
            best_rtg, best_state = self._select_best_rtg_state(candidate_rtgs, candidate_states)

            # 3. Generate action and reward for this new step using forward model
            #    Create a 1-step context with just the predicted (RTG, state)
            new_rtg_t = best_rtg.unsqueeze(0)     # (1, 1, 1)
            new_s_t = best_state.unsqueeze(0)      # (1, 1, state_dim)

            # Use forward model to predict action
            # We need a minimal context; use best_state and best_rtg
            new_ts = torch.tensor([[max(0, (suffix_start_t - step - 1))]], device=device)
            new_a_dummy = torch.zeros(1, 1, self.action_dim, device=device)
            new_r_dummy = torch.zeros(1, 1, 1, device=device)

            _, pred_act, pred_rew_fwd, _ = self.forward_rt.generate_step(
                new_rtg_t, new_s_t, new_a_dummy, new_r_dummy, new_ts
            )
            pred_rew = pred_rew_fwd  # (1, 1)

            # 4. Reliability check
            # Build context including the newly generated step
            # For reliability, we need next_state = current context's first state (reversed)
            next_s_for_rel = ctx_obs_t[:, :1, :]  # (1, 1, state_dim) — most recent in context

            # Simple single-step reliability: VAE error on (best_state, pred_act, next_s)
            vae_err = self.vae.reconstruction_error(
                best_state,     # (1, state_dim)
                pred_act,       # (1, action_dim)
                next_s_for_rel.squeeze(1),  # (1, state_dim)
            )  # (1,)
            gamma_val = vae_err  # simplified single-step reliability

            reliable = self.reliability.is_reliable(gamma_val)
            if not reliable.item():
                break  # stop backward generation

            # 5. Compute pessimistic reward
            raw_rew = pred_rew.squeeze(-1)  # (1,)
            pess_rew = self.reliability.pessimistic_reward(raw_rew, gamma_val, reliable)

            # 6. Store generated step
            gen_obs.append(best_state.squeeze(0).cpu().numpy())
            gen_acts.append(pred_act.squeeze(0).cpu().numpy())
            gen_rews.append(pess_rew.squeeze(0).cpu().numpy())
            gen_rtg.append(best_rtg.squeeze(0).cpu().numpy())

            # 7. Update context: prepend new step (reversed order means append to ctx)
            new_step_t = ctx_len + step + 1
            new_ts_val = max(0, suffix_start_t - step - 1)

            ctx_obs_t = torch.cat([
                best_state.unsqueeze(0),  # (1, 1, state_dim)
                ctx_obs_t
            ], dim=1)[:, :self.context_len]
            ctx_acts_t = torch.cat([
                pred_act.unsqueeze(0),
                ctx_acts_t
            ], dim=1)[:, :self.context_len]
            ctx_rews_t = torch.cat([
                pess_rew.unsqueeze(0).unsqueeze(-1),
                ctx_rews_t
            ], dim=1)[:, :self.context_len]
            ctx_rtg_t = torch.cat([
                best_rtg.unsqueeze(0),
                ctx_rtg_t
            ], dim=1)[:, :self.context_len]
            new_ts_tensor = torch.tensor([[new_ts_val]], device=device)
            ctx_ts_t = torch.cat([new_ts_tensor, ctx_ts_t], dim=1)[:, :self.context_len]

        # gen_obs etc. are in order: most-recently-generated first (i.e., earlier in time first)
        # Reverse to get chronological order (earliest first)
        gen_obs = gen_obs[::-1]
        gen_acts = gen_acts[::-1]
        gen_rews = gen_rews[::-1]
        gen_rtg = gen_rtg[::-1]

        return gen_obs, gen_acts, gen_rews, gen_rtg

    def generate_synthetic_trajectory(
        self, trajectories: List[dict]
    ) -> Optional[dict]:
        """
        Generate one synthetic trajectory by:
          1. Sampling a trajectory from D_env
          2. Picking a random split point T'
          3. Generating a backward prefix
          4. Merging prefix + suffix

        Returns dict with keys: observations, actions, rewards, terminals
        or None if generation fails.
        """
        if len(trajectories) == 0:
            return None

        # Sample trajectory
        traj = random.choice(trajectories)
        T = len(traj["observations"])

        if T < 3:
            return None

        # Random split point (leave at least 1 step on each side)
        T_prime = random.randint(1, T - 1)

        obs = traj["observations"].astype(np.float32)
        acts = traj["actions"].astype(np.float32)
        rews = traj["rewards"].astype(np.float32)
        terms = traj.get("terminals", np.zeros(T, dtype=np.float32)).astype(np.float32)

        # Normalize states
        obs_norm = normalize_states(obs, self.state_mean, self.state_std)

        # Compute RTG for the whole trajectory
        rtg = compute_rtg(rews, self.gamma)

        # Suffix: τ_{≥T'}
        suffix_obs = obs_norm[T_prime:]
        suffix_acts = acts[T_prime:]
        suffix_rews = rews[T_prime:]
        suffix_rtg = rtg[T_prime:]

        # Generate backward prefix
        horizon = min(self.H, T_prime)
        gen_obs, gen_acts, gen_rews, gen_rtg_list = self._generate_backward_prefix(
            suffix_obs=suffix_obs,
            suffix_acts=suffix_acts,
            suffix_rews=suffix_rews,
            suffix_rtg=suffix_rtg,
            suffix_start_t=T_prime,
            horizon=horizon,
        )

        if len(gen_obs) == 0:
            # No prefix generated; use original trajectory
            return {
                "observations": obs,
                "actions": acts,
                "rewards": rews,
                "terminals": terms,
            }

        # Merge: generated prefix (in original space) + suffix
        gen_obs_arr = np.stack(gen_obs, axis=0)  # (n_gen, state_dim)
        gen_acts_arr = np.stack(gen_acts, axis=0)
        gen_rews_arr = np.array([r.item() if hasattr(r, 'item') else float(r) for r in gen_rews])

        # Denormalize generated states
        gen_obs_denorm = denormalize_states(gen_obs_arr, self.state_mean, self.state_std)

        # Suffix in original space
        suffix_obs_orig = obs[T_prime:]
        suffix_rews_orig = rews[T_prime:]
        suffix_acts_orig = acts[T_prime:]
        suffix_terms_orig = terms[T_prime:]

        # Combined trajectory
        new_obs = np.concatenate([gen_obs_denorm, suffix_obs_orig], axis=0)
        new_acts = np.concatenate([gen_acts_arr, suffix_acts_orig], axis=0)
        new_rews = np.concatenate([gen_rews_arr, suffix_rews_orig], axis=0)
        new_terms = np.concatenate([
            np.zeros(len(gen_obs_denorm), dtype=np.float32),
            suffix_terms_orig
        ], axis=0)

        return {
            "observations": new_obs,
            "actions": new_acts,
            "rewards": new_rews,
            "terminals": new_terms,
        }

    def augment_dataset(
        self,
        trajectories: List[dict],
        n_generations: int,
    ) -> List[dict]:
        """
        Generate n_generations synthetic trajectories.
        Returns D_model as list of trajectory dicts.
        """
        d_model = []
        successes = 0
        attempts = 0

        while successes < n_generations:
            attempts += 1
            traj = self.generate_synthetic_trajectory(trajectories)
            if traj is not None and len(traj["observations"]) > 1:
                d_model.append(traj)
                successes += 1

            if attempts > n_generations * 10:
                print(f"[Augmentor] Warning: generated {successes}/{n_generations} after {attempts} attempts")
                break

        return d_model

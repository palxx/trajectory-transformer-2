"""
RTTransformer: Reliability-Guaranteed and Reward-Seeking Transformer.

Sequence format per timestep: [R_t, s_t, a_t, r_t]
Total tokens for T timesteps: 4*T

Prediction scheme (causal, same-position convention following DT):
  - output at R_t position  → predicts s_t
  - output at s_t position  → predicts a_t   (sees R_t via causal attention)
  - output at a_t position  → predicts r_t
  - output at r_t position  → predicts R_{t+1}

Both forward and backward transformers share the same architecture class.
The backward transformer is trained on time-reversed sequences.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Utility: causal self-attention with stored weights
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention that also returns attention weights."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Stored attention weights from last forward pass (for reliability)
        self.last_attn_weights: Optional[torch.Tensor] = None  # (B, H, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Causal mask: upper triangle = -inf
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # handle all-inf rows
        self.last_attn_weights = attn_weights.detach()          # store for reliability

        attn_weights_drop = self.attn_drop(attn_weights)
        out = torch.matmul(attn_weights_drop, V)                # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Main RTTransformer
# ---------------------------------------------------------------------------

class RTTransformer(nn.Module):
    """
    Reliability-Guaranteed and Reward-Seeking Transformer.

    Works for both forward and backward directions (same architecture).
    Backward: caller reverses sequences before/after passing through.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_ep_len: int = 1000,
        context_len: int = 20,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.context_len = context_len
        self.max_ep_len = max_ep_len

        # Input embeddings: separate projection for each token type
        # Token types: 0=RTG, 1=state, 2=action, 3=reward
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(action_dim, d_model)
        self.embed_reward = nn.Linear(1, d_model)

        # Positional / timestep embedding (shared across token types in same timestep)
        self.timestep_embed = nn.Embedding(max_ep_len, d_model)
        # Token-type embedding
        self.token_type_embed = nn.Embedding(4, d_model)

        self.embed_ln = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

        # Output heads — predict next element in sequence
        self.predict_state = nn.Linear(d_model, state_dim)
        self.predict_action = nn.Linear(d_model, action_dim)
        self.predict_reward = nn.Linear(d_model, 1)
        self.predict_rtg = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_sequence(
        self,
        rtg: torch.Tensor,        # (B, T, 1)
        states: torch.Tensor,     # (B, T, state_dim)
        actions: torch.Tensor,    # (B, T, action_dim)
        rewards: torch.Tensor,    # (B, T, 1)
        timesteps: torch.Tensor,  # (B, T) integer timestep indices
    ) -> torch.Tensor:
        """
        Interleave embeddings into (B, 4*T, d_model) sequence:
        [R_1, s_1, a_1, r_1, R_2, s_2, a_2, r_2, ...]
        """
        B, T, _ = states.shape
        device = states.device

        # Clamp timesteps to valid range
        timesteps = timesteps.clamp(0, self.max_ep_len - 1)

        # Compute per-token embeddings
        e_rtg = self.embed_rtg(rtg)                      # (B, T, d_model)
        e_state = self.embed_state(states)
        e_action = self.embed_action(actions)
        e_reward = self.embed_reward(rewards)

        # Timestep embedding (same for all 4 tokens at timestep t)
        t_emb = self.timestep_embed(timesteps)            # (B, T, d_model)

        # Token-type embeddings
        type_ids = torch.arange(4, device=device)         # [0,1,2,3]
        tt_emb = self.token_type_embed(type_ids)          # (4, d_model)
        tt_rtg, tt_state, tt_action, tt_reward = tt_emb[0], tt_emb[1], tt_emb[2], tt_emb[3]

        e_rtg = e_rtg + t_emb + tt_rtg
        e_state = e_state + t_emb + tt_state
        e_action = e_action + t_emb + tt_action
        e_reward = e_reward + t_emb + tt_reward

        # Stack: (B, T, 4, d_model) → (B, 4T, d_model)
        # Order per timestep: R, s, a, r
        stacked = torch.stack([e_rtg, e_state, e_action, e_reward], dim=2)  # (B,T,4,d)
        sequence = stacked.view(B, 4 * T, self.d_model)
        return sequence

    def forward(
        self,
        rtg: torch.Tensor,        # (B, T, 1)  return-to-go
        states: torch.Tensor,     # (B, T, state_dim)
        actions: torch.Tensor,    # (B, T, action_dim)
        rewards: torch.Tensor,    # (B, T, 1)
        timesteps: torch.Tensor,  # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns predictions aligned with the input sequence:
          pred_state  (B, T, state_dim)    ← from R_t output
          pred_action (B, T, action_dim)   ← from s_t output
          pred_reward (B, T, 1)            ← from a_t output
          pred_rtg    (B, T, 1)            ← from r_t output (predicts R_{t+1})
        """
        B, T, _ = states.shape
        x = self._build_sequence(rtg, states, actions, rewards, timesteps)
        x = self.embed_ln(x)
        x = self.embed_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)

        # x shape: (B, 4T, d_model)
        # Reshape to (B, T, 4, d_model)
        x = x.view(B, T, 4, self.d_model)

        pred_state = self.predict_state(x[:, :, 0, :])    # from R_t → predicts s_t
        pred_action = self.predict_action(x[:, :, 1, :])  # from s_t → predicts a_t
        pred_reward = self.predict_reward(x[:, :, 2, :])  # from a_t → predicts r_t
        pred_rtg = self.predict_rtg(x[:, :, 3, :])        # from r_t → predicts R_{t+1}

        return pred_state, pred_action, pred_reward, pred_rtg

    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Returns attention weights from the last layer's attention module.
        Shape: (B, n_heads, 4T, 4T), averaged over heads → (B, 4T, 4T).
        """
        last_attn = self.blocks[-1].attn.last_attn_weights
        if last_attn is None:
            return None
        # Average over heads
        return last_attn.mean(dim=1)  # (B, 4T, 4T)

    @torch.no_grad()
    def generate_step(
        self,
        rtg: torch.Tensor,        # (B, T, 1) context
        states: torch.Tensor,     # (B, T, state_dim) context
        actions: torch.Tensor,    # (B, T, action_dim) context
        rewards: torch.Tensor,    # (B, T, 1) context
        timesteps: torch.Tensor,  # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One-step generation: returns predicted (state, action, reward, next_rtg)
        for the LAST timestep in the context.
        """
        # Trim context to context_len
        if rtg.shape[1] > self.context_len:
            rtg = rtg[:, -self.context_len:]
            states = states[:, -self.context_len:]
            actions = actions[:, -self.context_len:]
            rewards = rewards[:, -self.context_len:]
            timesteps = timesteps[:, -self.context_len:]

        pred_state, pred_action, pred_reward, pred_rtg = self.forward(
            rtg, states, actions, rewards, timesteps
        )
        # Return predictions for the last timestep
        return (
            pred_state[:, -1],   # (B, state_dim)
            pred_action[:, -1],  # (B, action_dim)
            pred_reward[:, -1],  # (B, 1)
            pred_rtg[:, -1],     # (B, 1)
        )

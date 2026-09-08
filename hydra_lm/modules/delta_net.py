import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra_lm.config import HydraConfig


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet: O(1)-per-step linear attention (FR-4).

    Replaces O(N^2) self-attention with a fixed-size recurrent state S of
    shape (B, D, D). Memory and compute per generated token are constant
    regardless of context length.

    Pipeline (Implementation doc sec 3.4):
        x -> in_proj -> causal depthwise conv1d -> Q/K/V + SiLU
          -> beta (scalar write gate), gamma -> alpha = exp(-softplus(gamma))
          -> gated delta recurrence on state S
          -> read-out y_t = S_t @ q_t -> out_proj

    IMPORTANT: RoPE is NOT applied here. Only GQA layers use positional
    embeddings. The recurrent state order implicitly encodes position.

    This is the sequential reference implementation for correctness.
    Production code replaces the Python timestep loop with a chunked
    block-parallel scan kernel (Implementation doc pitfall #3).
    """

    def __init__(self, cfg: HydraConfig) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.d = d

        self.in_proj  = nn.Linear(d, 3 * d, bias=False)
        # Causal depthwise conv: pad=3 gives left-only padding for kernel_size=4
        self.conv1d   = nn.Conv1d(3 * d, 3 * d, kernel_size=4, groups=3 * d, padding=3)
        # Scalar gates: one value per batch per timestep
        self.beta_proj  = nn.Linear(d, 1, bias=False)   # write / learning-rate gate
        self.gamma_proj = nn.Linear(d, 1, bias=False)   # decay gate
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, state=None):
        """
        Args:
            x:     (B, T, D)
            state: (B, D, D) recurrent state, or None -> zero-initialised
        Returns:
            y:         (B, T, D)
            new_state: (B, D, D)
        """
        B, T, D = x.shape

        # Project + causal conv
        qkv = self.in_proj(x).transpose(1, 2)              # (B, 3D, T)
        qkv = self.conv1d(qkv)[..., :T].transpose(1, 2)    # causal crop -> (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                     # each (B, T, D)
        q = F.normalize(F.silu(q), p=2, dim=-1, eps=1e-6)
        k = F.normalize(F.silu(k), p=2, dim=-1, eps=1e-6)
        v = F.silu(v)

        # Gating parameters
        beta  = torch.sigmoid(self.beta_proj(x)).squeeze(-1)   # (B, T) write gate in (0,1)
        gamma = self.gamma_proj(x).squeeze(-1)                  # (B, T)
        alpha = torch.exp(-F.softplus(gamma))                   # (B, T) decay in (0,1)

        if state is None:
            state = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)

        # Sequential delta recurrence
        # At each step t:
        #   pred  = S_{t-1} @ k_t           (B, D)  -- prediction
        #   error = v_t - pred               (B, D)  -- delta
        #   S_t   = alpha_t * S + beta_t * outer(k_t, error)
        #   y_t   = S_t @ q_t               (B, D)  -- read-out
        outputs = []
        for t in range(T):
            k_t = k[:, t]    # (B, D)
            v_t = v[:, t]
            q_t = q[:, t]

            pred  = torch.einsum("bkd,bk->bd", state, k_t)   # S_{t-1} k_t
            error = v_t - pred

            a = alpha[:, t].view(B, 1, 1)
            b = beta[:, t].view(B, 1, 1)
            state = a * state + b * torch.einsum("bk,bd->bkd", k_t, error)

            y_t = torch.einsum("bkd,bk->bd", state, q_t)     # read-out
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)   # (B, T, D)
        return self.out_proj(y), state

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (FR-2).

    Used ONLY by full-attention (GQA) layers.
    Gated DeltaNet layers must NOT call this module.

    Supports 1-D position_ids (B, T) and mRoPE (num_axes, B, T).
    """

    def __init__(self, dim: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor):
        """
        Args:
            position_ids: (B, T)  or  (num_axes, B, T)
        Returns:
            cos, sin: each (..., T, dim)
        """
        freqs = torch.einsum("...i,j->...ij", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)   # duplicate across paired dims
        return emb.cos(), emb.sin()


def apply_rotary(q, k, cos, sin):
    """Apply rotary embeddings to Q and K tensors.

    Rotation is norm-preserving: ||q_rot|| == ||q|| within fp tolerance.

    Args:
        q, k: (B, T, heads, head_dim)
        cos, sin: (B, T, 1, head_dim)  (broadcast over heads)
    Returns:
        q_rot, k_rot: same shape as q, k
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

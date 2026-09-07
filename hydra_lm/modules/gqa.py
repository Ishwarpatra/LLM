import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra_lm.config import HydraConfig
from hydra_lm.modules.rms_norm import RMSNorm
from hydra_lm.modules.rope import RotaryEmbedding, apply_rotary


class GatedGQA(nn.Module):
    """Grouped Query Attention with QK-Norm and sigmoid output gating (FR-3).

    Design decisions (SRS + Implementation doc):
    - Q projection -> 2x head_dim per head: [query | gate]
    - QK-Norm (RMSNorm on Q and K individually) applied BEFORE RoPE
    - KV heads repeated via repeat_interleave to match Q group count
    - Elementwise sigmoid(gate) on attention output before o_proj
    """

    def __init__(self, cfg: HydraConfig) -> None:
        super().__init__()
        self.n_q   = cfg.num_query_heads
        self.n_kv  = cfg.num_kv_heads
        self.hd    = cfg.head_dim
        self.group = self.n_q // self.n_kv   # query groups per KV head

        # Q: 2x head_dim (query + gate); K, V: normal
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_q * self.hd * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd,    bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd,    bias=False)
        self.o_proj = nn.Linear(self.n_q * self.hd, cfg.hidden_size,     bias=False)

        # QK-Norm: separate RMSNorm per head for Q and K
        self.q_norm = RMSNorm(self.hd, eps=cfg.rms_eps)
        self.k_norm = RMSNorm(self.hd, eps=cfg.rms_eps)

        self.rope = RotaryEmbedding(self.hd, theta=cfg.rope_theta)

    def forward(
        self,
        x,
        position_ids,
        attn_mask=None,
        kv_cache=None,
    ):
        """
        Args:
            x:            (B, T, hidden_size)
            position_ids: (B, T)
            attn_mask:    (1, 1, T, S) bool, True=masked. None=no mask.
            kv_cache:     KVCache object or None
        Returns:
            (B, T, hidden_size)
        """
        B, T, _ = x.shape

        # Project
        q_and_gate = self.q_proj(x).view(B, T, self.n_q, self.hd * 2)
        q, gate = q_and_gate.chunk(2, dim=-1)       # each (B, T, n_q, hd)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd)

        # QK-Norm (before RoPE)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        cos, sin = self.rope(position_ids)           # (B, T, hd)
        cos = cos.unsqueeze(2)                       # (B, T, 1, hd) -> broadcast
        sin = sin.unsqueeze(2)
        q, k = apply_rotary(q, k, cos, sin)

        # KV cache update
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)            # -> (B, S, n_kv, hd)

        S = k.shape[1]

        # Expand KV heads to match Q groups (MUST happen after cache update)
        k = k.repeat_interleave(self.group, dim=2)  # (B, S, n_q, hd)
        v = v.repeat_interleave(self.group, dim=2)

        # Scaled dot-product attention
        q = q.transpose(1, 2)   # (B, n_q, T, hd)
        k = k.transpose(1, 2)   # (B, n_q, S, hd)
        v = v.transpose(1, 2)   # (B, n_q, S, hd)

        scores = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)  # (B, n_q, T, S)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn_w = scores.softmax(dim=-1)

        out = (attn_w.to(v.dtype) @ v).transpose(1, 2).reshape(B, T, self.n_q * self.hd)

        # Sigmoid gating
        gate_flat = gate.reshape(B, T, self.n_q * self.hd)
        out = out * torch.sigmoid(gate_flat)

        return self.o_proj(out)

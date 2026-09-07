"""Unit tests for RotaryEmbedding and apply_rotary (FR-2)."""
import torch
import pytest
from hydra_lm.modules.rope import RotaryEmbedding, apply_rotary


class TestRotaryEmbedding:
    def test_cos_sin_shapes(self, cfg, B, T):
        rope = RotaryEmbedding(cfg.head_dim, theta=cfg.rope_theta)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        assert cos.shape == (B, T, cfg.head_dim)
        assert sin.shape == (B, T, cfg.head_dim)

    def test_broadcast_against_qk(self, cfg, B, T):
        """cos/sin must broadcast correctly against Q/K (B, T, heads, head_dim)."""
        rope = RotaryEmbedding(cfg.head_dim, theta=cfg.rope_theta)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        cos = cos.unsqueeze(2)    # (B, T, 1, hd) -> broadcast over heads
        sin = sin.unsqueeze(2)

        q = torch.randn(B, T, cfg.num_query_heads, cfg.head_dim)
        k = torch.randn(B, T, cfg.num_kv_heads, cfg.head_dim)
        q_rot, k_rot = apply_rotary(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_preserves_norm(self, cfg, B, T):
        """Rotation is norm-preserving: ||q_rot|| == ||q|| within fp tolerance."""
        rope = RotaryEmbedding(cfg.head_dim, theta=cfg.rope_theta)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q = torch.randn(B, T, cfg.num_query_heads, cfg.head_dim)
        k = torch.randn(B, T, cfg.num_kv_heads, cfg.head_dim)
        q_rot, k_rot = apply_rotary(q, k, cos, sin)

        q_norm     = q.norm(dim=-1)
        q_rot_norm = q_rot.norm(dim=-1)
        assert torch.allclose(q_norm, q_rot_norm, atol=1e-5), (
            f"Norm not preserved: max diff {(q_norm - q_rot_norm).abs().max():.2e}"
        )

    def test_no_nans(self, cfg, B, T):
        rope = RotaryEmbedding(cfg.head_dim)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()

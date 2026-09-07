"""Unit tests for RMSNorm (FR-1)."""
import torch
import pytest
from hydra_lm.modules.rms_norm import RMSNorm


class TestRMSNorm:
    def test_output_shape(self, cfg, B, T):
        norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        x = torch.randn(B, T, cfg.hidden_size)
        out = norm(x)
        assert out.shape == (B, T, cfg.hidden_size), f"Expected {(B, T, cfg.hidden_size)}, got {out.shape}"

    def test_no_nans_float32(self, cfg, B, T):
        norm = RMSNorm(cfg.hidden_size)
        x = torch.randn(B, T, cfg.hidden_size)
        out = norm(x)
        assert not torch.isnan(out).any(), "NaNs in float32 output"

    def test_no_nans_float16(self, cfg, B, T):
        norm = RMSNorm(cfg.hidden_size)
        x = torch.randn(B, T, cfg.hidden_size).half()
        out = norm(x)
        assert not torch.isnan(out).any(), "NaNs in float16 output"

    def test_output_dtype_preserved(self, cfg, B, T):
        norm = RMSNorm(cfg.hidden_size)
        x = torch.randn(B, T, cfg.hidden_size).half()
        out = norm(x)
        assert out.dtype == torch.float16, "Output dtype should match input dtype"

    def test_numerical_stability_small_values(self, cfg, B, T):
        norm = RMSNorm(cfg.hidden_size, eps=1e-6)
        x = torch.full((B, T, cfg.hidden_size), 1e-8)
        out = norm(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_weight_learned(self, cfg, B, T):
        norm = RMSNorm(cfg.hidden_size)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (cfg.hidden_size,)

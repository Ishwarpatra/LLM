"""Unit tests for SwiGLU (FR-5)."""
import torch
import pytest
from hydra_lm.modules.swiglu import SwiGLU


@pytest.fixture
def mlp(cfg):
    return SwiGLU(cfg)


class TestSwiGLU:
    def test_output_shape(self, mlp, cfg, B, T):
        x = torch.randn(B, T, cfg.hidden_size)
        out = mlp(x)
        assert out.shape == (B, T, cfg.hidden_size), (
            f"Expected {(B, T, cfg.hidden_size)}, got {out.shape}"
        )

    def test_no_nans(self, mlp, cfg, B, T):
        x = torch.randn(B, T, cfg.hidden_size)
        assert not torch.isnan(mlp(x)).any()

    def test_intermediate_dim(self, cfg):
        mlp = SwiGLU(cfg)
        assert mlp.gate_proj.out_features == cfg.intermediate_size
        assert mlp.down_proj.out_features == cfg.hidden_size

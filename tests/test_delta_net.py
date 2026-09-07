"""Unit tests for GatedDeltaNet (FR-4)."""
import time
import torch
import pytest
from hydra_lm.config import HydraConfig
from hydra_lm.modules.delta_net import GatedDeltaNet


@pytest.fixture
def dnet(cfg):
    return GatedDeltaNet(cfg)


class TestGatedDeltaNet:
    def test_output_shape(self, dnet, cfg, B, T):
        x = torch.randn(B, T, cfg.hidden_size)
        y, state = dnet(x)
        assert y.shape == (B, T, cfg.hidden_size)

    def test_state_shape(self, dnet, cfg, B, T):
        x = torch.randn(B, T, cfg.hidden_size)
        _, state = dnet(x)
        assert state.shape == (B, cfg.hidden_size, cfg.hidden_size)

    def test_no_nans(self, dnet, cfg, B, T):
        x = torch.randn(B, T, cfg.hidden_size)
        y, state = dnet(x)
        assert not torch.isnan(y).any()
        assert not torch.isnan(state).any()

    def test_state_memory_constant(self, cfg, B):
        """State size must not grow with sequence length (O(1) memory per step)."""
        dnet = GatedDeltaNet(cfg)
        states = {}
        for T_len in [20, 100, 500]:
            x = torch.randn(B, T_len, cfg.hidden_size)
            _, state = dnet(x)
            states[T_len] = state.numel()

        # All states must be identical in size
        sizes = list(states.values())
        assert sizes[0] == sizes[1] == sizes[2], (
            f"State size changed with T: {states}"
        )

    def test_state_passthrough(self, dnet, cfg, B, T):
        """State from first half must be accepted and usable in second half.
        
        Note: due to the causal conv1d's padding history, chunked output != full
        output numerically (conv has no access to prior-chunk context). What we
        validate here is that:
          1. State passes between chunks without error.
          2. Second-chunk output shape is correct.
          3. Second-chunk output differs from running without prior state
             (proving the state is actually being used).
        """
        x = torch.randn(B, T, cfg.hidden_size)
        x1, x2 = x[:, :T//2], x[:, T//2:]

        # Run first half, get state
        y_a, state_a = dnet(x1)
        assert y_a.shape == (B, T//2, cfg.hidden_size)

        # Run second half with state from first half
        y_b_with_state, _ = dnet(x2, state=state_a)
        assert y_b_with_state.shape == (B, T//2, cfg.hidden_size)

        # Run second half without state (zero-init)
        y_b_no_state, _ = dnet(x2, state=None)

        # With state should differ from without state (state is being used)
        assert not torch.allclose(y_b_with_state, y_b_no_state, atol=1e-6), (
            "State passthrough had no effect — state is not being used"
        )

    def test_no_rope_dependency(self, dnet, cfg, B, T):
        """DeltaNet must work without any position_ids (no RoPE interface)."""
        x = torch.randn(B, T, cfg.hidden_size)
        # forward() must NOT require position_ids
        y, state = dnet(x)
        assert y is not None

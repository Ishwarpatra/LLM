"""Integration tests for HybridDecoderLayer (FR-6)."""
import torch
import pytest
from hydra_lm.config import HydraConfig
from hydra_lm.model.decoder_layer import HybridDecoderLayer
from hydra_lm.model.cache import KVCache, StateCache


class TestHybridDecoderLayer:
    def _mask(self, T):
        m = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        return m.unsqueeze(0).unsqueeze(0)

    def test_full_attention_shape(self, cfg, B, T):
        layer = HybridDecoderLayer(cfg, layer_idx=1)   # "full" in toy pattern
        assert layer.is_full_attn
        x = torch.randn(B, T, cfg.hidden_size)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        out, cache = layer(x, pos_ids, self._mask(T), None)
        assert out.shape == (B, T, cfg.hidden_size)
        assert isinstance(cache, KVCache)

    def test_linear_attention_shape(self, cfg, B, T):
        layer = HybridDecoderLayer(cfg, layer_idx=0)   # "linear" in toy pattern
        assert not layer.is_full_attn
        x = torch.randn(B, T, cfg.hidden_size)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        out, cache = layer(x, pos_ids, None, None)
        assert out.shape == (B, T, cfg.hidden_size)
        assert isinstance(cache, StateCache)

    def test_residual_path_exists(self, cfg, B, T):
        """Output must differ from the pre-norm input (residual is being added)."""
        layer = HybridDecoderLayer(cfg, layer_idx=0)
        x = torch.randn(B, T, cfg.hidden_size)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        out, _ = layer(x, pos_ids, None, None)
        assert not torch.allclose(out, x), "Output == input: residual path broken"

    def test_cache_reuse(self, cfg, B, T):
        """Passing the same cache a second time should not error."""
        layer = HybridDecoderLayer(cfg, layer_idx=0)  # linear
        x = torch.randn(B, T, cfg.hidden_size)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        out1, cache = layer(x, pos_ids, None, None)
        out2, cache = layer(x, pos_ids, None, cache)
        assert out2.shape == (B, T, cfg.hidden_size)

    def test_both_layer_types_no_nan(self, cfg, B, T):
        for idx in range(cfg.num_layers):
            layer = HybridDecoderLayer(cfg, layer_idx=idx)
            x = torch.randn(B, T, cfg.hidden_size)
            pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
            mask = self._mask(T) if layer.is_full_attn else None
            out, _ = layer(x, pos_ids, mask, None)
            assert not torch.isnan(out).any(), f"NaN in layer {idx}"

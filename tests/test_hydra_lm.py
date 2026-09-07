"""Smoke tests for the full HydraLM stack (FR-7)."""
import torch
import pytest
from hydra_lm.model.hydra_lm import HydraLM


@pytest.fixture
def model(cfg):
    return HydraLM(cfg)


class TestHydraLM:
    def test_forward_shape(self, model, cfg, B, T):
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        logits, caches = model(ids, pos_ids, mask)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert len(caches) == cfg.num_layers

    def test_weight_tying(self, model):
        """LM head must share weights with the embedding table."""
        assert model.lm_head.weight is model.embed.weight, (
            "Weight tying broken: lm_head.weight != embed.weight"
        )

    def test_generate_length(self, model, cfg, B):
        prompt = torch.randint(0, cfg.vocab_size, (B, 5))
        out = model.generate(prompt, max_new_tokens=10)
        assert out.shape == (B, 15), f"Expected (B, 15), got {out.shape}"

    def test_generate_no_nan(self, model, cfg, B):
        prompt = torch.randint(0, cfg.vocab_size, (B, 5))
        out = model.generate(prompt, max_new_tokens=10)
        assert not torch.isnan(out.float()).any()

    def test_generate_multiple_calls_no_corruption(self, model, cfg, B):
        """3 successive generate() calls must not corrupt each other."""
        prompt = torch.randint(0, cfg.vocab_size, (B, 3))
        for _ in range(3):
            out = model.generate(prompt, max_new_tokens=5)
            assert out.shape[0] == B
            assert not torch.isnan(out.float()).any()

    def test_generate_repetition_penalty(self, model, cfg, B):
        """repetition_penalty > 1.0 and top_k must generate valid non-NaN token tensor."""
        prompt = torch.randint(0, cfg.vocab_size, (B, 5))
        out = model.generate(prompt, max_new_tokens=10, repetition_penalty=1.5, top_k=20)
        assert out.shape == (B, 15)
        assert not torch.isnan(out.float()).any()

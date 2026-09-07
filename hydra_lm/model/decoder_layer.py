"""Hybrid Decoder Layer (FR-6)."""
import torch
import torch.nn as nn
from hydra_lm.config import HydraConfig
from hydra_lm.modules.rms_norm import RMSNorm
from hydra_lm.modules.gqa import GatedGQA
from hydra_lm.modules.delta_net import GatedDeltaNet
from hydra_lm.modules.swiglu import SwiGLU
from hydra_lm.model.cache import KVCache, StateCache


class HybridDecoderLayer(nn.Module):
    """Single hybrid decoder layer (FR-6).

    Pre-norm residual pattern:
        x -> attn_norm -> attention (GQA or DeltaNet) -> residual
          -> mlp_norm  -> SwiGLU                      -> residual

    Attention type is fixed at construction from cfg.layer_pattern[layer_idx]:
        "full"   -> GatedGQA   (with KVCache)
        "linear" -> GatedDeltaNet (with StateCache)
    """

    def __init__(self, cfg: HydraConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_full_attn: bool = cfg.layer_pattern[layer_idx] == "full"

        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.attn = GatedGQA(cfg) if self.is_full_attn else GatedDeltaNet(cfg)
        self.mlp_norm  = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, position_ids, attn_mask, cache):
        """
        Args:
            x:            (B, T, hidden_size)
            position_ids: (B, T)
            attn_mask:    (1,1,T,S) bool or None
            cache:        KVCache | StateCache | None
        Returns:
            x:     (B, T, hidden_size)
            cache: updated cache object
        """
        # -- Attention sub-layer --------------------------------------------
        residual = x
        h = self.attn_norm(x)

        if self.is_full_attn:
            if not isinstance(cache, KVCache):
                cache = KVCache()
            h = self.attn(h, position_ids, attn_mask, kv_cache=cache)
        else:
            if not isinstance(cache, StateCache):
                cache = StateCache()
            h, new_state = self.attn(h, state=cache.get())
            cache.set(new_state)

        x = residual + h

        # -- MLP sub-layer --------------------------------------------------
        residual = x
        x = residual + self.mlp(self.mlp_norm(x))

        return x, cache

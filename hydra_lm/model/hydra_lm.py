"""Full HYDRA-LM model with dual caching and generate() (FR-7, FR-8)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from hydra_lm.config import HydraConfig
from hydra_lm.modules.rms_norm import RMSNorm
from hydra_lm.model.decoder_layer import HybridDecoderLayer


def _causal_mask(seq_len: int, device) -> torch.Tensor:
    """Upper-triangular bool mask: True = masked (future tokens forbidden)."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)


class HydraLM(nn.Module):
    """Full HYDRA-LM decoder stack (FR-7 + FR-8).

    Architecture:
        token_embed -> N x HybridDecoderLayer -> final_norm -> lm_head

    lm_head weight is tied to the embedding table (weight tying).

    Inference uses dual caching:
        - KVCache   for full-attention (GQA) layers  -- grows with seq length
        - StateCache for Gated DeltaNet layers       -- constant size
    """

    def __init__(self, cfg: HydraConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed  = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            HybridDecoderLayer(cfg, i) for i in range(cfg.num_layers)
        )
        self.norm    = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Weight tying: lm_head shares weights with the embedding table
        self.lm_head.weight = self.embed.weight
        self.gradient_checkpointing = getattr(cfg, "gradient_checkpointing", False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, position_ids=None, attn_mask=None, caches=None):
        """
        Args:
            input_ids:    (B, T)
            position_ids: (B, T) or None (auto-generated if None)
            attn_mask:    (1,1,T,S) bool or None
            caches:       List[Cache] or None
        """
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        """
        Returns:
            logits: (B, T, vocab_size)
            caches: updated list
        """
        x = self.embed(input_ids)
        if caches is None:
            caches = [None] * len(self.layers)

        if self.training and self.gradient_checkpointing and caches[0] is None:
            for i, layer in enumerate(self.layers):
                def make_ckpt_fn(l):
                    def _forward(hidden, pos, mask):
                        out_hidden, _ = l(hidden, pos, mask, None)
                        return out_hidden
                    return _forward

                x = torch.utils.checkpoint.checkpoint(
                    make_ckpt_fn(layer),
                    x,
                    position_ids,
                    attn_mask,
                    use_reentrant=False,
                )
        else:
            for i, layer in enumerate(self.layers):
                x, caches[i] = layer(x, position_ids, attn_mask, caches[i])

        x = self.norm(x)
        return self.lm_head(x), caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.15,
        recent_window: int = 20,
    ) -> torch.Tensor:
        """Autoregressive generation with prefill + decode (FR-7, FR-8).

        Step 1 (prefill): process full prompt in one pass, seed all caches.
        Step 2 (decode):  generate one token at a time using cached state.

        Args:
            input_ids:          (B, T_prompt)
            max_new_tokens:     tokens to produce
            temperature:        softmax temperature
            top_k:              restrict to top-k logits (None = disabled)
            top_p:              nucleus probability threshold (None = disabled)
            repetition_penalty: base penalty factor (1.0 = none, >1.0 = penalize)
            recent_window:      token window for stronger immediate-context penalty
        Returns:
            (B, T_prompt + max_new_tokens)
        """
        device = input_ids.device
        B, prompt_len = input_ids.shape
        caches = None

        # Prefill
        pos_ids = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B, -1)
        mask    = _causal_mask(prompt_len, device)
        logits, caches = self.forward(input_ids, pos_ids, mask, caches)

        next_tok = self._sample(logits[:, -1], input_ids, temperature, top_k, top_p, repetition_penalty, recent_window)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

        # Decode
        for _ in range(max_new_tokens - 1):
            cur_len = input_ids.shape[1]
            pos_ids = torch.full((B, 1), cur_len - 1, device=device, dtype=torch.long)
            logits, caches = self.forward(input_ids[:, -1:], pos_ids, None, caches)
            next_tok = self._sample(logits[:, -1], input_ids, temperature, top_k, top_p, repetition_penalty, recent_window)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    @staticmethod
    def _sample(
        logits,
        history_ids,
        temperature=0.8,
        top_k=40,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.15,
        recent_window: int = 20,
    ):
        """(B, vocab) -> (B, 1) with two-tier repetition penalty + top-k/top-p.

        Penalty tiers:
          - Recent window (last `recent_window` tokens): penalty^count, uncapped.
            Stops immediate bigram/trigram loops.
          - Full history: penalty^min(count,3), capped to avoid crushing
            common Shakespeare words that appear legitimately many times.
        """
        logits = logits.clone()  # never mutate the caller's tensor

        if repetition_penalty != 1.0 and history_ids is not None:
            B = logits.shape[0]
            for b in range(B):
                full_hist = history_ids[b].tolist()
                recent = full_hist[-recent_window:] if len(full_hist) > recent_window else full_hist

                # Full-history pass (capped at exponent 3)
                full_counts: dict = {}
                for tok in full_hist:
                    full_counts[tok] = full_counts.get(tok, 0) + 1
                for tok, cnt in full_counts.items():
                    p = repetition_penalty ** min(cnt, 3)
                    logits[b, tok] = logits[b, tok] / p if logits[b, tok] > 0 else logits[b, tok] * p

                # Recent-window pass (uncapped — kills bigram loops)
                recent_counts: dict = {}
                for tok in recent:
                    recent_counts[tok] = recent_counts.get(tok, 0) + 1
                for tok, cnt in recent_counts.items():
                    p = repetition_penalty ** cnt
                    logits[b, tok] = logits[b, tok] / p if logits[b, tok] > 0 else logits[b, tok] * p

        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # top-k
        if top_k is not None and top_k > 0:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))

        # top-p nucleus (applied after top-k)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # remove tokens beyond the nucleus
            remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)



"""Dual cache system for autoregressive inference (FR-8).

KVCache  -- grows linearly with sequence length (standard KV cache)
StateCache -- constant size regardless of sequence length (DeltaNet recurrence)
"""
from typing import Optional
import torch


class KVCache:
    """Key-Value cache for full-attention (GQA) layers.

    K and V are (B, T, n_kv_heads, head_dim) and grow along dim=1.
    """

    def __init__(self) -> None:
        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Append new K/V; return the full accumulated tensors."""
        if self._k is None:
            self._k, self._v = k, v
        else:
            self._k = torch.cat([self._k, k], dim=1)
            self._v = torch.cat([self._v, v], dim=1)
        return self._k, self._v

    @property
    def seq_len(self) -> int:
        return 0 if self._k is None else self._k.shape[1]

    def reset(self) -> None:
        self._k = self._v = None


class StateCache:
    """Recurrent-state cache for Gated DeltaNet layers.

    Stores only the current state matrix S of shape (B, D, D).
    Memory is O(D^2), independent of sequence length.
    """

    def __init__(self) -> None:
        self._state: Optional[torch.Tensor] = None

    def get(self) -> Optional[torch.Tensor]:
        return self._state

    def set(self, state: torch.Tensor) -> None:
        self._state = state

    def reset(self) -> None:
        self._state = None

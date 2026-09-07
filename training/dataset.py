"""Token datasets for HYDRA-LM training.

Three dataset classes, matching the pipeline in FareedKhan-dev/train-llm-from-scratch:

  PretrainDataset  -- flat HDF5 array of token ids (from The Pile or similar).
                      Yields (input_ids, target_ids) by sliding a window.

  SFTDataset       -- instruction-tuning rows packed into fixed-length sequences.
                      Carries a loss_mask so we only train on assistant tokens.

  TextDataset      -- raw text fallback (no HDF5 needed), for quick smoke tests
                      and laptop-scale pre-training on a single text file.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import h5py
    _HAS_H5 = True
except ImportError:
    _HAS_H5 = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pretrain dataset (HDF5 flat token array)
# ─────────────────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """Flat token-id array stored in HDF5; each sample is a seq_len window.

    Based on FareedKhan-dev's data pipeline:
      data/pile_train.h5  ->  dataset["tokens"]  shape (N,)

    Each __getitem__ returns:
        input_ids  (seq_len,)   -- tokens[i : i+seq_len]
        target_ids (seq_len,)   -- tokens[i+1 : i+seq_len+1]  (next-token labels)
    """

    def __init__(self, h5_path: str, seq_len: int = 1024, stride: Optional[int] = None):
        """
        Args:
            h5_path:  path to the HDF5 file with a "tokens" dataset of shape (N,).
            seq_len:  context window length.
            stride:   how many tokens to advance between samples.
                      Defaults to seq_len (non-overlapping chunks).
                      Set stride=1 for maximum data usage (slow).
        """
        if not _HAS_H5:
            raise ImportError("h5py is required for PretrainDataset. Run: uv add h5py")
        import h5py as _h5
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.stride  = stride or seq_len

        with _h5.File(h5_path, "r") as f:
            self.n_tokens = f["tokens"].shape[0]

        self.n_samples = max(0, (self.n_tokens - seq_len - 1) // self.stride)

        # Lazy file handle -- opened once per worker in _get_handle()
        self._h5_handle = None

    def _get_handle(self):
        if self._h5_handle is None:
            import h5py as _h5
            self._h5_handle = _h5.File(self.h5_path, "r")
        return self._h5_handle

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = torch.tensor(
            self._get_handle()["tokens"][start : start + self.seq_len + 1],
            dtype=torch.long,
        )
        return chunk[:-1], chunk[1:]

    @staticmethod
    def create_h5(token_ids: List[int], out_path: str) -> None:
        """Utility: write a flat list of token ids to an HDF5 file."""
        import h5py as _h5
        import numpy as np
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with _h5.File(out_path, "w") as f:
            f.create_dataset("tokens", data=np.array(token_ids, dtype=np.int32),
                             compression="gzip", chunks=True)
        print(f"Saved {len(token_ids):,} tokens -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SFT dataset (instruction-tuning with loss mask)
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """Packed SFT rows with per-token loss masks.

    Chat format (same as FareedKhan-dev's chat_template.py):
        <|user|>
        {prompt}<|endoftext|><|assistant|>
        {completion}<|endoftext|>

    The loss mask is 1 only on the assistant completion tokens and the
    trailing <|endoftext|> — the model never trains on the prompt.

    Args:
        rows: list of dicts with keys "prompt" and "completion".
        tokenizer_fn: callable that encodes text -> list[int].
        seq_len: pad / truncate to this length.
        eot_id:  end-of-text token id (default 0 = vocab boundary).
    """

    def __init__(
        self,
        rows: List[Dict[str, str]],
        tokenizer_fn,
        seq_len: int = 512,
        eot_id:  int = 0,
        user_header:      str = "<|user|>\n",
        assistant_header: str = "<|assistant|>\n",
    ):
        self.seq_len           = seq_len
        self.eot_id            = eot_id
        self.tokenizer_fn      = tokenizer_fn
        self.user_header       = user_header
        self.assistant_header  = assistant_header
        self.samples = [self._encode(r) for r in rows]

    def _encode(self, row: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, loss_mask), each of shape (seq_len,)."""
        tok = self.tokenizer_fn

        # Prompt part — not trained on
        prompt_ids = (
            tok(self.user_header)
            + tok(row["prompt"])
            + [self.eot_id]
            + tok(self.assistant_header)
        )
        prompt_mask = [0] * len(prompt_ids)

        # Completion part — trained on
        comp_ids   = tok(row["completion"]) + [self.eot_id]
        comp_mask  = [1] * len(comp_ids)

        ids  = (prompt_ids + comp_ids)[: self.seq_len]
        mask = (prompt_mask + comp_mask)[: self.seq_len]

        # Pad to seq_len
        pad_len = self.seq_len - len(ids)
        ids  = ids  + [0] * pad_len
        mask = mask + [0] * pad_len

        return (
            torch.tensor(ids,  dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, mask = self.samples[idx]
        return ids[:-1], ids[1:], mask[1:]   # input, target, loss_mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. Raw text fallback (no HDF5 dependency)
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Minimal dataset from a raw text file. No external dependencies.

    Tokenises with a simple character-level or word-level split depending on
    the tokenizer_fn provided. When tokenizer_fn is None, uses char-level
    encoding (useful for smoke tests with vocab_size=256).

    Matches the "input_ids, target_ids" contract of PretrainDataset.
    """

    def __init__(self, text: str, seq_len: int = 256, tokenizer_fn=None):
        if tokenizer_fn is not None:
            tokens = tokenizer_fn(text)
        else:
            # Char-level: each byte maps to an id in 0..255
            tokens = list(text.encode("utf-8", errors="replace"))

        self.data    = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.n       = max(0, len(tokens) - seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

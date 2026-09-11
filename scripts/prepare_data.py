"""Data preparation script for HYDRA-LM.

Tokenizes a raw corpus (e.g. WikiText-103) using a trained HF BPE tokenizer
(or tiktoken as fallback) and saves token IDs directly to HDF5 format
for PretrainDataset.

Usage:
    python scripts/prepare_data.py \
        --corpus data/raw/wikitext103.txt \
        --tokenizer tokenizers/hydra_bpe \
        --out data/wikitext103.h5
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Run: pip install h5py")
    sys.exit(1)


class TokenizerAdapter:
    """Unified wrapper around HuggingFace Tokenizers and tiktoken."""

    def __init__(self, backend, vocab_size: int, is_hf: bool = True):
        self.backend = backend
        self.vocab_size = vocab_size
        self.is_hf = is_hf

    def encode(self, text: str) -> List[int]:
        if self.is_hf:
            res = self.backend.encode(text)
            return res.ids if hasattr(res, "ids") else list(res)
        else:
            return self.backend.encode(text, allowed_special={"<|endoftext|>"})

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        if self.is_hf and hasattr(self.backend, "encode_batch"):
            encodings = self.backend.encode_batch(texts)
            return [enc.ids if hasattr(enc, "ids") else list(enc) for enc in encodings]
        return [self.encode(t) for t in texts]

    def decode(self, ids: List[int]) -> str:
        return self.backend.decode(ids)


def load_tokenizer(tokenizer_path_or_name: str) -> TokenizerAdapter:
    """Load tokenizer from HF directory/file or fallback to tiktoken."""
    tok_path = Path(tokenizer_path_or_name)

    # 1. HuggingFace tokenizer path
    if tok_path.exists():
        # Check for tokenizer.json (fast Tokenizer)
        json_file = tok_path / "tokenizer.json" if tok_path.is_dir() else tok_path
        if json_file.is_file() and json_file.suffix == ".json":
            try:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(json_file))
                vocab_size = tok.get_vocab_size()
                print(f"Loaded HF Tokenizer from {json_file} (vocab_size={vocab_size:,})", flush=True)
                return TokenizerAdapter(tok, vocab_size, is_hf=True)
            except Exception as e:
                print(f"Warning: Failed loading with Tokenizer.from_file ({e}), trying ByteLevelBPETokenizer...")

        # Check for vocab.json + merges.txt
        vocab_file = tok_path / "vocab.json" if tok_path.is_dir() else tok_path.parent / "vocab.json"
        merges_file = tok_path / "merges.txt" if tok_path.is_dir() else tok_path.parent / "merges.txt"
        if vocab_file.is_file() and merges_file.is_file():
            try:
                from tokenizers import ByteLevelBPETokenizer
                tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
                vocab_size = tok.get_vocab_size()
                print(f"Loaded ByteLevelBPETokenizer from {tok_path} (vocab_size={vocab_size:,})", flush=True)
                return TokenizerAdapter(tok, vocab_size, is_hf=True)
            except Exception as e:
                print(f"Warning: Failed loading ByteLevelBPETokenizer: {e}")

    # 2. Tiktoken encoding name
    try:
        import tiktoken
        tok = tiktoken.get_encoding(tokenizer_path_or_name)
        vocab_size = tok.n_vocab
        print(f"Loaded tiktoken encoding '{tokenizer_path_or_name}' (vocab_size={vocab_size:,})", flush=True)
        return TokenizerAdapter(tok, vocab_size, is_hf=False)
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load tokenizer from '{tokenizer_path_or_name}'. "
        f"Ensure the path exists or is a valid tiktoken encoding (e.g. 'gpt2')."
    )


def parse_args():
    p = argparse.ArgumentParser(description="Prepare HDF5 dataset for HYDRA-LM pretraining")
    default_corpus = "data/raw/corpus.txt" if Path("data/raw/corpus.txt").exists() else "data/raw/wikitext103.txt"
    p.add_argument("--corpus", default=default_corpus,
                   help="Path to raw text corpus (default: data/raw/corpus.txt or data/raw/wikitext103.txt)")
    p.add_argument("--tokenizer", default="tokenizers/hydra_bpe",
                   help="Path to HF tokenizer dir/file or tiktoken encoding name")
    default_out = "data/corpus.h5" if Path("data/raw/corpus.txt").exists() else "data/wikitext103.h5"
    p.add_argument("--out", default=default_out,
                   help="Path to output HDF5 file")
    p.add_argument("--batch_lines", type=int, default=10000,
                   help="Batch size (in lines) for tokenization")
    return p.parse_args()


def main():
    args = parse_args()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        if (PROJECT_ROOT / args.corpus).exists():
            corpus_path = PROJECT_ROOT / args.corpus
        else:
            alt_corpus = Path("data/raw/corpus.txt") if corpus_path.name == "wikitext103.txt" else Path("data/raw/wikitext103.txt")
            if alt_corpus.exists():
                print(f"Note: '{corpus_path}' not found, using '{alt_corpus}'")
                corpus_path = alt_corpus
            elif (PROJECT_ROOT / alt_corpus).exists():
                corpus_path = PROJECT_ROOT / alt_corpus
            else:
                print(f"ERROR: Corpus file not found: {corpus_path}")
                print("Download the corpus first with:")
                print("  python scripts/download_wikitext103.py --out data/raw/corpus.txt")
                sys.exit(1)

    out_path = Path(args.out)
    if not out_path.is_absolute() and not out_path.parent.exists() and (PROJECT_ROOT / out_path.parent).exists():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok_target = args.tokenizer
    if not Path(tok_target).exists() and (PROJECT_ROOT / tok_target).exists():
        tok_target = str(PROJECT_ROOT / tok_target)

    if corpus_path.stat().st_size == 0:
        print(f"\n[ERROR] Corpus file '{corpus_path}' is empty (0 bytes)!")
        print("Please download/extract the corpus first:")
        print(f"  python scripts/download_wikitext103.py --out {corpus_path}\n")
        sys.exit(1)

    print(f"Corpus: {corpus_path} ({corpus_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Tokenizer: {tok_target}")
    print(f"Output: {out_path}")

    tokenizer = load_tokenizer(tok_target)

    t0 = time.time()
    total_tokens = 0
    total_chars = 0
    lines_batch = []

    # Stream write directly into HDF5 in resizable chunks
    with h5py.File(str(out_path), "w") as h5f:
        dset = h5f.create_dataset(
            "tokens",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(65536,),
            compression="gzip",
        )

        def flush_batch(batch: List[str]):
            nonlocal total_tokens, total_chars
            if not batch:
                return
            batch_tokens_list = tokenizer.encode_batch(batch)
            all_ids = []
            for ids in batch_tokens_list:
                all_ids.extend(ids)

            if all_ids:
                old_len = dset.shape[0]
                new_len = old_len + len(all_ids)
                dset.resize((new_len,))
                dset[old_len:new_len] = np.array(all_ids, dtype=np.int32)
                total_tokens += len(all_ids)

        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f, 1):
                if line.strip():
                    lines_batch.append(line)
                    total_chars += len(line)

                if len(lines_batch) >= args.batch_lines:
                    flush_batch(lines_batch)
                    lines_batch = []
                    elapsed = time.time() - t0
                    print(
                        f"  Processed {line_idx:,} lines | {total_tokens:,} tokens "
                        f"({total_tokens / max(1e-5, elapsed):,.0f} tok/s) ...",
                        flush=True,
                    )

            if lines_batch:
                flush_batch(lines_batch)

    elapsed = time.time() - t0
    compression_ratio = total_chars / max(1, total_tokens)
    print("\n" + "=" * 60, flush=True)
    print(f"Successfully prepared HDF5 dataset at {out_path}", flush=True)
    print(f"Total tokens: {total_tokens:,}", flush=True)
    print(f"Total chars:  {total_chars:,} (compression: {compression_ratio:.2f} chars/token)", flush=True)
    print(f"Time taken:   {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

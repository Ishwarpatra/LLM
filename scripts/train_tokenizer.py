"""Train a BPE tokenizer on a custom corpus.

Usage:
    python scripts/train_tokenizer.py \\
        --corpus data/tinyshakespeare.txt \\
        --vocab_size 8000 \\
        --out_dir tokenizer/

Requires:
    uv add tokenizers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train a BPE tokenizer for HYDRA-LM")
    p.add_argument("--corpus", required=True, help="Path to raw text file")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--out_dir", default="tokenizer/", help="Directory to save vocab + merges")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        print("ERROR: 'tokenizers' package not found. Run: uv add tokenizers")
        sys.exit(1)

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"ERROR: corpus file not found: {corpus}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE tokenizer on {corpus} (vocab_size={args.vocab_size}) …")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
    )

    tokenizer.save_model(str(out_dir))
    print(f"Saved tokenizer vocab + merges to {out_dir}/")
    print(f"vocab_size={args.vocab_size}  min_frequency={args.min_frequency}")
    print(f"\nUpdate hydra_lm/config.py: vocab_size={args.vocab_size}")


if __name__ == "__main__":
    main()

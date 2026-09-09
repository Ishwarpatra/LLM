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
import io
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train a BPE tokenizer for HYDRA-LM")
    default_corpus = "data/raw/corpus.txt" if Path("data/raw/corpus.txt").exists() else ("data/raw/wikitext103.txt" if Path("data/raw/wikitext103.txt").exists() else "data/raw/corpus.txt")
    p.add_argument("--corpus", default=default_corpus, help="Path to raw text file (default: data/raw/corpus.txt or data/raw/wikitext103.txt)")
    p.add_argument("--vocab_size", type=int, default=12000, help="Target vocabulary size (default: 12000)")
    p.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    p.add_argument("--out", "--out_dir", dest="out_dir", default="tokenizers/hydra_bpe",
                   help="Directory to save vocab + merges and tokenizer.json")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        print("ERROR: 'tokenizers' package not found. Run: uv add tokenizers (or pip install tokenizers)")
        sys.exit(1)

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"ERROR: corpus file not found: {corpus}")
        print("To download the real corpus, run:")
        print("    python scripts/download_wikitext103.py --out data/raw/corpus.txt")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE tokenizer on {corpus} (target vocab_size={args.vocab_size}) …")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
    )

    tokenizer.save_model(str(out_dir))
    tokenizer.save(str(out_dir / "tokenizer.json"))
    actual_vocab = tokenizer.get_vocab_size()
    print(f"Saved tokenizer files (vocab.json, merges.txt, tokenizer.json) to {out_dir}/")
    print(f"Target vocab_size: {args.vocab_size} | Actual vocab_size: {actual_vocab}")
    print(f"\nUpdate hydra_lm/config.py: vocab_size = {actual_vocab}")


if __name__ == "__main__":
    main()

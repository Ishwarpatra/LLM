"""Download WikiText-103 and write it to a single flat text file.

Uses the HuggingFace datasets library to stream the corpus so it works
even on machines where the full download is large.

Usage:
    python scripts/download_wikitext103.py --out data/raw/wikitext103.txt

Output:
    A single UTF-8 text file with one article per line (blank lines stripped).
    Typical size: ~500 MB, ~103M tokens with our BPE vocab.

Requires:
    uv add datasets
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
    p = argparse.ArgumentParser(description="Download WikiText-103 to a flat text file")
    p.add_argument("--out", default="data/raw/wikitext103.txt",
                   help="Output path for the flat text file")
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test"],
                   help="Dataset split to download")
    p.add_argument("--max_articles", type=int, default=None,
                   help="Cap article count (useful for smoke tests, e.g. --max_articles 5000)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: uv add datasets")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading WikiText-103 ({args.split} split) ...", flush=True)
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split)
    except Exception as e:
        print(f"Direct load failed ({e}), using streaming mode ...", flush=True)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split, streaming=True)

    n_lines = 0
    n_chars = 0
    write_buffer = []

    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            write_buffer.append(text + "\n")
            n_chars += len(text)
            n_lines += 1

            if len(write_buffer) >= 10_000:
                f.writelines(write_buffer)
                write_buffer = []
                print(f"  ... {n_lines:,} lines, {n_chars/1e6:.1f} MB written", flush=True)

            if args.max_articles and n_lines >= args.max_articles:
                break

        if write_buffer:
            f.writelines(write_buffer)

    print(f"\nDone. {n_lines:,} lines, {n_chars:,} chars -> {out}", flush=True)
    print(f"Next: python scripts/train_tokenizer.py --corpus {out} --vocab_size 12000 --out tokenizers/hydra_bpe", flush=True)


if __name__ == "__main__":
    main()

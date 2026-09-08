"""Download WikiText-103 and write it to a single flat text file.

Downloads the official WikiText-103 raw parquet files directly from HuggingFace
via HTTPS and extracts the text column without depending on Hugging Face Hub's
buggy legacy repo resolution.

Usage:
    python scripts/download_wikitext103.py --out data/raw/wikitext103.txt
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PARQUET_URLS = {
    "train": [
        "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet",
    ],
    "validation": [
        "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet",
    ],
    "test": [
        "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Download WikiText-103 to a flat text file")
    p.add_argument("--out", default="data/raw/wikitext103.txt",
                   help="Output path for the flat text file")
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test"],
                   help="Dataset split to download")
    p.add_argument("--max_articles", type=int, default=None,
                   help="Cap article/line count (useful for smoke tests)")
    return p.parse_args()


def download_file(url: str, dest_path: Path):
    print(f"Downloading {dest_path.name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = (downloaded / total) * 100
                print(f"\r  Downloaded {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({percent:.1f}%)", end="", flush=True)
    print()


def extract_with_pyarrow(urls: list[str], out_path: Path, max_lines: int | None = None) -> tuple[int, int]:
    import pyarrow.parquet as pq

    cache_dir = Path("data/raw/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    n_chars = 0
    write_buffer = []

    with out_path.open("w", encoding="utf-8") as f:
        for url in urls:
            filename = url.split("/")[-1]
            local_parquet = cache_dir / filename
            if not local_parquet.exists() or local_parquet.stat().st_size == 0:
                download_file(url, local_parquet)

            print(f"Extracting lines from {filename} ...", flush=True)
            table = pq.read_table(str(local_parquet), columns=["text"])
            for chunk in table["text"].chunks:
                for text_val in chunk.to_pylist():
                    clean_text = text_val.strip() if text_val else ""
                    if not clean_text:
                        continue
                    write_buffer.append(clean_text + "\n")
                    n_chars += len(clean_text)
                    n_lines += 1

                    if len(write_buffer) >= 20_000:
                        f.writelines(write_buffer)
                        write_buffer = []
                        print(f"  ... {n_lines:,} lines, {n_chars/1e6:.1f} MB written", flush=True)

                    if max_lines and n_lines >= max_lines:
                        break
                if max_lines and n_lines >= max_lines:
                    break
            if max_lines and n_lines >= max_lines:
                break

        if write_buffer:
            f.writelines(write_buffer)

    return n_lines, n_chars


def extract_with_datasets(urls: list[str], out_path: Path, max_lines: int | None = None) -> tuple[int, int]:
    from datasets import load_dataset
    print("Loading parquet files with datasets ...", flush=True)
    ds = load_dataset("parquet", data_files=urls, split="train")

    n_lines = 0
    n_chars = 0
    write_buffer = []

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            clean_text = row["text"].strip() if row["text"] else ""
            if not clean_text:
                continue
            write_buffer.append(clean_text + "\n")
            n_chars += len(clean_text)
            n_lines += 1

            if len(write_buffer) >= 20_000:
                f.writelines(write_buffer)
                write_buffer = []
                print(f"  ... {n_lines:,} lines, {n_chars/1e6:.1f} MB written", flush=True)

            if max_lines and n_lines >= max_lines:
                break

        if write_buffer:
            f.writelines(write_buffer)

    return n_lines, n_chars


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    urls = PARQUET_URLS.get(args.split)
    if not urls:
        print(f"ERROR: Unknown split '{args.split}'")
        sys.exit(1)

    print(f"Extracting WikiText-103 ({args.split} split) -> {out} ...", flush=True)

    try:
        n_lines, n_chars = extract_with_pyarrow(urls, out, max_lines=args.max_articles)
    except Exception as e:
        print(f"pyarrow direct extraction failed ({e}), falling back to datasets parquet loader ...", flush=True)
        n_lines, n_chars = extract_with_datasets(urls, out, max_lines=args.max_articles)

    print(f"\nDone. {n_lines:,} lines, {n_chars:,} chars -> {out}", flush=True)
    print(f"Next: python scripts/prepare_data.py --corpus {out} --tokenizer tokenizers/hydra_bpe --out data/wikitext103.h5", flush=True)


if __name__ == "__main__":
    main()

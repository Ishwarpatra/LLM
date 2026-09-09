"""Project Gutenberg Data Pipeline for HYDRA-LM.

Handles:
1. Paced, cached HTTP acquisition of curated public-domain books from Project Gutenberg.
2. Robust header/footer boilerplate stripping.
3. Text normalization (line endings, hyphenation rejoining, space collapse).
4. SHA-256 fingerprint deduplication.
5. Concatenation with document boundaries (<|endoftext|>).
6. Custom ByteLevelBPETokenizer training (12,000 vocab).
7. HDF5 tokenization for PretrainDataset.
8. Sanity check: chars/token ratio, document stats, decoded spot-checks.

Usage:
    # Download curated books, process text, train tokenizer, and create HDF5:
    python scripts/prepare_gutenberg.py --download --out data/raw/corpus.txt --train_tokenizer --to_h5 data/gutenberg.h5 --sanity_check

    # Process an existing directory of raw .txt files:
    python scripts/prepare_gutenberg.py --input_dir data/raw/gutenberg_raw --out data/raw/corpus.txt --train_tokenizer --to_h5 data/gutenberg.h5
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Special boundary token
DOC_SEP = "<|endoftext|>"

# Curated catalog of classic public-domain literature (Project Gutenberg IDs)
CURATED_BOOKS: Dict[int, Dict[str, str]] = {
    1342: {"title": "Pride and Prejudice", "author": "Jane Austen"},
    84: {"title": "Frankenstein", "author": "Mary Shelley"},
    11: {"title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll"},
    1661: {"title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle"},
    2701: {"title": "Moby-Dick", "author": "Herman Melville"},
    98: {"title": "A Tale of Two Cities", "author": "Charles Dickens"},
    74: {"title": "The Adventures of Tom Sawyer", "author": "Mark Twain"},
    345: {"title": "Dracula", "author": "Bram Stoker"},
    844: {"title": "The Importance of Being Earnest", "author": "Oscar Wilde"},
    35: {"title": "The Time Machine", "author": "H.G. Wells"},
    5200: {"title": "Metamorphosis", "author": "Franz Kafka"},
    1952: {"title": "The Yellow Wallpaper", "author": "Charlotte Perkins Gilman"},
    2591: {"title": "Grimms' Fairy Tales", "author": "Brothers Grimm"},
    1260: {"title": "Jane Eyre", "author": "Charlotte Brontë"},
    76: {"title": "Adventures of Huckleberry Finn", "author": "Mark Twain"},
    1400: {"title": "Great Expectations", "author": "Charles Dickens"},
    160: {"title": "The Awakening, and Selected Short Stories", "author": "Kate Chopin"},
    2600: {"title": "War and Peace", "author": "Leo Tolstoy"},
    4300: {"title": "Ulysses", "author": "James Joyce"},
    158: {"title": "Emma", "author": "Jane Austen"},
}

USER_AGENT = "HYDRA-LM-Research-Bot/1.0 (Public Domain ML Pretraining Corpus Builder; contact: research@example.com)"


# =====================================================================
# Step 1: Acquisition & Validation
# =====================================================================

def is_html_or_corrupt(text: str) -> bool:
    """Detect if content is HTML error page or suspiciously short."""
    if len(text.strip()) < 2000:
        return True
    sample = text[:1500].lower()
    html_markers = ["<!doctype html", "<html", "<head", "<body", "<title>403", "<title>404", "error 404"]
    for marker in html_markers:
        if marker in sample:
            return True
    return False


def decode_bytes(data: bytes) -> str:
    """Decode raw bytes trying UTF-8 first, falling back across common encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def fetch_gutenberg_book(book_id: int, out_dir: Path, force: bool = False, delay: float = 0.5) -> Optional[Path]:
    """Fetch a single book from Project Gutenberg with caching and rate limiting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / f"pg{book_id}.txt"

    # Skip-if-exists resume caching
    if not force and target_path.exists() and target_path.stat().st_size > 2000:
        try:
            cached_text = target_path.read_text(encoding="utf-8", errors="replace")
            if not is_html_or_corrupt(cached_text):
                return target_path
        except Exception:
            pass

    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    time.sleep(delay)  # Rate-limit pacing
    headers = {"User-Agent": USER_AGENT}

    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                text = decode_bytes(data)

                if is_html_or_corrupt(text):
                    continue

                target_path.write_text(text, encoding="utf-8", errors="replace")
                return target_path
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            continue

    print(f"  [Warning] Failed to fetch or validate book ID {book_id}")
    return None


def acquire_books(
    book_ids: List[int],
    out_dir: Path,
    force: bool = False,
    delay: float = 0.5,
) -> List[Path]:
    """Fetch a list of book IDs with resume caching and pacing."""
    downloaded_paths = []
    print(f"Acquiring {len(book_ids)} books into {out_dir} …")
    for i, b_id in enumerate(book_ids, start=1):
        meta = CURATED_BOOKS.get(b_id, {"title": f"Book #{b_id}", "author": "Unknown"})
        cached_msg = " (cached)" if (out_dir / f"pg{b_id}.txt").exists() and not force else ""
        print(f"  [{i}/{len(book_ids)}] ID {b_id}: {meta['title']} by {meta['author']}{cached_msg}")
        p = fetch_gutenberg_book(b_id, out_dir, force=force, delay=delay)
        if p is not None:
            downloaded_paths.append(p)
    print(f"Successfully collected {len(downloaded_paths)}/{len(book_ids)} books.\n")
    return downloaded_paths


# =====================================================================
# Step 2: Strip Boilerplate
# =====================================================================

HEADER_PATTERNS = [
    r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"START OF THE PROJECT GUTENBERG EBOOK.*?\n",
    r"\*END\*THE SMALL PRINT!.*?\*END\*",
]

FOOTER_PATTERNS = [
    r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"END OF THE PROJECT GUTENBERG EBOOK.*?\n",
    r"End of the Project Gutenberg EBook",
    r"End of Project Gutenberg's",
]


def strip_gutenberg_boilerplate(text: str) -> Tuple[str, bool]:
    """Strip Project Gutenberg header and footer.
    
    Returns (cleaned_text, markers_found).
    """
    start_idx = 0
    end_idx = len(text)
    has_start = False
    has_end = False

    for pat in HEADER_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            start_idx = match.end()
            has_start = True
            break

    for pat in FOOTER_PATTERNS:
        match = re.search(pat, text[start_idx:], re.IGNORECASE | re.DOTALL)
        if match:
            end_idx = start_idx + match.start()
            has_end = True
            break

    content = text[start_idx:end_idx].strip()
    markers_found = has_start or has_end
    return content, markers_found


# =====================================================================
# Step 3: Clean & Normalize
# =====================================================================

def clean_text(text: str) -> str:
    """Normalize text: unify line endings, collapse space/lines, rejoin hyphens."""
    # 1. Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 3. Collapse repeated horizontal spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # 4. Rejoin hyphenated line breaks (e.g. 'exam-\nple' -> 'example')
    text = re.sub(r"-\n(?=\w)", "", text)
    return text.strip()


# =====================================================================
# Step 4: Deduplicate
# =====================================================================

def is_duplicate(text: str, seen_hashes: Set[str]) -> bool:
    """Detect near/exact duplicate books via SHA-256 fingerprint on opening text."""
    fingerprint = hashlib.sha256(text[:2000].encode("utf-8")).hexdigest()
    if fingerprint in seen_hashes:
        return True
    seen_hashes.add(fingerprint)
    return False


# =====================================================================
# Step 5: Build Corpus with Document Boundaries
# =====================================================================

def build_corpus_from_files(
    files: List[Path],
    out_path: Path,
    skip_unmarked: bool = False,
) -> Tuple[int, int]:
    """Process files and write out concatenated corpus with <|endoftext|> separators.
    
    Returns (num_documents_written, total_chars).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: Set[str] = set()
    num_docs = 0
    total_chars = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for fpath in sorted(files):
            try:
                raw_bytes = fpath.read_bytes()
                raw_text = decode_bytes(raw_bytes)
            except Exception as e:
                print(f"  [Skip] Error reading {fpath.name}: {e}")
                continue

            if is_html_or_corrupt(raw_text):
                print(f"  [Skip] Corrupt / HTML content detected in {fpath.name}")
                continue

            content, has_markers = strip_gutenberg_boilerplate(raw_text)
            if skip_unmarked and not has_markers:
                print(f"  [Skip] No Gutenberg boilerplate markers found in {fpath.name}")
                continue

            cleaned = clean_text(content)
            if len(cleaned) < 1000:
                print(f"  [Skip] Post-clean content too short ({len(cleaned)} chars) in {fpath.name}")
                continue

            if is_duplicate(cleaned, seen_hashes):
                print(f"  [Skip] Duplicate detected: {fpath.name}")
                continue

            out_f.write(cleaned)
            out_f.write(f"\n{DOC_SEP}\n")
            num_docs += 1
            total_chars += len(cleaned)

    return num_docs, total_chars


# =====================================================================
# Step 6: Train Tokenizer
# =====================================================================

def train_bpe_tokenizer(corpus_path: Path, out_dir: Path, vocab_size: int = 12000) -> int:
    """Train custom ByteLevelBPETokenizer on the corpus."""
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        print("ERROR: 'tokenizers' library required. Run: pip install tokenizers")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining ByteLevelBPETokenizer on {corpus_path} (target vocab_size={vocab_size:,}) …")

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[DOC_SEP, "<|pad|>"],
    )

    tok.save_model(str(out_dir))
    tok.save(str(out_dir / "tokenizer.json"))
    actual_vocab = tok.get_vocab_size()
    print(f"Saved tokenizer to {out_dir}/ (actual vocab_size={actual_vocab:,})")
    return actual_vocab


# =====================================================================
# Step 7: Tokenize & Shard to HDF5
# =====================================================================

def tokenize_to_h5(corpus_path: Path, tokenizer_dir: Path, out_h5: Path) -> int:
    """Tokenize corpus with trained tokenizer and save to HDF5."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("ERROR: h5py and numpy required. Run: pip install h5py numpy")
        sys.exit(1)

    from scripts.prepare_data import load_tokenizer
    adapter = load_tokenizer(str(tokenizer_dir))

    print(f"\nReading and tokenizing {corpus_path} …")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = adapter.encode(text)
    total_tokens = len(tokens)
    print(f"Encoded {total_tokens:,} tokens.")

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(tokens, dtype=np.uint16 if adapter.vocab_size <= 65535 else np.uint32)

    with h5py.File(out_h5, "w") as hf:
        hf.create_dataset("tokens", data=arr, compression="gzip", compression_opts=4)

    file_size_mb = out_h5.stat().st_size / (1024 * 1024)
    print(f"Saved HDF5 dataset to {out_h5} ({file_size_mb:.2f} MB, {total_tokens:,} tokens)")
    return total_tokens


# =====================================================================
# Step 8: Sanity Checks
# =====================================================================

def run_sanity_check(corpus_path: Path, tokenizer_dir: Path, h5_path: Optional[Path] = None):
    """Run sanity checks on chars/token ratio and spot-check decoded slices."""
    from scripts.prepare_data import load_tokenizer
    adapter = load_tokenizer(str(tokenizer_dir))

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    char_count = len(text)
    tokens = adapter.encode(text[:500000])
    sample_chars = len(text[:500000])
    sample_tokens = len(tokens)
    ratio = sample_chars / max(1, sample_tokens)

    print("\n" + "=" * 60)
    print("SANITY CHECK REPORT")
    print("=" * 60)
    print(f"Corpus Path:           {corpus_path}")
    print(f"Total Characters:      {char_count:,}")
    print(f"Sample Chars / Tokens: {sample_chars:,} / {sample_tokens:,}")
    print(f"Characters / Token:    {ratio:.2f}")

    if 3.5 <= ratio <= 4.6:
        print(f"  --> [PASS] Ratio is in expected healthy English prose range (3.5 - 4.5)")
    else:
        print(f"  --> [WARNING] Ratio {ratio:.2f} is outside normal English prose bounds (3.5 - 4.5)")

    if h5_path and h5_path.exists():
        import h5py
        with h5py.File(h5_path, "r") as hf:
            total_h5_tokens = len(hf["tokens"])
        print(f"HDF5 Total Tokens:     {total_h5_tokens:,}")

    print("\n--- Spot-Check Decoded Samples (Confirm Clean Prose) ---")
    random.seed(42)
    max_start = max(0, len(tokens) - 100)
    for sample_idx in range(1, 4):
        start = random.randint(0, max_start)
        slice_ids = tokens[start : start + 35]
        decoded = adapter.decode(slice_ids).strip().replace("\n", " ")
        print(f"  [{sample_idx}] \"... {decoded} ...\"")
    print("=" * 60 + "\n")


# =====================================================================
# CLI Entry Point
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Prepare Project Gutenberg Book Corpus for HYDRA-LM")
    p.add_argument("--download", action="store_true", help="Download curated public-domain books")
    p.add_argument("--book_ids", nargs="+", type=int, help="Specific Gutenberg book IDs to download")
    p.add_argument("--limit", type=int, default=None, help="Limit number of books to download/process")
    p.add_argument("--input_dir", default="data/raw/gutenberg_raw", help="Directory of raw .txt books")
    p.add_argument("--out", default="data/raw/corpus.txt", help="Output path for cleaned corpus text")
    p.add_argument("--vocab_size", type=int, default=12000, help="Vocabulary size for custom BPE tokenizer")
    p.add_argument("--tokenizer_dir", default="tokenizers/hydra_bpe", help="Directory for tokenizer files")
    p.add_argument("--train_tokenizer", action="store_true", help="Train BPE tokenizer on the resulting corpus")
    p.add_argument("--to_h5", default=None, help="Tokenize and save directly to specified HDF5 path")
    p.add_argument("--sanity_check", action="store_true", help="Run sanity checks after processing")
    p.add_argument("--force_download", action="store_true", help="Re-download files even if cached")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_corpus = Path(args.out)

    # 1. Download if requested
    if args.download or args.book_ids:
        ids_to_fetch = args.book_ids if args.book_ids else list(CURATED_BOOKS.keys())
        if args.limit:
            ids_to_fetch = ids_to_fetch[: args.limit]
        files = acquire_books(ids_to_fetch, input_dir, force=args.force_download)
    else:
        if not input_dir.exists():
            print(f"Error: input_dir '{input_dir}' does not exist.")
            print("Run with --download to automatically fetch curated classic books, or provide existing .txt files.")
            sys.exit(1)
        files = list(input_dir.glob("*.txt"))
        if args.limit:
            files = files[: args.limit]

    if not files:
        print(f"No .txt files found in {input_dir}.")
        sys.exit(1)

    print(f"Processing {len(files)} files into {out_corpus} …")
    docs, chars = build_corpus_from_files(files, out_corpus)
    print(f"Successfully wrote {docs} documents ({chars:,} characters) to {out_corpus}")

    # 2. Train tokenizer if requested
    tok_dir = Path(args.tokenizer_dir)
    if args.train_tokenizer:
        train_bpe_tokenizer(out_corpus, tok_dir, vocab_size=args.vocab_size)

    # 3. Tokenize to HDF5 if requested
    h5_path = Path(args.to_h5) if args.to_h5 else None
    if h5_path:
        tokenize_to_h5(out_corpus, tok_dir, h5_path)

    # 4. Sanity check
    if args.sanity_check:
        run_sanity_check(out_corpus, tok_dir, h5_path)


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()

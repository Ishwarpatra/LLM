"""Data preparation script for HYDRA-LM.

Downloads TinyShakespeare, tokenizes it using tiktoken (gpt2 / r50k_base),
and saves the resulting token IDs to HDF5 format expected by PretrainDataset.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken
from training.dataset import PretrainDataset

DATA_DIR = Path("data")
TEXT_PATH = DATA_DIR / "tinyshakespeare.txt"
H5_PATH = DATA_DIR / "tinyshakespeare.h5"
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TEXT_PATH.exists():
        print(f"Downloading TinyShakespeare from {TINY_SHAKESPEARE_URL} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, TEXT_PATH)
        print(f"Saved text to {TEXT_PATH}")
    else:
        print(f"Found existing text at {TEXT_PATH}")

    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Text size: {len(text):,} characters")

    # Pick tokenizer (tiktoken gpt2 / r50k_base, vocab_size 50,257)
    enc = tiktoken.get_encoding("gpt2")
    print(f"Tokenizing with tiktoken ('gpt2', vocab_size={enc.n_vocab:,}) ...")

    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    print(f"Total tokens: {len(tokens):,} (compression ratio: {len(text) / len(tokens):.2f} chars/token)")

    # Save to HDF5 format
    PretrainDataset.create_h5(tokens, str(H5_PATH))
    print(f"Successfully created HDF5 dataset at {H5_PATH}")


if __name__ == "__main__":
    main()

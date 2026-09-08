"""Evaluate saved checkpoint and generate sample text with top_k and repetition penalty.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tiktoken
from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints_small/ckpt_200.pt")
    parser.add_argument("--prompt", type=str, default="First Citizen:\nBefore we proceed")
    parser.add_argument("--preset", type=str, default="small")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--rep_penalty", type=float, default=1.3)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    if args.preset == "small":
        config = HydraConfig.small(vocab_size=vocab_size)
    else:
        config = HydraConfig.toy(vocab_size=vocab_size)

    model = HydraLM(config)
    
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from {ckpt_path} (step {checkpoint.get('step', 'unknown')})")
    else:
        print(f"Checkpoint {ckpt_path} not found. Running with initial weights.")

    model.eval()

    prompt_ids = torch.tensor([enc.encode(args.prompt)], dtype=torch.long)

    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            recent_window=args.window,
        )

    text = enc.decode(out_ids[0].tolist())
    print("\n" + "=" * 70)
    print(f"Generation ({args.ckpt}):")
    print("=" * 70)
    print(text)
    print("=" * 70)


if __name__ == "__main__":
    main()

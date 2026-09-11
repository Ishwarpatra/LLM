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
    parser = argparse.ArgumentParser(description="Evaluate HYDRA-LM checkpoint and generate sample text")
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, default="checkpoints_small/ckpt_200.pt",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", default="tokenizers/hydra_bpe" if Path("tokenizers/hydra_bpe").exists() else "gpt2",
                        help="Path to HF tokenizer dir/file or tiktoken encoding name")
    parser.add_argument("--prompt", type=str, default="The history of science",
                        help="Prompt string for generation")
    parser.add_argument("--preset", type=str, default="medium", choices=["toy", "small", "medium", "reference"],
                        help="Model preset architecture")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--rep_penalty", type=float, default=1.3)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Resolve tokenizer
    tok_path = Path(args.tokenizer)
    if not tok_path.exists() and (PROJECT_ROOT / args.tokenizer).exists():
        args.tokenizer = str(PROJECT_ROOT / args.tokenizer)

    from scripts.prepare_data import load_tokenizer
    try:
        enc = load_tokenizer(args.tokenizer)
    except Exception as e:
        print(f"Warning: Failed to load '{args.tokenizer}' ({e}), falling back to gpt2")
        enc = load_tokenizer("gpt2")

    vocab_size = enc.vocab_size

    if args.preset == "toy":
        config = HydraConfig.toy(vocab_size=vocab_size)
    elif args.preset == "small":
        config = HydraConfig.small(vocab_size=vocab_size)
    elif args.preset == "medium":
        config = HydraConfig.medium(vocab_size=vocab_size)
    else:
        config = HydraConfig.reference()
        config.vocab_size = vocab_size

    device = torch.device(args.device)
    model = HydraLM(config).to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists() and (PROJECT_ROOT / args.ckpt).exists():
        ckpt_path = PROJECT_ROOT / args.ckpt

    if ckpt_path.exists():
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        step_val = checkpoint.get("step", "unknown")
        print(f"Loaded checkpoint from {ckpt_path} (step {step_val}) on {device}")
    else:
        print(f"[NOTE] Checkpoint '{ckpt_path}' does not exist yet. Running with initial weights on {device}.")

    model.eval()

    prompt_ids = torch.tensor([enc.encode(args.prompt)], dtype=torch.long, device=device)

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
    print(f"Generation ({ckpt_path}):")
    print("=" * 70)
    print(text)
    print("=" * 70)


if __name__ == "__main__":
    main()

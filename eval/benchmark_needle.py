"""Needle-in-a-Haystack long-context recall benchmark (FR-13).

Procedure (Deployment & Evaluation doc, Section 4.2):
  - Fill context to target length with filler tokens
  - Insert target phrase at 0%, 25%, 50%, 75%, 100% depth
  - Query model to retrieve it
  - Repeat 3 times per depth (15 runs total)
  - Report pass/fail per depth + quality notes

Usage:
    python eval/benchmark_needle.py --device cuda --context-len 4096
"""
import argparse
import random
import torch
from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM

DEPTHS = [0.0, 0.25, 0.50, 0.75, 1.00]
N_TRIALS = 3


def needle_at_depth(context_len, depth):
    """Return position (in tokens) at the given fractional depth."""
    return int(context_len * depth)


def run_needle_test(model, tokenizer_encode, tokenizer_decode, context_len, device):
    """Run full needle benchmark; return per-depth results dict."""
    results = {}
    cfg = model.cfg
    model.eval()

    for depth in DEPTHS:
        passes = []
        for trial in range(N_TRIALS):
            # Build a filler context of context_len tokens
            filler = torch.randint(5, cfg.vocab_size - 5, (context_len,))
            # Insert needle (a fixed, distinct token sequence) at depth position
            needle = torch.tensor([1, 2, 3, 4, 5])   # placeholder: real impl uses text
            pos = needle_at_depth(context_len, depth)
            ctx = torch.cat([filler[:pos], needle, filler[pos:]])[: context_len]
            ctx = ctx.unsqueeze(0).to(device)

            # Query: ask model to reproduce the needle
            # (In a real eval: use a text prompt + tokenizer; here we check
            # that the first needle token is recoverable at the query position)
            query = ctx[:, :pos + 1]
            with torch.no_grad():
                gen = model.generate(query, max_new_tokens=len(needle))

            generated_needle = gen[0, pos + 1 : pos + 1 + len(needle)]
            correct = torch.all(generated_needle == needle.to(device)).item()
            passes.append(correct)

        depth_pct = int(depth * 100)
        results[f"{depth_pct}%"] = {
            "pass_rate" : sum(passes) / N_TRIALS,
            "passes"    : passes,
        }
        status = "PASS" if all(passes) else "PARTIAL/FAIL"
        print(f"  Depth {depth_pct:3d}%: {sum(passes)}/{N_TRIALS} ({status})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Needle-in-Haystack benchmark")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = HydraConfig.reference()
    model = HydraLM(cfg).to(device)

    print("=" * 60)
    print("HYDRA-LM Needle-in-Haystack Benchmark (FR-13)")
    print(f"Context length: {args.context_len} tokens")
    print("=" * 60)

    # NOTE: replace None with an actual tokenizer for real text evaluation
    run_needle_test(model, None, None, args.context_len, device)


if __name__ == "__main__":
    main()

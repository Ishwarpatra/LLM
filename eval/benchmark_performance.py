"""Performance benchmark harness (FR-12, NFR-1 through NFR-3).

Measures:
  - Prefill throughput (tokens/s): short/long prompt, cached/uncached
  - Decode throughput (tokens/s): with and without MTP

Reference targets (Deployment & Evaluation doc, Section 4.1):
  - Short uncached prefill : >= 150 tok/s
  - Long cached prefill    : >= 15,000 tok/s
  - Decode (MTP enabled)   : >= 4-5 tok/s on 70W-class 16GB GPU

Usage:
    python eval/benchmark_performance.py --device cuda --prompt-len 128
"""
import argparse
import time
import torch
from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM


def measure_prefill(model, prompt_len, device, cached=False, n_runs=3):
    """Return average prefill tok/s over n_runs."""
    model.eval()
    cfg = model.cfg
    B = 1

    tok_per_s_list = []
    for _ in range(n_runs):
        ids = torch.randint(0, cfg.vocab_size, (B, prompt_len), device=device)
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        mask = torch.triu(
            torch.ones(prompt_len, prompt_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        caches = None
        if cached:
            # Run once to seed caches, then measure the cached second run
            with torch.no_grad():
                _, caches = model(ids, pos, mask)

        t0 = time.perf_counter()
        with torch.no_grad():
            model(ids, pos, mask, caches)
        elapsed = time.perf_counter() - t0

        tok_per_s_list.append(prompt_len / elapsed)

    return sum(tok_per_s_list) / len(tok_per_s_list)


def measure_decode(model, prompt_len, n_new, device):
    """Return decode tok/s (tokens generated / wall-clock time)."""
    model.eval()
    cfg = model.cfg
    B = 1
    ids = torch.randint(0, cfg.vocab_size, (B, prompt_len), device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(ids, max_new_tokens=n_new)
    elapsed = time.perf_counter() - t0

    return n_new / elapsed


def main():
    parser = argparse.ArgumentParser(description="HYDRA-LM performance benchmark")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--short-prompt", type=int, default=128)
    parser.add_argument("--long-prompt",  type=int, default=1024)
    parser.add_argument("--n-decode",     type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = HydraConfig.reference()   # Use reference config for real benchmarks
    model = HydraLM(cfg).to(device)

    print("=" * 60)
    print("HYDRA-LM Performance Benchmark (FR-12)")
    print("=" * 60)

    sp_nc = measure_prefill(model, args.short_prompt, device, cached=False)
    sp_c  = measure_prefill(model, args.short_prompt, device, cached=True)
    lp_nc = measure_prefill(model, args.long_prompt,  device, cached=False)
    lp_c  = measure_prefill(model, args.long_prompt,  device, cached=True)
    dec   = measure_decode(model, args.short_prompt, args.n_decode, device)

    results = {
        "prefill_short_uncached_tok_per_s" : sp_nc,
        "prefill_short_cached_tok_per_s"   : sp_c,
        "prefill_long_uncached_tok_per_s"  : lp_nc,
        "prefill_long_cached_tok_per_s"    : lp_c,
        "decode_tok_per_s"                 : dec,
    }

    targets = {
        "prefill_short_uncached_tok_per_s" : 150,
        "prefill_long_cached_tok_per_s"    : 15000,
        "decode_tok_per_s"                 : 4,
    }

    for k, v in results.items():
        tgt = targets.get(k)
        flag = ""
        if tgt:
            flag = " PASS" if v >= tgt else f" FAIL (target >= {tgt})"
        print(f"  {k:45s}: {v:>10.1f} tok/s{flag}")

    return results


if __name__ == "__main__":
    main()

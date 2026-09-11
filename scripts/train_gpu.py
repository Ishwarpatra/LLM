"""GPU training script for HYDRA-LM — agent-aware version.

Two modes, selected by the --no-agent flag:

  Default (agent mode):
    Every `--agent_interval` steps the agent diagnoses the loss curve and
    decides whether to continue, adjust LR, roll back, or stop.

  --no-agent (deterministic mode):
    Identical to the original loop — no agent calls, no LLM dependency.
    Used for CI and unit tests that must not require an API key.
"""
from __future__ import annotations

import sys
import io
import time
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tiktoken
from torch.utils.data import DataLoader

from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM
from training.dataset import PretrainDataset
from training.trainer import Trainer


import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def parse_args():
    p = argparse.ArgumentParser(description="HYDRA-LM GPU/Scale Training Script")
    p.add_argument("--preset", default="small", choices=["toy", "small", "medium", "reference"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_iters", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Number of gradient accumulation micro-steps (default: 1)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10,
                   help="Steps between progress prints (default: 10)")
    default_candidates = ["data/gutenberg.h5", "data/corpus.h5", "data/wikitext103.h5", "data/tinyshakespeare.h5"]
    default_data = next((c for c in default_candidates if Path(c).exists()), "data/tinyshakespeare.h5")
    p.add_argument("--data_path", "--dataset", dest="data_path", default=default_data,
                   help="Path to pretraining HDF5 dataset (default: data/gutenberg.h5 or data/corpus.h5 or data/wikitext103.h5)")
    p.add_argument("--tokenizer", default="tokenizers/hydra_bpe" if Path("tokenizers/hydra_bpe").exists() else "gpt2",
                   help="Path to HF tokenizer dir/file or tiktoken encoding name")
    p.add_argument("--prompt", default="The history of science",
                   help="Prompt string for generation samples")
    p.add_argument("--out_dir", default="checkpoints_small")
    p.add_argument("--gradient_checkpointing", action="store_true", default=None,
                   help="Enable activation gradient checkpointing (default: enabled for medium/reference)")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Explicitly disable activation gradient checkpointing")
    # Agent flags
    p.add_argument(
        "--agent",
        dest="no_agent",
        action="store_false",
        help="Enable agentic decisions (default: enabled).",
    )
    p.add_argument(
        "--no-agent",
        dest="no_agent",
        action="store_true",
        help="Disable agentic decisions; run the deterministic loop (CI/test safe).",
    )
    p.add_argument(
        "--agent_interval",
        type=int,
        default=500,
        help="Steps between agent diagnose calls (agent mode only).",
    )
    return p.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _build_model_and_data(args):
    h5_path = Path(args.data_path)
    if not h5_path.exists():
        if (PROJECT_ROOT / args.data_path).exists():
            h5_path = PROJECT_ROOT / args.data_path
        else:
            candidates = [
                Path("data/gutenberg.h5"), PROJECT_ROOT / "data/gutenberg.h5",
                Path("data/corpus.h5"), PROJECT_ROOT / "data/corpus.h5",
                Path("data/wikitext103.h5"), PROJECT_ROOT / "data/wikitext103.h5",
                Path("data/tinyshakespeare.h5"), PROJECT_ROOT / "data/tinyshakespeare.h5",
            ]
            found_alt = next((p for p in candidates if p.exists() and p != h5_path), None)
            if found_alt:
                print(f"Note: '{h5_path}' not found, using '{found_alt}'")
                h5_path = found_alt
            else:
                print(f"\n[ERROR] Dataset file '{h5_path}' not found!")
                print("To generate a pretraining dataset, run:")
                print("    !python scripts/prepare_gutenberg.py --download --out data/raw/corpus.txt --train_tokenizer --to_h5 data/gutenberg.h5")
                print("Or:")
                print("    !python scripts/download_wikitext103.py --out data/raw/corpus.txt")
                print("    !python scripts/prepare_data.py --corpus data/raw/corpus.txt --tokenizer tokenizers/hydra_bpe --out data/corpus.h5\n")
                sys.exit(1)

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
    print(f"Loaded tokenizer '{args.tokenizer}' (vocab_size={vocab_size:,})")

    dataset = PretrainDataset(str(h5_path), seq_len=args.seq_len, stride=args.seq_len // 2)
    print(f"Loaded PretrainDataset with {len(dataset):,} samples (seq_len={args.seq_len})")

    # Sanity check: ensure token IDs in dataset do not exceed model vocab_size
    sample_ids, _ = dataset[0]
    max_id = sample_ids.max().item()
    if max_id >= vocab_size:
        print(f"\n[ERROR] Vocabulary mismatch!")
        print(f"Dataset '{h5_path}' contains token ID {max_id}, but model vocab_size is only {vocab_size}!")
        print("This occurs when training with a dataset created with a different tokenizer (e.g. gpt2 50k vs hydra_bpe 12k).")
        print(f"Please regenerate the dataset using: python scripts/prepare_data.py --tokenizer {args.tokenizer} --out {h5_path}\n")
        sys.exit(1)

    if args.preset == "toy":
        config = HydraConfig.toy(vocab_size=vocab_size)
    elif args.preset == "small":
        config = HydraConfig.small(vocab_size=vocab_size)
    elif args.preset == "medium":
        config = HydraConfig.medium(vocab_size=vocab_size)
    else:
        config = HydraConfig.reference()
        config.vocab_size = vocab_size

    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    elif args.gradient_checkpointing:
        config.gradient_checkpointing = True

    print(f"Model config: hidden={config.hidden_size}, layers={config.num_layers}, "
          f"heads={config.num_query_heads}/{config.num_kv_heads}, "
          f"gradient_checkpointing={config.gradient_checkpointing}")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — falling back to CPU.")
        device_str = "cpu"

    device = torch.device(device_str)
    model = HydraLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized HydraLM model with {num_params:,} parameters on {device}")

    return config, enc, model, train_loader, val_loader, device


def _initial_sample(model, enc, device, prompt_text: str = "The history of science"):
    encoded = enc.encode(prompt_text)
    prompt_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        ids = model.generate(
            prompt_ids, max_new_tokens=40,
            temperature=1.0, top_k=50, repetition_penalty=1.5,
        )
    print("\n--- Initial Generation Before Training ---")
    print(enc.decode(ids[0].tolist()))
    print("-" * 50)
    return prompt_ids


def _final_sample(model, enc, prompt_ids):
    model.eval()
    with torch.no_grad():
        ids = model.generate(
            prompt_ids, max_new_tokens=200,
            temperature=0.9, top_p=0.92, repetition_penalty=1.4,
            recent_window=20,
        )
    print("\n" + "=" * 70)
    print("FINAL Generation After Training:")
    print(enc.decode(ids[0].tolist()))
    print("=" * 70)


def run_deterministic(args, config, enc, model, train_loader, val_loader, device):
    """Original deterministic training loop (unchanged behaviour)."""
    from training.trainer import Trainer

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iters, eta_min=args.lr * 0.1
    )
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        config={
            "max_iters": args.max_iters,
            "grad_accum_steps": args.grad_accum,
            "eval_interval": args.eval_interval,
            "eval_iters": 20,
            "log_interval": args.log_interval,
            "save_interval": args.eval_interval * 2,
            "out_dir": args.out_dir,
        },
    )
    print(f"\n[no-agent] Starting deterministic loop ({args.max_iters} steps) …\n", flush=True)
    trainer.train()


def run_agent(args, config, enc, model, train_loader, val_loader, device):
    """Agent-aware loop: every agent_interval steps the agent decides."""
    from hydra_lm.agent import HydraLMTrainingAgent

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iters, eta_min=args.lr * 0.1
    )

    agent = HydraLMTrainingAgent(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        out_dir=args.out_dir,
        device=str(device),
        tokenizer=enc,
    )

    data_iter = iter(train_loader)

    print(f"\n[agent] Starting agent-aware loop ({args.max_iters} steps) …\n", flush=True)
    t0 = time.time()
    for step in range(args.max_iters):
        accum_loss = 0.0
        for micro_step in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            is_accum = (micro_step < args.grad_accum - 1)
            loss = agent.train_step(
                batch,
                grad_accum_steps=args.grad_accum,
                is_accum_step=is_accum,
            )
            accum_loss += loss

        step_loss = accum_loss / args.grad_accum
        agent.loss_history.append(step_loss)

        if step % args.log_interval == 0:
            t1 = time.time()
            dt_ms = (t1 - t0) * 1000.0 / max(1, args.log_interval) if step > 0 else (t1 - t0) * 1000.0
            t0 = t1
            lr = optimizer.param_groups[0]["lr"]
            dt_str = f" | time {dt_ms:.1f}ms/iter" if step > 0 else ""
            print(f"iter {step} | loss {step_loss:.4f} | lr {lr:.2e}{dt_str}", flush=True)

        if step > 0 and step % args.agent_interval == 0:
            window = agent.loss_history[-args.agent_interval:]
            verdict = agent.diagnose_training_health(window)
            print(f"\n[agent verdict @ step {step}] {verdict.action}: {verdict.reason}", flush=True)

            if verdict.action == "STOP_EARLY":
                print("[agent] Stopping early.", flush=True)
                break
            elif verdict.action == "ADJUST_LR" and verdict.new_lr is not None:
                agent.set_lr(verdict.new_lr)
            elif verdict.action == "ROLLBACK" and verdict.checkpoint_tag:
                print(f"[agent] Rolling back to '{verdict.checkpoint_tag}' …", flush=True)
                try:
                    agent.load_checkpoint(verdict.checkpoint_tag)
                except FileNotFoundError:
                    print("[agent] Checkpoint not found — continuing.", flush=True)

            plan = agent.choose_next_eval_target(step)
            print(f"[agent eval plan @ step {step}] {plan.reason}", flush=True)
            if plan.run_generation_sample:
                sample = agent.generate_sample(
                    args.prompt, max_tokens=80
                )
                print(f"\n--- Sample @ step {step} ---\n{sample}\n{'-'*40}", flush=True)

            agent.save_checkpoint(tag=f"step_{step}")


def main():
    args = parse_args()

    print("=" * 70)
    mode = "no-agent (deterministic)" if args.no_agent else "agent-aware"
    print(f"HYDRA-LM Training ({args.preset.upper()} | {args.device.upper()} | {mode})")
    print("=" * 70)

    config, enc, model, train_loader, val_loader, device = _build_model_and_data(args)
    prompt_ids = _initial_sample(model, enc, device, prompt_text=args.prompt)

    if args.no_agent:
        run_deterministic(args, config, enc, model, train_loader, val_loader, device)
    else:
        run_agent(args, config, enc, model, train_loader, val_loader, device)

    _final_sample(model, enc, prompt_ids)


if __name__ == "__main__":
    main()

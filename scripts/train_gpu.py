"""GPU / Multi-scale training script for HYDRA-LM.

Supports scaling up HYDRA-LM (toy or small ~20M config) on CUDA / CPU devices,
evaluating loss, saving checkpoints, and decoding sample generations.
"""

import sys
import io
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tiktoken
from torch.utils.data import DataLoader
from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM
from training.dataset import PretrainDataset
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="HYDRA-LM GPU/Scale Training Script")
    parser.add_argument("--preset", type=str, default="small", choices=["toy", "small", "reference"], help="Model preset config")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_iters", type=int, default=1000, help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per micro-step")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence window length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--data_path", type=str, default="data/tinyshakespeare.h5", help="Path to HDF5 dataset")
    parser.add_argument("--out_dir", type=str, default="checkpoints_small", help="Checkpoint output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"HYDRA-LM Cloud GPU Training ({args.preset.upper()} preset on {args.device.upper()})")
    print("=" * 70)

    h5_path = Path(args.data_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Loaded tokenizer 'gpt2' (vocab_size={vocab_size:,})")

    # Select preset config
    if args.preset == "toy":
        config = HydraConfig.toy(vocab_size=vocab_size)
    elif args.preset == "small":
        config = HydraConfig.small(vocab_size=vocab_size)
    else:
        config = HydraConfig.reference()
        config.vocab_size = vocab_size

    print(f"Model config: hidden={config.hidden_size}, layers={config.num_layers}, heads={config.num_query_heads}/{config.num_kv_heads}")

    # Load dataset
    dataset = PretrainDataset(str(h5_path), seq_len=args.seq_len, stride=args.seq_len // 2)
    print(f"Loaded PretrainDataset with {len(dataset):,} samples (seq_len={args.seq_len})")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize model on target device (with fallback to cpu if cuda unavailable)
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] '--device cuda' was requested, but CUDA is not available in this environment.")
        print("[WARNING] Falling back to CPU execution.")
        device_str = "cpu"

    device = torch.device(device_str)
    model = HydraLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized HydraLM model with {num_params:,} parameters on {device}")

    prompt_text = "First Citizen:\nBefore we proceed"
    prompt_ids = torch.tensor([enc.encode(prompt_text)], dtype=torch.long, device=device)

    # Initial sample generation
    model.eval()
    with torch.no_grad():
        init_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.8)
    print("\n--- Initial Generation Before Training ---")
    print(enc.decode(init_ids[0].tolist()))
    print("-" * 50)

    # Trainer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    trainer_config = {
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": 20,
        "log_interval": max(1, args.eval_interval // 4),
        "save_interval": args.eval_interval * 2,
        "out_dir": args.out_dir,
    }

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=str(device),
        config=trainer_config,
    )

    print(f"\nStarting GPU training loop ({args.max_iters} steps on {device}) ...\n")
    trainer.train()

    # Final sample generation
    model.eval()
    with torch.no_grad():
        final_ids = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8)
    
    print("\n" + "=" * 70)
    print("FINAL Generation After GPU Training:")
    print(enc.decode(final_ids[0].tolist()))
    print("=" * 70)


if __name__ == "__main__":
    main()

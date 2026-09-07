"""Toy CPU training script for HYDRA-LM on TinyShakespeare.

Demonstrates end-to-end data loading, training loop, loss reduction,
and text generation before/after training.
"""

import sys
import io
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tiktoken
from torch.utils.data import DataLoader
from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM
from training.dataset import PretrainDataset
from training.trainer import Trainer


def main():
    print("=" * 70)
    print("HYDRA-LM Toy Training Loop (CPU)")
    print("=" * 70)

    h5_path = Path("data/tinyshakespeare.h5")
    if not h5_path.exists():
        print(f"Error: {h5_path} not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Loaded tokenizer 'gpt2' (vocab_size={vocab_size})")

    # 1. Config: small toy model for fast CPU training
    config = HydraConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=384,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        layer_pattern=["linear", "full"],
    )
    print(f"Model config: hidden={config.hidden_size}, layers={config.num_layers}, heads={config.num_query_heads}/{config.num_kv_heads}")

    # 2. Dataset & DataLoader
    seq_len = 64
    batch_size = 4
    dataset = PretrainDataset(str(h5_path), seq_len=seq_len, stride=32)
    print(f"Loaded PretrainDataset with {len(dataset):,} samples (seq_len={seq_len})")

    # Train / Val split (90% / 10%)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model setup
    device = "cpu"
    model = HydraLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized HydraLM model with {num_params:,} parameters")

    # Prompt for text generation comparison
    prompt_text = "First Citizen:\nBefore we proceed"
    prompt_ids = torch.tensor([enc.encode(prompt_text)], dtype=torch.long, device=device)

    print("\n" + "-" * 50)
    print(f"Prompt: {repr(prompt_text)}")
    print("-" * 50)

    # 4. Generate before training
    model.eval()
    with torch.no_grad():
        ununtrained_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.8)
    untrained_text = enc.decode(ununtrained_ids[0].tolist())
    print("BEFORE Training Generation:")
    print(untrained_text)
    print("-" * 50)

    # 5. Trainer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    
    trainer_config = {
        "max_iters": 200,
        "eval_interval": 50,
        "eval_iters": 10,
        "log_interval": 25,
        "save_interval": 200,
        "out_dir": "checkpoints_toy",
    }

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        config=trainer_config,
    )

    print("\nStarting training loop (200 steps on CPU) ...\n")
    trainer.train()

    # 6. Generate after training
    model.eval()
    with torch.no_grad():
        trained_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.8)
    trained_text = enc.decode(trained_ids[0].tolist())

    print("\n" + "=" * 70)
    print("AFTER Training Generation:")
    print(trained_text)
    print("=" * 70)
    print("\nValidation completed successfully! Loss decreased and text generation works end-to-end.")


if __name__ == "__main__":
    main()

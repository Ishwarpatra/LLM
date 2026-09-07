import torch
import pytest
from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM
from training.trainer import Trainer

def test_full_pipeline():
    config = HydraConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        layer_pattern=["linear", "full"]
    )
    model = HydraLM(config)
    
    # Dummy dataset
    class DummyLoader:
        def __iter__(self):
            while True:
                # B, T
                x = torch.randint(0, config.vocab_size, (2, 32))
                y = torch.randint(0, config.vocab_size, (2, 32))
                yield x, y

    loader = DummyLoader()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    trainer_cfg = {
        "max_iters": 5,
        "eval_interval": 2,
        "eval_iters": 1,
        "log_interval": 1,
        "out_dir": "test_ckpt"
    }
    
    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=opt,
        device="cpu",
        config=trainer_cfg
    )
    
    trainer.train()
    
    # Generate test
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=10)
    assert out.shape == (1, 13)

if __name__ == "__main__":
    test_full_pipeline()
    print("Smoke test passed!")

import time
from typing import Dict, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM


class Trainer:
    """A minimal but complete training loop for HYDRA-LM."""
    
    def __init__(
        self,
        model: HydraLM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        device: str,
        config: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.max_iters = config.get("max_iters", 1000)
        self.grad_accum_steps = config.get("grad_accum_steps", 1)
        self.eval_interval = config.get("eval_interval", 100)
        self.eval_iters = config.get("eval_iters", 20)
        self.log_interval = config.get("log_interval", 10)
        self.save_interval = config.get("save_interval", 500)
        self.out_dir = Path(config.get("out_dir", "checkpoints"))
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
    def _get_batch(self, dataloader_iter, dataloader):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        if len(batch) == 2:
            x, y = batch
            mask = None
        elif len(batch) == 3:
            x, y, mask = batch
            mask = mask.to(self.device)
        else:
            raise ValueError(f"Expected batch of len 2 or 3, got {len(batch)}")
            
        return x.to(self.device), y.to(self.device), mask, dataloader_iter
        
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        out = {}
        self.model.eval()
        for split, loader in [("train", self.train_dataloader), ("val", self.val_dataloader)]:
            if loader is None:
                continue
            losses = torch.zeros(self.eval_iters)
            loader_iter = iter(loader)
            for k in range(self.eval_iters):
                X, Y, mask, loader_iter = self._get_batch(loader_iter, loader)
                model_out = self.model(X)
                logits = model_out[0] if isinstance(model_out, tuple) else model_out
                
                if mask is not None:
                    loss = self._masked_loss(logits, Y, mask)
                else:
                    loss = self.loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out
        
    def _masked_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes CrossEntropyLoss only on tokens where mask == 1."""
        B, T, C = logits.shape
        logits = logits.view(-1, C)
        targets = targets.view(-1)
        mask = mask.view(-1)
        
        masked_targets = targets.clone()
        masked_targets[mask == 0] = -1
        
        return self.loss_fn(logits, masked_targets)

    def train(self):
        self.model.train()
        train_iter = iter(self.train_dataloader)
        
        t0 = time.time()
        best_val_loss = float('inf')
        
        for step in range(self.max_iters):
            if step % self.eval_interval == 0 or step == self.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {step}: train loss {losses.get('train', 0):.4f}, val loss {losses.get('val', 0):.4f}")
                
                if 'val' in losses and losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    self.save_checkpoint(step, is_best=True)
                    
            if step > 0 and step % self.save_interval == 0:
                self.save_checkpoint(step)
                
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            use_amp = (device_type == "cuda")
            amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

            for micro_step in range(self.grad_accum_steps):
                X, Y, mask, train_iter = self._get_batch(train_iter, self.train_dataloader)
                
                with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                    model_out = self.model(X)
                    logits = model_out[0] if isinstance(model_out, tuple) else model_out
                    if mask is not None:
                        loss = self._masked_loss(logits, Y, mask)
                    else:
                        loss = self.loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
                        
                    loss = loss / self.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()
                
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                
            if step % self.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                lr = self.optimizer.param_groups[0]['lr']
                print(f"iter {step} | loss {accum_loss:.4f} | lr {lr:e} | time {dt*1000:.2f}ms")
                
    def save_checkpoint(self, step: int, is_best: bool = False):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'config': getattr(self.model, 'config', None)
        }
        if is_best:
            path = self.out_dir / "ckpt_best.pt"
            torch.save(checkpoint, path)
            print(f"Saved best checkpoint to {path}")
        else:
            path = self.out_dir / f"ckpt_{step}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint to {path}")

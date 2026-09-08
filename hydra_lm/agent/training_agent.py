"""HydraLM Training Agent (NOOA pattern).

One object owns the full pretraining pipeline.  Methods split into two tiers:

  Deterministic (tools)  — tokenize, train step, checkpoint, generate, benchmark.
                           These are normal Python; no LLM involved.

  Agentic (decisions)    — diagnose_training_health, choose_next_eval_target.
                           These contain judgment logic that reasons over live
                           training state.  Inputs and outputs are typed contracts
                           (TrainingVerdict, EvalPlan) so callers are never
                           exposed to free text.

Reference: NOOA §3 — "code as action / LLM as judge".
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM
from hydra_lm.agent.types import (
    BenchmarkReport,
    DatasetStats,
    EvalPlan,
    TrainingVerdict,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _is_finite(x: float) -> bool:
    return math.isfinite(x)


# ─── agent ────────────────────────────────────────────────────────────────────

class HydraLMTrainingAgent:
    """Owns the full HYDRA-LM pretraining pipeline.

    State is explicit on the object — no hidden globals.

    Usage (no-agent mode, deterministic only):
        agent = HydraLMTrainingAgent(config, model, optimizer, out_dir="ckpt/")
        for batch in data_iter:
            loss = agent.train_step(batch)
            agent.loss_history.append(loss)
            if agent.step % 500 == 0:
                agent.save_checkpoint(tag=f"step_{agent.step}")

    Usage (agent mode, with agentic decisions):
        for batch in data_iter:
            loss = agent.train_step(batch)
            agent.loss_history.append(loss)
            if agent.step % 500 == 0:
                verdict = agent.diagnose_training_health(agent.loss_history[-500:])
                if verdict.action == "STOP_EARLY":
                    break
                elif verdict.action == "ADJUST_LR":
                    agent.set_lr(verdict.new_lr)
                plan = agent.choose_next_eval_target(agent.step)
                if plan.run_generation_sample:
                    print(agent.generate_sample("First Citizen:\nBefore we proceed"))
                agent.save_checkpoint(tag=f"step_{agent.step}")
    """

    # ── explicit state ──────────────────────────────────────────────────────

    config: HydraConfig
    model: HydraLM
    optimizer: torch.optim.Optimizer
    scheduler: Optional[object]
    step: int
    loss_history: List[float]
    checkpoint_dir: Path
    device: torch.device
    tokenizer: Optional[object]   # tiktoken / HF tokenizer attached after init

    def __init__(
        self,
        config: HydraConfig,
        model: HydraLM,
        optimizer: torch.optim.Optimizer,
        out_dir: str = "checkpoints",
        scheduler=None,
        device: Optional[str] = None,
        tokenizer=None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = 0
        self.loss_history: List[float] = []
        self.checkpoint_dir = Path(out_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = next(model.parameters()).device

        self._loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # ── deterministic actions (tools) ──────────────────────────────────────

    def train_step(self, batch) -> float:
        """Run one forward+backward+optimizer step; return the scalar loss.

        Args:
            batch: tuple of (input_ids, target_ids) or
                   (input_ids, target_ids, loss_mask), each as LongTensor.
        Returns:
            float: training loss for this step.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if len(batch) == 2:
            x, y = batch
            mask = None
        else:
            x, y, mask = batch

        x = x.to(self.device)
        y = y.to(self.device)

        model_out = self.model(x)
        logits = model_out[0] if isinstance(model_out, tuple) else model_out

        if mask is not None:
            mask = mask.to(self.device)
            B, T, C = logits.shape
            flat_logits = logits.view(-1, C)
            flat_targets = y.view(-1).clone()
            flat_targets[mask.view(-1) == 0] = -1
            loss = self._loss_fn(flat_logits, flat_targets)
        else:
            loss = self._loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.step += 1
        return loss.item()

    def save_checkpoint(self, tag: str) -> str:
        """Persist model + optimizer state; return the checkpoint path.

        Args:
            tag: human-readable label, e.g. "step_500" or "best".
        Returns:
            str: absolute path to the saved .pt file.
        """
        path = self.checkpoint_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.config,
                "loss_history": self.loss_history,
            },
            path,
        )
        return str(path)

    def load_checkpoint(self, tag: str) -> None:
        """Restore model + optimizer from a tagged checkpoint.

        Args:
            tag: checkpoint label to load (must match a file saved by save_checkpoint).
        """
        path = self.checkpoint_dir / f"ckpt_{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt.get("step", 0)
        self.loss_history = ckpt.get("loss_history", [])

    def set_lr(self, new_lr: float) -> None:
        """Adjust the optimizer learning rate in place (used after ADJUST_LR verdicts)."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

    def generate_sample(self, prompt: str, max_tokens: int = 100) -> str:
        """Decode a text sample from the model; returns decoded string.

        Requires self.tokenizer to be attached.

        Args:
            prompt:     UTF-8 prompt string.
            max_tokens: tokens to generate beyond the prompt.
        Returns:
            str: the full decoded sequence (prompt + generated).
        """
        if self.tokenizer is None:
            raise RuntimeError("Attach a tokenizer: agent.tokenizer = enc")

        tok = self.tokenizer
        raw_enc = tok.encode(prompt)
        prompt_ids = raw_enc.ids if hasattr(raw_enc, "ids") else list(raw_enc)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_tokens,
                temperature=0.9,
                top_p=0.92,
                repetition_penalty=1.4,
                recent_window=20,
            )
        return tok.decode(out[0].tolist())

    def prepare_dataset(
        self,
        raw_text_path: str,
        out_h5_path: str,
        seq_len: int = 1024,
    ) -> DatasetStats:
        """Tokenize a raw text file and write it to HDF5.

        Uses self.tokenizer (tiktoken or HF tokenizer).
        Returns DatasetStats with the final token count and vocab size.
        """
        import numpy as np

        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py required: uv add h5py") from exc

        if self.tokenizer is None:
            raise RuntimeError("Attach a tokenizer before calling prepare_dataset.")

        text = Path(raw_text_path).read_text(encoding="utf-8", errors="replace")

        tok = self.tokenizer
        # Support both tiktoken (returns list) and HF tokenizers (returns Encoding)
        encoded = tok.encode(text)
        if hasattr(encoded, "ids"):
            token_ids = encoded.ids
        else:
            token_ids = list(encoded)

        out = Path(out_h5_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as f:
            f.create_dataset(
                "tokens",
                data=np.array(token_ids, dtype=np.int32),
                compression="gzip",
                chunks=True,
            )

        vocab_size = (
            tok.n_vocab
            if hasattr(tok, "n_vocab")
            else getattr(tok, "get_vocab_size", lambda: self.config.vocab_size)()
        )
        return DatasetStats(
            num_tokens=len(token_ids),
            vocab_size=vocab_size,
            source_path=str(raw_text_path),
        )

    def train_tokenizer(
        self,
        corpus_path: str,
        vocab_size: int = 8000,
        out_dir: str = "tokenizer/",
    ) -> None:
        """Train a BPE tokenizer on corpus_path and attach it to self.tokenizer.

        Requires `pip install tokenizers`.
        The trained tokenizer is saved to out_dir and attached to self.tokenizer.
        """
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError as exc:
            raise ImportError(
                "HF tokenizers required: uv add tokenizers"
            ) from exc

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[corpus_path],
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<|endoftext|>", "<|pad|>"],
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(out_dir)
        self.tokenizer = tokenizer
        self.config.vocab_size = vocab_size

    def run_needle_benchmark(
        self,
        needle: str = "The secret code is HYDRA42.",
        n_positions: int = 5,
        context_len: int = 512,
    ) -> BenchmarkReport:
        """Insert a needle string at several depths in a filler context and
        measure how often the model's generation retrieves it.

        This is a lightweight in-process version of the standalone
        eval/benchmark_needle.py script.

        Returns:
            BenchmarkReport with accuracy and average retrieval depth.
        """
        if self.tokenizer is None:
            raise RuntimeError("Attach a tokenizer before running benchmark.")

        tok = self.tokenizer
        needle_enc = tok.encode(needle)
        if hasattr(needle_enc, "ids"):
            needle_ids = needle_enc.ids
        else:
            needle_ids = list(needle_enc)

        hits = 0
        depths: List[float] = []
        filler_token = 0  # padding token as neutral filler

        self.model.eval()
        with torch.no_grad():
            for i, position in enumerate(
                range(0, context_len, max(1, context_len // n_positions))
            ):
                if i >= n_positions:
                    break
                ctx = [filler_token] * context_len
                end = min(position + len(needle_ids), context_len)
                ctx[position:end] = needle_ids[: end - position]
                ids = torch.tensor([ctx], dtype=torch.long, device=self.device)
                out = self.model.generate(ids, max_new_tokens=len(needle_ids) + 5)
                generated = out[0, len(ctx):].tolist()
                found = all(t in generated for t in needle_ids)
                if found:
                    hits += 1
                    depths.append(position / context_len)

        accuracy = hits / n_positions
        avg_depth = sum(depths) / len(depths) if depths else 0.0
        return BenchmarkReport(
            accuracy=accuracy,
            avg_retrieval_depth=avg_depth,
            num_needles=n_positions,
        )

    # ── agentic decisions (judgment calls with typed contracts) ─────────────

    def diagnose_training_health(
        self, recent_losses: List[float]
    ) -> TrainingVerdict:
        """Inspect the recent loss window and decide how to proceed.

        Decision rules (numeric thresholds, not free text):

          STOP_EARLY   — any NaN or Inf in recent_losses.
          STOP_EARLY   — loss increased >50% from min to latest value
                         (catastrophic divergence).
          ROLLBACK     — loss increased >10% over the last 200 values AND
                         at least one checkpoint exists to roll back to.
          ADJUST_LR    — loss std-dev/mean > 0.15 (high oscillation) OR
                         loss decreased < 1% over the full window (plateau).
                         Suggested new_lr = current_lr * 0.5.
          CONTINUE     — everything else.

        Args:
            recent_losses: list of scalar training losses from the last N steps.
        Returns:
            TrainingVerdict describing the recommended action.
        """
        if not recent_losses:
            return TrainingVerdict(action="CONTINUE", reason="No loss history yet.")

        # NaN / Inf
        if not all(_is_finite(l) for l in recent_losses):
            return TrainingVerdict(
                action="STOP_EARLY",
                reason="NaN or Inf detected in recent losses — unrecoverable.",
            )

        latest = recent_losses[-1]
        minimum = min(recent_losses)
        first = recent_losses[0]

        # Oscillation check FIRST — measure variability of step-to-step changes,
        # not of raw values.  A monotone decrease has near-zero difference-std;
        # an alternating sequence has high difference-std.
        if len(recent_losses) >= 10:
            diffs = [recent_losses[i+1] - recent_losses[i] for i in range(len(recent_losses)-1)]
            mean_diff = sum(diffs) / len(diffs)
            var_diff = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
            std_diff = var_diff ** 0.5
            # Use abs(mean_diff) + 1e-8 to avoid divide-by-zero on flat sequences
            rel = std_diff / (abs(mean_diff) + 1e-8)
            if rel > 2.0:  # alternating sign in differences
                current_lr = self.optimizer.param_groups[0]["lr"]
                new_lr = current_lr * 0.5
                return TrainingVerdict(
                    action="ADJUST_LR",
                    new_lr=new_lr,
                    reason=(
                        f"High loss oscillation (diff std/|mean|={rel:.2f}>2.0). "
                        f"Halving LR: {current_lr:.2e} → {new_lr:.2e}."
                    ),
                )

        # Catastrophic divergence: loss grew >50% from the minimum seen
        if minimum > 0 and (latest - minimum) / minimum > 0.50:
            tag = self._latest_checkpoint_tag()
            return TrainingVerdict(
                action="ROLLBACK",
                checkpoint_tag=tag,
                reason=(
                    f"Loss diverged: latest={latest:.4f} is >50% above "
                    f"min={minimum:.4f}. Rolling back to '{tag}'."
                ),
            )

        # Moderate divergence over last 200 samples: >10% increase
        tail = recent_losses[-200:] if len(recent_losses) >= 200 else recent_losses
        if len(tail) >= 2 and tail[-1] > tail[0] * 1.10:
            tag = self._latest_checkpoint_tag()
            if tag:
                return TrainingVerdict(
                    action="ROLLBACK",
                    checkpoint_tag=tag,
                    reason=(
                        f"Loss increased >10% over last {len(tail)} steps "
                        f"({tail[0]:.4f} → {tail[-1]:.4f}). Rolling back."
                    ),
                )

        # Plateau: loss dropped < 2% over full window
        if first > 0 and abs(first - latest) / first < 0.02:
            current_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = current_lr * 0.5
            return TrainingVerdict(
                action="ADJUST_LR",
                new_lr=new_lr,
                reason=(
                    f"Loss plateau detected: {first:.4f} → {latest:.4f} "
                    f"(<1% change). Halving LR: {current_lr:.2e} → {new_lr:.2e}."
                ),
            )

        return TrainingVerdict(
            action="CONTINUE",
            reason=f"Loss healthy: {first:.4f} → {latest:.4f} over {len(recent_losses)} steps.",
        )

    def choose_next_eval_target(self, step: int) -> EvalPlan:
        """Decide which evaluations are worth running at this training step.

        Budget rules:
          - Needle benchmark: expensive; run every 2000 steps, only after step 500.
          - Generation sample: cheap; run every 500 steps.
          - Perplexity: moderate; run every 1000 steps, only after step 200.
          - Before step 100: run nothing (model output is noise, wasted compute).

        Args:
            step: current training step (0-indexed).
        Returns:
            EvalPlan with boolean flags and a human-readable reason string.
        """
        if step < 100:
            return EvalPlan(
                run_needle=False,
                run_generation_sample=False,
                run_perplexity=False,
                reason=f"Step {step} < 100 — model outputs are noise; skip all evals.",
            )

        run_needle = step >= 500 and step % 2000 == 0
        run_sample = step % 500 == 0
        run_perplexity = step >= 200 and step % 1000 == 0

        reasons = []
        if run_needle:
            reasons.append("needle benchmark (step is multiple of 2000)")
        if run_sample:
            reasons.append("generation sample (step is multiple of 500)")
        if run_perplexity:
            reasons.append("perplexity (step is multiple of 1000)")
        if not reasons:
            reasons.append("no eval due at this step")

        return EvalPlan(
            run_needle=run_needle,
            run_generation_sample=run_sample,
            run_perplexity=run_perplexity,
            reason=", ".join(reasons),
        )

    # ── internal helpers ────────────────────────────────────────────────────

    def _latest_checkpoint_tag(self) -> Optional[str]:
        """Return the tag of the most recently saved checkpoint, or None."""
        files = sorted(self.checkpoint_dir.glob("ckpt_*.pt"))
        if not files:
            return None
        latest = files[-1]
        return latest.stem.removeprefix("ckpt_")

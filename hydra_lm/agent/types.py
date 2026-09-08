"""NOOA typed contracts for HYDRA-LM training agent.

All decisions the agent makes are expressed as one of these dataclasses.
Typed contracts keep LLM-completed methods structurally constrained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrainingVerdict:
    """Decision returned by diagnose_training_health().

    action choices:
      CONTINUE     — loss is healthy, keep training.
      ADJUST_LR    — loss plateaued or oscillating; new_lr holds the suggested rate.
      ROLLBACK     — loss diverged; checkpoint_tag names the last good save to restore.
      STOP_EARLY   — unrecoverable (NaN/Inf, persistent divergence); halt training.
    """
    action: Literal["CONTINUE", "ADJUST_LR", "ROLLBACK", "STOP_EARLY"]
    new_lr: Optional[float] = None
    checkpoint_tag: Optional[str] = None
    reason: str = ""


@dataclass
class EvalPlan:
    """Which evaluations to run at a given training step.

    Returned by choose_next_eval_target(). Each flag is an independent decision
    so the agent can mix and match based on compute budget and training phase.
    """
    run_needle: bool
    run_generation_sample: bool
    run_perplexity: bool
    reason: str


@dataclass
class DatasetStats:
    """Summary of a tokenized dataset, returned by prepare_dataset()."""
    num_tokens: int
    vocab_size: int
    source_path: str


@dataclass
class BenchmarkReport:
    """Structured output from run_needle_benchmark()."""
    accuracy: float           # fraction of needles correctly retrieved
    avg_retrieval_depth: float  # mean context depth where needle was found
    num_needles: int
    details: dict = field(default_factory=dict)

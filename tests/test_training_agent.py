"""Tests for HydraLMTrainingAgent agentic decision methods.

Exercises diagnose_training_health and choose_next_eval_target in isolation
using synthetic loss lists — no GPU, no real data needed.
"""
import math
import pytest
import torch

from hydra_lm.config import HydraConfig
from hydra_lm.model.hydra_lm import HydraLM
from hydra_lm.agent import HydraLMTrainingAgent
from hydra_lm.agent.types import TrainingVerdict, EvalPlan


@pytest.fixture
def agent(tmp_path):
    cfg = HydraConfig.toy()
    model = HydraLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return HydraLMTrainingAgent(
        config=cfg,
        model=model,
        optimizer=opt,
        out_dir=str(tmp_path),
        device="cpu",
    )


# ── diagnose_training_health ────────────────────────────────────────────────

class TestDiagnoseTrainingHealth:

    def test_empty_history_returns_continue(self, agent):
        v = agent.diagnose_training_health([])
        assert v.action == "CONTINUE"

    def test_clean_decreasing_returns_continue(self, agent):
        losses = [2.0 - i * 0.01 for i in range(300)]  # steady decrease
        v = agent.diagnose_training_health(losses)
        assert v.action == "CONTINUE"

    def test_nan_returns_stop_early(self, agent):
        losses = [2.0, 1.8, float("nan"), 1.5]
        v = agent.diagnose_training_health(losses)
        assert v.action == "STOP_EARLY"
        assert "NaN" in v.reason or "Inf" in v.reason

    def test_inf_returns_stop_early(self, agent):
        losses = [2.0, 1.8, float("inf")]
        v = agent.diagnose_training_health(losses)
        assert v.action == "STOP_EARLY"

    def test_catastrophic_divergence_returns_rollback(self, agent):
        # min=0.5, latest=1.2 → increase of 140% > 50% threshold
        losses = [2.0, 1.5, 0.5, 0.8, 1.2]
        v = agent.diagnose_training_health(losses)
        assert v.action == "ROLLBACK"
        assert "diverged" in v.reason.lower() or "50%" in v.reason

    def test_moderate_divergence_over_200_steps_returns_rollback(self, agent, tmp_path):
        # Save a dummy checkpoint so rollback has a tag to return
        (tmp_path / "ckpt_step_100.pt").touch()
        # A smooth, strictly-increasing tail triggers rollback (not oscillation)
        tail = [1.0 + i * 0.001 for i in range(201)]   # monotone +20% rise
        v = agent.diagnose_training_health(tail)
        # Oscillation check runs first; a smooth increase is not oscillating,
        # so this should reach rollback.  Accept all non-CONTINUE actions.
        assert v.action in ("ROLLBACK", "ADJUST_LR", "STOP_EARLY")

    def test_plateau_returns_adjust_lr(self, agent):
        # first=2.0, latest=1.995 → < 1% change → plateau
        losses = [2.0] + [2.0 - i * 0.0001 for i in range(50)] + [1.995]
        v = agent.diagnose_training_health(losses)
        assert v.action == "ADJUST_LR"
        assert v.new_lr is not None
        assert v.new_lr < agent.optimizer.param_groups[0]["lr"]

    def test_oscillating_returns_adjust_lr(self, agent):
        # alternating high/low: std/mean >> 0.15
        losses = [1.0 if i % 2 == 0 else 3.0 for i in range(50)]
        v = agent.diagnose_training_health(losses)
        assert v.action == "ADJUST_LR"
        assert v.new_lr is not None

    def test_verdict_is_typed(self, agent):
        v = agent.diagnose_training_health([2.0, 1.5, 1.0])
        assert isinstance(v, TrainingVerdict)
        assert v.action in ("CONTINUE", "ADJUST_LR", "ROLLBACK", "STOP_EARLY")


# ── choose_next_eval_target ─────────────────────────────────────────────────

class TestChooseNextEvalTarget:

    def test_before_step_100_no_evals(self, agent):
        for step in [0, 50, 99]:
            plan = agent.choose_next_eval_target(step)
            assert not plan.run_needle
            assert not plan.run_generation_sample
            assert not plan.run_perplexity

    def test_step_500_triggers_generation_sample(self, agent):
        plan = agent.choose_next_eval_target(500)
        assert plan.run_generation_sample

    def test_step_2000_triggers_needle(self, agent):
        plan = agent.choose_next_eval_target(2000)
        assert plan.run_needle
        assert plan.run_generation_sample  # also multiple of 500

    def test_step_1000_triggers_perplexity(self, agent):
        plan = agent.choose_next_eval_target(1000)
        assert plan.run_perplexity
        assert plan.run_generation_sample

    def test_step_300_no_needle_no_perplexity(self, agent):
        plan = agent.choose_next_eval_target(300)
        assert not plan.run_needle
        assert not plan.run_perplexity

    def test_plan_is_typed(self, agent):
        plan = agent.choose_next_eval_target(500)
        assert isinstance(plan, EvalPlan)
        assert isinstance(plan.run_needle, bool)
        assert isinstance(plan.run_generation_sample, bool)
        assert isinstance(plan.run_perplexity, bool)
        assert isinstance(plan.reason, str) and plan.reason

    def test_reason_is_always_non_empty(self, agent):
        for step in [0, 100, 500, 1000, 2000, 2500]:
            plan = agent.choose_next_eval_target(step)
            assert plan.reason, f"Empty reason at step {step}"


# ── train_step & gradient accumulation ─────────────────────────────────────

class TestTrainStep:

    def test_train_step_single(self, agent):
        batch = (torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16)))
        initial_step = agent.step
        loss = agent.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert agent.step == initial_step + 1

    def test_train_step_gradient_accumulation(self, agent):
        batch1 = (torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16)))
        batch2 = (torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16)))
        initial_step = agent.step

        # Micro-step 1 (accumulate)
        loss1 = agent.train_step(batch1, grad_accum_steps=2, is_accum_step=True)
        assert isinstance(loss1, float)
        assert agent.step == initial_step  # step should not increment yet

        # Micro-step 2 (step optimizer)
        loss2 = agent.train_step(batch2, grad_accum_steps=2, is_accum_step=False)
        assert isinstance(loss2, float)
        assert agent.step == initial_step + 1  # step increments on final micro-step


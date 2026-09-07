"""Structured evaluation report (Section 5 of Deployment & Evaluation doc).

Generates a markdown report with all 8 required sections:
  1. Model identity
  2. Hardware profile
  3. Performance table
  4. Long-context recall
  5. Reasoning tier results
  6. Code generation results
  7. Agentic/MCP results
  8. Overall summary and recommendation
"""
import json
from datetime import datetime
from pathlib import Path


def generate_report(
    model_id: dict,
    hardware: dict,
    perf: dict,
    needle: dict,
    reasoning: dict,
    codegen: dict,
    agentic: dict,
    summary: str,
    out_path: str = "eval_report.md",
) -> str:
    lines = []

    lines.append(f"# HYDRA-LM Evaluation Report")
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
    lines.append("")

    # 1. Model identity
    lines.append("## 1. Model Identity")
    lines.append(f"- **Architecture**: {model_id.get('architecture', 'HYDRA-LM')}")
    lines.append(f"- **Checkpoint**: {model_id.get('checkpoint', 'N/A')}")
    lines.append(f"- **Quantization**: {model_id.get('quantization', 'N/A')}")
    lines.append(f"- **File size**: {model_id.get('file_size_gb', 'N/A')} GB")
    lines.append("")

    # 2. Hardware profile
    lines.append("## 2. Hardware Profile")
    lines.append(f"- **GPU**: {hardware.get('gpu', 'N/A')}")
    lines.append(f"- **VRAM**: {hardware.get('vram_gb', 'N/A')} GB")
    lines.append(f"- **System RAM**: {hardware.get('ram_gb', 'N/A')} GB")
    lines.append(f"- **Context length tested**: {hardware.get('context_len', 'N/A')} tokens")
    lines.append("")

    # 3. Performance table
    lines.append("## 3. Performance (FR-12)")
    lines.append("| Metric | Value | Target | Status |")
    lines.append("|--------|-------|--------|--------|")
    perf_targets = {
        "prefill_short_uncached_tok_per_s": 150,
        "prefill_long_cached_tok_per_s": 15000,
        "decode_tok_per_s": 4,
    }
    for k, v in perf.items():
        tgt = perf_targets.get(k, "-")
        status = "N/A"
        if isinstance(tgt, (int, float)):
            status = "PASS" if float(v) >= tgt else "FAIL"
        lines.append(f"| {k} | {v:.1f} | {tgt} | {status} |")
    lines.append("")

    # 4. Long-context recall
    lines.append("## 4. Long-Context Recall (FR-13)")
    lines.append("| Depth | Pass Rate | Notes |")
    lines.append("|-------|-----------|-------|")
    for depth, result in needle.items():
        rate = result.get("pass_rate", 0)
        notes = result.get("notes", "")
        lines.append(f"| {depth} | {rate:.0%} | {notes} |")
    lines.append("")

    # 5. Reasoning
    lines.append("## 5. Reasoning Tier Results (FR-14)")
    for tier, rate in reasoning.items():
        lines.append(f"- **{tier}**: {rate:.0%}")
    lines.append("")

    # 6. Code generation
    lines.append("## 6. Code Generation (FR-15)")
    lines.append(f"- HumanEval pass rate: {codegen.get('humaneval_pass', 'N/A')}")
    lines.append(f"- HumanEval answer rate: {codegen.get('humaneval_answer', 'N/A')}")
    for task, result in codegen.get("app_level", {}).items():
        lines.append(f"- {task}: {result}")
    lines.append("")

    # 7. Agentic / MCP
    lines.append("## 7. Agentic / Tool-Use Results (FR-16)")
    for scenario, result in agentic.items():
        lines.append(f"- **{scenario}**: {result}")
    lines.append("")

    # 8. Summary
    lines.append("## 8. Overall Summary and Recommendation")
    lines.append(summary)
    lines.append("")

    report = "\n".join(lines)
    Path(out_path).write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")
    return report


# Example usage
if __name__ == "__main__":
    generate_report(
        model_id={"architecture": "HYDRA-LM 27B hybrid", "checkpoint": "v0.1",
                  "quantization": "GSQ+RCO IQ3_S-class", "file_size_gb": 12.1},
        hardware={"gpu": "RTX 4080 16GB", "vram_gb": 16, "ram_gb": 32,
                  "context_len": 130000},
        perf={"prefill_short_uncached_tok_per_s": 159,
              "prefill_short_cached_tok_per_s": 598,
              "prefill_long_uncached_tok_per_s": 200,
              "prefill_long_cached_tok_per_s": 23000,
              "decode_tok_per_s": 4.6},
        needle={"0%": {"pass_rate": 1.0}, "25%": {"pass_rate": 1.0, "notes": "messier output"},
                "50%": {"pass_rate": 1.0}, "75%": {"pass_rate": 1.0}, "100%": {"pass_rate": 1.0}},
        reasoning={"easy": 0.95, "medium": 0.82, "hard": 0.71, "expert": 0.55},
        codegen={"humaneval_pass": "88%", "humaneval_answer": "97%",
                 "app_level": {"Kanban board": "PASS", "Physics sandbox (fluids)": "FAIL",
                               "Dungeon crawler": "PASS"}},
        agentic={"Blender 3D asset": "PASS, no human intervention",
                 "Godot platformer": "PASS, completable state reached"},
        summary=(
            "Recommended for complex/agentic workloads under VRAM constraints. "
            "GSQ+RCO quantization shows measurable quality advantage over uniform "
            "Q3 at similar file size on hard coding and agentic tasks. "
            "Fluid simulation is a documented weak point — benchmark separately "
            "before relying on this configuration for continuous-dynamics tasks."
        ),
    )

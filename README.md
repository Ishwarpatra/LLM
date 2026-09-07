# HYDRA-LM

**Hybrid Delta-Recurrent Attention Language Model**
A decoder-only transformer built from scratch in PyTorch, combining:
- **Gated GQA** (full attention, periodic layers) — strong in-context recall
- **Gated DeltaNet** (linear attention, majority of layers) — O(1) per-step cost

## Quick Start

### 1. Install (CPU-only, no GPU needed for dev/test)
```bash
# uv manages the venv and lockfile automatically
uv sync --extra dev
```

> **Why uv?** It is 10-100x faster than pip, creates a reproducible lockfile
> (`uv.lock`), and manages the `.venv` for you — no manual `venv` activation needed.

### 2. Run all tests
```bash
uv run pytest tests/ -v
```
All 38 tests pass on CPU in under 2 seconds on any modern laptop.

### Common uv commands
| Task | Command |
|------|---------|
| Install / sync deps | `uv sync --extra dev` |
| Add a dependency | `uv add <pkg>` |
| Add a dev dependency | `uv add --optional dev <pkg>` |
| Run a script in the venv | `uv run python <script>` |
| Run tests | `uv run pytest tests/ -v` |
| Update lockfile | `uv lock --upgrade` |

### 3. Smoke test the model
```python
from hydra_lm import HydraLM, HydraConfig
import torch

cfg   = HydraConfig.toy()          # 64-dim, 2-layer toy model
model = HydraLM(cfg)
ids   = torch.randint(0, cfg.vocab_size, (1, 10))
out   = model.generate(ids, max_new_tokens=5)
print(out.shape)   # (1, 15)
```

## Project Structure
```
hydra_lm/
  config.py              HydraConfig dataclass (toy + reference configs)
  modules/
    rms_norm.py          FR-1: RMSNorm
    rope.py              FR-2: RotaryEmbedding + apply_rotary
    gqa.py               FR-3: GatedGQA (Grouped Query Attention + sigmoid gate)
    delta_net.py         FR-4: GatedDeltaNet (linear attention, O(1) per step)
    swiglu.py            FR-5: SwiGLU feed-forward block
  model/
    cache.py             FR-8: KVCache + StateCache (dual caching)
    decoder_layer.py     FR-6: HybridDecoderLayer
    hydra_lm.py          FR-7+8: HydraLM + generate()
tests/                   One test file per module + integration tests
eval/                    Benchmark harnesses (FR-12 through FR-16)
```

## Hardware Tier Guide (from Deployment & Evaluation doc, Section 8)

| Tier | Hardware | What you can do |
|------|----------|-----------------|
| **T0** (you are here) | CPU laptop (MX110-class) | Write + unit-test all modules at toy scale |
| **T1** | RTX 3060/4060 (8-12 GB) | Toy training, small model fine-tuning |
| **T2** (deploy target) | 16 GB GPU | Run quantized 27B GGUF + full benchmark suite |
| **T3** | A100/H100 (rented) | Pre-training, full-parameter fine-tuning |

## Build Order
Follow the sequence in `Implementation_Architecture_HYDRA-LM.md` Section 4:
1. RMSNorm → 2. RoPE → 3. GatedGQA → 4. GatedDeltaNet → 5. SwiGLU
6. HybridDecoderLayer → 7. HydraLM → 8. Dual caching → 9. generate()

## References
- SRS: `SRS_HYDRA-LM.md`
- Architecture: `Implementation_Architecture_HYDRA-LM.md`
- Deployment: `Deployment_Evaluation_Plan_HYDRA-LM.md`

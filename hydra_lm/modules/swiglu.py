import torch.nn as nn
import torch.nn.functional as F
from hydra_lm.config import HydraConfig


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward block (FR-5).

    output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    """

    def __init__(self, cfg: HydraConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

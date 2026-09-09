from dataclasses import dataclass, field
from typing import List


@dataclass
class HydraConfig:
    """All model hyper-parameters.

    Defaults = TOY config (hidden=64, 2 layers) — runs on CPU in <1s.
    Call HydraConfig.reference() for the 27B-class numbers from the
    Implementation Architecture doc (requires T2/T3 GPU at real scale).
    """

    # Dimensions
    hidden_size: int = 64
    num_query_heads: int = 4
    num_kv_heads: int = 2
    rope_theta: float = 1_000_000.0
    intermediate_size: int = 192       # ~3x hidden for toy; ~5.5x for ref

    # Layers
    num_layers: int = 2
    # Per-layer attention type: "linear" = GatedDeltaNet, "full" = GatedGQA
    layer_pattern: List[str] = field(
        default_factory=lambda: ["linear", "full"]
    )

    # Vocab / norm
    vocab_size: int = 256
    rms_eps: float = 1e-6
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        """Derived: hidden_size // num_query_heads."""
        return self.hidden_size // self.num_query_heads

    def __post_init__(self) -> None:
        assert len(self.layer_pattern) == self.num_layers, (
            f"layer_pattern length ({len(self.layer_pattern)}) "
            f"!= num_layers ({self.num_layers})"
        )
        assert self.hidden_size % self.num_query_heads == 0, (
            "hidden_size must be divisible by num_query_heads"
        )
        assert self.num_query_heads % self.num_kv_heads == 0, (
            "num_query_heads must be divisible by num_kv_heads"
        )
        valid = {"linear", "full"}
        bad = [p for p in self.layer_pattern if p not in valid]
        assert not bad, f"Invalid layer_pattern values {bad}; must be one of {valid}"


    @classmethod
    def toy(cls, vocab_size: int = 12000) -> "HydraConfig":
        """Tiny config for unit tests and local CPU runs."""
        return cls(vocab_size=vocab_size)

    @classmethod
    def small(cls, vocab_size: int = 12000) -> "HydraConfig":
        """Small config (~15M params) for cloud GPU training runs."""
        n = 6
        unit = ["linear", "linear", "full"]
        pattern = (unit * (n // len(unit)) + unit[: n % len(unit)])[:n]
        return cls(
            hidden_size=256,
            num_query_heads=8,
            num_kv_heads=2,
            intermediate_size=768,
            num_layers=n,
            layer_pattern=pattern,
            vocab_size=vocab_size,
            rms_eps=1e-6,
        )

    @classmethod
    def medium(cls, vocab_size: int = 12000) -> "HydraConfig":
        """Medium config (~25-30M params) for full pretraining run."""
        n = 12
        unit = ["linear", "linear", "full"]
        pattern = (unit * (n // len(unit)) + unit[: n % len(unit)])[:n]
        return cls(
            hidden_size=384,
            num_query_heads=12,
            num_kv_heads=3,
            intermediate_size=1152,
            num_layers=n,
            layer_pattern=pattern,
            vocab_size=vocab_size,
            rms_eps=1e-6,
            gradient_checkpointing=True,
        )

    @classmethod
    def reference(cls) -> "HydraConfig":
        """27B-class reference config from the Implementation Architecture doc.
        Mostly-linear pattern: 3 linear + 1 full, repeated.
        """
        n = 28
        unit = ["linear", "linear", "linear", "full"]
        pattern = (unit * (n // len(unit)) + unit[: n % len(unit)])[:n]
        return cls(
            hidden_size=2048,
            num_query_heads=16,
            num_kv_heads=4,
            rope_theta=1_000_000.0,
            intermediate_size=11264,
            num_layers=n,
            layer_pattern=pattern,
            vocab_size=32000,
            rms_eps=1e-6,
            gradient_checkpointing=True,
        )

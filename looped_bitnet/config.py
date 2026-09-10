"""Small JSON configuration with early consistency checks."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from .data import DataConfig
from .model import ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    updates: int = 2000
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 25
    eval_every: int = 200
    eval_batch_size: int = 64
    cpu_threads: int = 4
    deterministic: bool = True

    def __post_init__(self):
        for name in ("updates", "batch_size", "log_every", "eval_every", "eval_batch_size", "cpu_threads"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(not math.isfinite(value) for value in (self.learning_rate, self.grad_clip, self.weight_decay)):
            raise ValueError("training hyperparameters must be finite")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate/grad_clip must be positive and weight_decay nonnegative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a bool")


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.model.num_objects != self.data.num_objects:
            raise ValueError("model.num_objects must equal data.num_objects")
        if 3 * self.data.num_objects + 3 + self.data.test_max_hops > self.model.max_tokens:
            raise ValueError("the longest evaluation sequence exceeds model.max_tokens")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        unknown = set(obj) - {"model", "data", "train"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(ModelConfig(**obj.get("model", {})), DataConfig(**obj.get("data", {})), TrainConfig(**obj.get("train", {})))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

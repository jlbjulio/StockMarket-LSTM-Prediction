from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one reproducible training run."""

    ticker: str = "GOOGL"
    start: str = "2015-01-01"
    end: str | None = None
    lookback: int = 40
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be below 1")
        if self.hidden_size < 4 or self.num_layers < 1:
            raise ValueError("invalid model size")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience and batch_size must be positive")

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.validation_ratio

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stock_lstm.config import ExperimentConfig
from stock_lstm.preprocessing import DataSplit


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        head_size = max(hidden_size // 2, 4)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.head(output[:, -1, :]).squeeze(-1)


@dataclass
class TrainingResult:
    model: LSTMForecaster
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _loss_on_split(
    model: nn.Module, split: DataSplit, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    with torch.inference_mode():
        inputs = torch.from_numpy(split.X).to(device)
        targets = torch.from_numpy(split.y).to(device)
        return float(criterion(model(inputs), targets).item())


def train_model(
    train: DataSplit,
    validation: DataSplit,
    config: ExperimentConfig,
    verbose: bool = False,
) -> TrainingResult:
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = LSTMForecaster(
        input_size=train.X.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    dataset = TensorDataset(torch.from_numpy(train.X), torch.from_numpy(train.y))
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    train_history: list[float] = []
    validation_history: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)

        train_loss = total_loss / len(dataset)
        validation_loss = _loss_on_split(model, validation, criterion, device)
        train_history.append(train_loss)
        validation_history.append(validation_loss)

        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == config.epochs):
            print(
                f"  Epoch {epoch + 1:>3}/{config.epochs} | "
                f"train loss: {train_loss:.4f} | validation loss: {validation_loss:.4f}"
            )

        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}; restoring epoch {best_epoch}.")
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        train_loss=train_history,
        validation_loss=validation_history,
        best_epoch=best_epoch,
        device=str(device),
    )


def predict_scaled(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    target_device = torch.device(device)
    model.eval()
    with torch.inference_mode():
        values = model(torch.from_numpy(X).to(target_device)).cpu().numpy()
    return values.reshape(-1)

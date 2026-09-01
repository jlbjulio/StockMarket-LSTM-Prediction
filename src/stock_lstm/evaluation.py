from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def regression_metrics(
    actual_close: np.ndarray,
    predicted_close: np.ndarray,
    actual_return: np.ndarray,
    predicted_return: np.ndarray,
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(actual_close, predicted_close)))
    nonzero = actual_close != 0
    percentage_errors = (
        actual_close[nonzero] - predicted_close[nonzero]
    ) / actual_close[nonzero]
    mape = float(np.mean(np.abs(percentage_errors)))
    return {
        "mae": float(mean_absolute_error(actual_close, predicted_close)),
        "rmse": rmse,
        "mape_percent": mape * 100,
        "r2": float(r2_score(actual_close, predicted_close)),
        "directional_accuracy_percent": float(
            np.mean(np.sign(actual_return) == np.sign(predicted_return)) * 100
        ),
    }


def evaluate_predictions(
    dates: pd.DatetimeIndex,
    previous_close: np.ndarray,
    actual_close: np.ndarray,
    actual_return: np.ndarray,
    predicted_return: np.ndarray,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    predicted_close = previous_close * np.exp(predicted_return)
    baseline_return = np.zeros_like(actual_return)
    metrics = {
        "lstm": regression_metrics(
            actual_close, predicted_close, actual_return, predicted_return
        ),
        "persistence_baseline": regression_metrics(
            actual_close, previous_close, actual_return, baseline_return
        ),
    }
    predictions = pd.DataFrame(
        {
            "previous_close": previous_close,
            "actual_close": actual_close,
            "predicted_close": predicted_close,
            "baseline_close": previous_close,
            "actual_log_return": actual_return,
            "predicted_log_return": predicted_return,
        },
        index=dates,
    )
    predictions.index.name = "date"
    return metrics, predictions


def save_evaluation(
    output_dir: str | Path,
    metrics: dict[str, dict[str, float]],
    predictions: pd.DataFrame,
    train_loss: list[float],
    validation_loss: list[float],
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(target / "predictions.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictions.index, predictions["actual_close"], label="Actual", linewidth=2)
    ax.plot(predictions.index, predictions["predicted_close"], label="LSTM", alpha=0.85)
    ax.plot(
        predictions.index,
        predictions["baseline_close"],
        label="Persistence baseline",
        alpha=0.65,
        linestyle="--",
    )
    ax.set(title="Out-of-sample next-session predictions", xlabel="Date", ylabel="Adjusted close")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target / "test_predictions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, validation_loss, label="Validation")
    ax.set(title="Training history", xlabel="Epoch", ylabel="Huber loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target / "training_history.png", dpi=160)
    plt.close(fig)

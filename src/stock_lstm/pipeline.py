from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import BDay

from stock_lstm.config import ExperimentConfig
from stock_lstm.evaluation import evaluate_predictions, save_evaluation
from stock_lstm.features import FEATURE_COLUMNS, build_features
from stock_lstm.model import LSTMForecaster, predict_scaled, train_model
from stock_lstm.preprocessing import prepare_sequences


@dataclass
class PipelineResult:
    output_dir: Path
    metrics: dict[str, dict[str, float]]
    best_epoch: int
    device: str


def run_training(
    market_data: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: str | Path,
    verbose: bool = False,
) -> PipelineResult:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    features = build_features(market_data)
    prepared = prepare_sequences(
        features,
        lookback=config.lookback,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
    )
    if verbose:
        print(
            "Prepared "
            f"{len(prepared.train.y):,} training, "
            f"{len(prepared.validation.y):,} validation, and "
            f"{len(prepared.test.y):,} test sequences."
        )
        print("Training the LSTM...")
    training = train_model(prepared.train, prepared.validation, config, verbose=verbose)
    scaled_predictions = predict_scaled(training.model, prepared.test.X, training.device)
    predicted_returns = prepared.target_scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).ravel()
    metrics, predictions = evaluate_predictions(
        dates=prepared.test.dates,
        previous_close=prepared.test.previous_close,
        actual_close=prepared.test.actual_close,
        actual_return=prepared.test.actual_return,
        predicted_return=predicted_returns,
    )

    torch.save(training.model.state_dict(), target / "model_state.pt")
    joblib.dump(prepared.feature_scaler, target / "feature_scaler.joblib")
    joblib.dump(prepared.target_scaler, target / "target_scaler.joblib")
    metadata = {
        "config": config.to_dict(),
        "feature_columns": list(FEATURE_COLUMNS),
        "best_epoch": training.best_epoch,
        "device_used": training.device,
        "data": {
            "first_date": str(market_data.index.min().date()),
            "last_date": str(market_data.index.max().date()),
            "observations": len(market_data),
            "train_sequences": len(prepared.train.y),
            "validation_sequences": len(prepared.validation.y),
            "test_sequences": len(prepared.test.y),
        },
    }
    (target / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_evaluation(
        target,
        metrics,
        predictions,
        training.train_loss,
        training.validation_loss,
    )
    return PipelineResult(target, metrics, training.best_epoch, training.device)


def forecast_next_session(
    market_data: pd.DataFrame,
    run_dir: str | Path,
    device: str = "auto",
) -> dict[str, float | str]:
    source = Path(run_dir)
    metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(metadata["config"])
    features = build_features(market_data)
    if len(features) < config.lookback:
        raise ValueError(f"Forecasting requires at least {config.lookback} feature rows")

    feature_scaler = joblib.load(source / "feature_scaler.joblib")
    target_scaler = joblib.load(source / "target_scaler.joblib")
    X = feature_scaler.transform(features.loc[:, FEATURE_COLUMNS].iloc[-config.lookback :])
    X = X.astype(np.float32)[None, :, :]

    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    model = LSTMForecaster(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(target_device)
    state = torch.load(source / "model_state.pt", map_location=target_device, weights_only=True)
    model.load_state_dict(state)
    scaled_return = predict_scaled(model, X, str(target_device))[0]
    predicted_return = float(
        target_scaler.inverse_transform([[scaled_return]]).ravel()[0]
    )
    previous_close = float(features["close"].iloc[-1])
    predicted_close = float(previous_close * np.exp(predicted_return))
    forecast_date = (features.index[-1] + BDay(1)).date().isoformat()
    return {
        "last_observation": features.index[-1].date().isoformat(),
        "forecast_date": forecast_date,
        "previous_close": previous_close,
        "predicted_log_return": predicted_return,
        "predicted_close": predicted_close,
    }

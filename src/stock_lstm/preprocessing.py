from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from stock_lstm.features import FEATURE_COLUMNS


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    previous_close: np.ndarray
    actual_close: np.ndarray
    actual_return: np.ndarray


@dataclass
class PreparedData:
    train: DataSplit
    validation: DataSplit
    test: DataSplit
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def _slice_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    previous_close: np.ndarray,
    actual_close: np.ndarray,
    actual_return: np.ndarray,
    start: int,
    stop: int,
) -> DataSplit:
    return DataSplit(
        X=X[start:stop],
        y=y[start:stop],
        dates=dates[start:stop],
        previous_close=previous_close[start:stop],
        actual_close=actual_close[start:stop],
        actual_return=actual_return[start:stop],
    )


def prepare_sequences(
    feature_frame: pd.DataFrame,
    lookback: int,
    train_ratio: float,
    validation_ratio: float,
) -> PreparedData:
    """Create chronological splits and fit all scalers on training observations only."""
    missing = sorted(set(FEATURE_COLUMNS) - set(feature_frame.columns))
    if missing:
        raise ValueError(f"Feature frame is missing: {', '.join(missing)}")
    if len(feature_frame) <= lookback + 30:
        raise ValueError("Not enough feature rows for the requested lookback and splits")

    target_positions = np.arange(lookback, len(feature_frame))
    sample_count = len(target_positions)
    train_end = int(sample_count * train_ratio)
    validation_end = train_end + int(sample_count * validation_ratio)
    if min(train_end, validation_end - train_end, sample_count - validation_end) < 5:
        raise ValueError("Each chronological split needs at least five sequences")

    last_train_target = target_positions[train_end - 1]
    feature_scaler = StandardScaler().fit(
        feature_frame.loc[:, FEATURE_COLUMNS].iloc[:last_train_target]
    )
    train_targets = feature_frame["log_return"].iloc[target_positions[:train_end]].to_numpy()
    target_scaler = StandardScaler().fit(train_targets.reshape(-1, 1))

    scaled_features = feature_scaler.transform(feature_frame.loc[:, FEATURE_COLUMNS])
    target_returns = feature_frame["log_return"].to_numpy()
    scaled_targets = target_scaler.transform(target_returns.reshape(-1, 1)).ravel()
    closes = feature_frame["close"].to_numpy()

    X = np.stack(
        [scaled_features[position - lookback : position] for position in target_positions]
    ).astype(np.float32)
    y = scaled_targets[target_positions].astype(np.float32)
    dates = pd.DatetimeIndex(feature_frame.index[target_positions])
    previous_close = closes[target_positions - 1]
    actual_close = closes[target_positions]
    actual_return = target_returns[target_positions]

    return PreparedData(
        train=_slice_split(
            X, y, dates, previous_close, actual_close, actual_return, 0, train_end
        ),
        validation=_slice_split(
            X, y, dates, previous_close, actual_close, actual_return, train_end, validation_end
        ),
        test=_slice_split(
            X,
            y,
            dates,
            previous_close,
            actual_close,
            actual_return,
            validation_end,
            sample_count,
        ),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


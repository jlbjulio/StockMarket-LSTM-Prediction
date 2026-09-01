import numpy as np

from stock_lstm.data import generate_demo_data
from stock_lstm.features import FEATURE_COLUMNS, build_features
from stock_lstm.preprocessing import prepare_sequences


def test_sequences_are_chronological_and_have_expected_shape() -> None:
    features = build_features(generate_demo_data(300))
    prepared = prepare_sequences(features, lookback=30, train_ratio=0.7, validation_ratio=0.15)
    assert prepared.train.X.shape[1:] == (30, len(FEATURE_COLUMNS))
    assert prepared.train.dates.max() < prepared.validation.dates.min()
    assert prepared.validation.dates.max() < prepared.test.dates.min()
    assert len(prepared.train.X) + len(prepared.validation.X) + len(prepared.test.X) == (
        len(features) - 30
    )


def test_scaler_is_fit_only_on_training_period() -> None:
    features = build_features(generate_demo_data(300))
    prepared = prepare_sequences(features, lookback=20, train_ratio=0.7, validation_ratio=0.15)
    train_end = int((len(features) - 20) * 0.7)
    last_train_target = 20 + train_end - 1
    expected_mean = features.loc[:, FEATURE_COLUMNS].iloc[:last_train_target].mean().to_numpy()
    np.testing.assert_allclose(prepared.feature_scaler.mean_, expected_mean)


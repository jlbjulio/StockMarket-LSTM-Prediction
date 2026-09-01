import numpy as np

from stock_lstm.data import generate_demo_data
from stock_lstm.features import FEATURE_COLUMNS, build_features


def test_features_are_finite_and_causal() -> None:
    data = generate_demo_data(250)
    full = build_features(data)
    shortened = build_features(data.iloc[:-10])
    common = shortened.index
    np.testing.assert_allclose(
        full.loc[common, FEATURE_COLUMNS],
        shortened.loc[common, FEATURE_COLUMNS],
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.isfinite(full.loc[:, FEATURE_COLUMNS].to_numpy()).all()


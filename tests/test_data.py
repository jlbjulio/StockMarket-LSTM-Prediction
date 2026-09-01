import numpy as np
import pandas as pd
import pytest

from stock_lstm.data import generate_demo_data, validate_market_data


def test_validate_market_data_normalizes_and_sorts() -> None:
    dates = pd.bdate_range("2024-01-01", periods=120)[::-1]
    frame = pd.DataFrame(
        {
            "Open": np.full(120, 100),
            "High": np.full(120, 102),
            "Low": np.full(120, 99),
            "Close": np.full(120, 101),
            "Volume": np.full(120, 1_000),
        },
        index=dates,
    )
    result = validate_market_data(frame)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.is_monotonic_increasing


def test_validate_market_data_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_market_data(pd.DataFrame({"close": [1, 2, 3]}))


def test_demo_data_is_deterministic() -> None:
    first = generate_demo_data(150, seed=7)
    second = generate_demo_data(150, seed=7)
    pd.testing.assert_frame_equal(first, second)


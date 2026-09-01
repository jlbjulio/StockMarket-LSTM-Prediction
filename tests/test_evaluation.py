import numpy as np
import pandas as pd

from stock_lstm.evaluation import evaluate_predictions


def test_evaluation_compares_model_with_persistence() -> None:
    previous = np.array([100.0, 101.0, 100.0])
    actual = np.array([101.0, 100.0, 102.0])
    actual_return = np.log(actual / previous)
    metrics, predictions = evaluate_predictions(
        pd.date_range("2024-01-01", periods=3),
        previous,
        actual,
        actual_return,
        actual_return,
    )
    assert metrics["lstm"]["rmse"] < 1e-10
    assert metrics["lstm"]["directional_accuracy_percent"] == 100.0
    assert "baseline_close" in predictions


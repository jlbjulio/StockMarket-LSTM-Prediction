from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "log_return",
    "high_low_range",
    "open_close_return",
    "volume_change",
    "volatility_5",
    "volatility_20",
    "momentum_5",
    "momentum_20",
    "sma_ratio_10",
    "sma_ratio_20",
    "volume_ratio_20",
    "rsi_14",
)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50) / 100


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build causal features; every row uses information available by that close."""
    close = data["close"]
    result = pd.DataFrame(index=data.index)
    result["log_return"] = np.log(close / close.shift(1))
    result["high_low_range"] = np.log(data["high"] / data["low"])
    result["open_close_return"] = np.log(close / data["open"])
    result["volume_change"] = np.log1p(data["volume"]).diff()
    result["volatility_5"] = result["log_return"].rolling(5).std()
    result["volatility_20"] = result["log_return"].rolling(20).std()
    result["momentum_5"] = np.log(close / close.shift(5))
    result["momentum_20"] = np.log(close / close.shift(20))
    result["sma_ratio_10"] = close / close.rolling(10).mean() - 1
    result["sma_ratio_20"] = close / close.rolling(20).mean() - 1
    result["volume_ratio_20"] = data["volume"] / data["volume"].rolling(20).mean() - 1
    result["rsi_14"] = _rsi(close)
    result["close"] = close
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    if result.empty:
        raise ValueError("Not enough observations to calculate rolling features")
    return result


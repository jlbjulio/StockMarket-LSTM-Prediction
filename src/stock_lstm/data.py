from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate an OHLCV dataframe."""
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        frame = frame.set_index("date")
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    frame.index = frame.index.tz_convert(None)
    frame.index.name = "date"

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")].dropna()

    invalid_price = (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalid_range = (frame["high"] < frame["low"]) | (frame["volume"] < 0)
    frame = frame.loc[~(invalid_price | invalid_range)]
    if len(frame) < 100:
        raise ValueError("At least 100 valid daily observations are required")
    return frame


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_market_data(pd.read_csv(path))


def download_market_data(
    ticker: str,
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted daily OHLCV prices and cache them as CSV."""
    safe_ticker = ticker.upper().replace("/", "-").replace("\\", "-")
    cache_path = Path(cache_dir) / f"{safe_ticker}.csv"
    if cache_path.exists() and not refresh:
        cached = load_csv(cache_path)
        requested_start = pd.Timestamp(start)
        requested_end = pd.Timestamp(end) if end else None
        covers_start = cached.index.min() <= requested_start
        cache_age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)
        if requested_end is None:
            recent_market_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
            covers_end = (
                cache_age < pd.Timedelta(hours=12)
                and cached.index.max() >= recent_market_date
            )
        else:
            covers_end = cached.index.max() >= requested_end - pd.Timedelta(days=7)
        if covers_start and covers_end:
            mask = cached.index >= requested_start
            if requested_end is not None:
                mask &= cached.index < requested_end
            return validate_market_data(cached.loc[mask])

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("Install project dependencies before downloading market data") from exc

    frame = yf.download(
        ticker.upper(),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        timeout=20,
    )
    if frame.empty:
        raise RuntimeError(f"No market data returned for ticker '{ticker}'")
    frame = validate_market_data(frame)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path)
    return frame


def generate_demo_data(rows: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic OHLCV data for an offline end-to-end smoke test."""
    if rows < 100:
        raise ValueError("rows must be at least 100")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=rows)
    market_cycle = 0.0015 * np.sin(np.arange(rows) / 18)
    log_returns = 0.00035 + market_cycle + rng.normal(0, 0.012, rows)
    close = 100 * np.exp(np.cumsum(log_returns))
    overnight = rng.normal(0, 0.003, rows)
    open_price = close * np.exp(overnight)
    spread = rng.uniform(0.002, 0.018, rows)
    high = np.maximum(open_price, close) * (1 + spread)
    low = np.minimum(open_price, close) * (1 - spread)
    volume = rng.lognormal(mean=16.1, sigma=0.35, size=rows)
    return validate_market_data(
        pd.DataFrame(
            {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
    )

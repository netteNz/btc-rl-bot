"""
Crypto training-data loader.

Reads the parquet files src/backfill.py and src/bar_builder.py maintain in data/raw/ and
turns them into the same normalized-return + stationary-indicator feature contract the
sibling stock bot's src/market_data.py produces, plus the 24/7-calendar deltas from
CLAUDE.md:

- No news features (crypto has none in scope).
- No session/open-close features (crypto has no sessions; nothing to drop, since none
  were ever added -- see feature_engineering.compute_cyclical_time_features docstring).
- Hour-of-day / day-of-week cyclical features are always added.
- 4h bars are derived by resampling 1h bars, never fetched/stored separately.
- No implicit network fetch: unlike the stock bot's get_tech_training_data (which calls
  yfinance inline), this module never talks to Coinbase. If data/raw/<PRODUCT>.parquet is
  missing or stale, the caller is told to run `python -m src.backfill` explicitly. This
  keeps the EXPERIMENT LAYER decoupled from the DATA LAYER per CLAUDE.md's process flow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import (
    compute_cyclical_time_features,
    compute_stationary_features,
    compute_vol_and_seasonality_features,
)
from src.products import BENCHMARK_PRODUCT, resolve_product

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

VALID_INTERVALS = ("1h", "4h")

# 24/7/365 crypto annualization -- CLAUDE.md: "Never reuse stock-calibrated Sharpe/alpha
# gate thresholds numerically." sqrt(8760) ~= 93.6 vs sqrt(1638) ~= 40.5 for hourly stock bars.
BARS_PER_YEAR = {
    "1h": 24 * 365,   # 8760
    "4h": 6 * 365,    # 2190
}


def raw_parquet_path(product_id: str) -> Path:
    return RAW_DATA_DIR / f"{product_id}.parquet"


def normalize_interval(interval: str) -> str:
    key = str(interval).strip().lower()
    if key not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Must be one of {VALID_INTERVALS}.")
    return key


def get_interval_bars_per_year(interval: str) -> int:
    return BARS_PER_YEAR[normalize_interval(interval)]


def load_raw_ohlcv(product_id: str) -> pd.DataFrame:
    path = raw_parquet_path(product_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached OHLCV at {path}. Run the backfill first:\n"
            f"  .venv/Scripts/python.exe -m src.backfill --product {product_id.lower()} --start 2020-01-01"
        )
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h bars into 4h bars aligned to UTC 00/04/08/12/16/20 boundaries."""
    indexed = df_1h.set_index("Date")
    agg = indexed.resample("4h", origin="epoch").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    agg = agg.dropna(subset=["Open", "High", "Low", "Close"]).reset_index()
    return agg


def _warn_on_gap_ratio(df: pd.DataFrame, interval: str) -> None:
    if len(df) < 10:
        return
    diffs = df["Date"].diff().dropna()
    expected = pd.Timedelta(hours=1) if interval == "1h" else pd.Timedelta(hours=4)
    gap_mask = diffs > (expected * 1.5)
    gap_ratio = float(gap_mask.mean())
    if gap_ratio > 0.02:
        warnings.warn(
            f"Bar-gap ratio is {gap_ratio:.1%} for interval={interval} -- unusually high for a "
            "24/7 market. Likely exchange downtime, a thin listing period, or a stale/partial "
            "backfill. Re-run src.backfill before training on this range.",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_training_data_quality(df: pd.DataFrame, interval: str) -> None:
    if "Date" not in df.columns:
        raise ValueError("Training data quality check failed: missing Date column.")
    if df["Date"].isna().any():
        raise ValueError("Training data quality check failed: Date column contains null values.")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Training data quality check failed: Date is not sorted ascending.")
    if df["Date"].duplicated().any():
        raise ValueError("Training data quality check failed: duplicate Date rows detected.")

    min_rows = 500
    if len(df) < min_rows:
        raise ValueError(
            f"Training data quality check failed: insufficient rows ({len(df)} < {min_rows}) "
            f"for interval={interval}. Backfill a longer history."
        )

    numeric_cols = [c for c in df.columns if c != "Date" and pd.api.types.is_numeric_dtype(df[c])]
    numeric = df[numeric_cols]
    bad_nan = numeric.isna().sum()
    bad_nan = bad_nan[bad_nan > 0]
    if not bad_nan.empty:
        top = ", ".join(f"{col}:{int(cnt)}" for col, cnt in bad_nan.sort_values(ascending=False).head(5).items())
        raise ValueError(f"Training data quality check failed: NaN values in numeric columns ({top}).")

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Training data quality check failed: non-finite numeric values (inf/-inf) detected.")

    _warn_on_gap_ratio(df, interval=interval)


def build_normalized_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Single-product analog of the stock bot's parse_and_normalize_ohlcv + build_training_frame.
    No cross-ticker groupby/basket averaging is needed here (one product per call), so this
    is a straight per-row transform.
    """
    cleaned = raw.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).copy()
    cleaned["Volume"] = cleaned["Volume"].clip(lower=0)
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)

    prev_close = cleaned["Close"].shift(1)
    frame = pd.DataFrame({"Date": cleaned["Date"]})
    frame["Open"] = (cleaned["Open"] / prev_close) - 1.0
    frame["High"] = (cleaned["High"] / cleaned["Close"]) - 1.0
    frame["Low"] = (cleaned["Low"] / cleaned["Close"]) - 1.0
    frame["Close"] = cleaned["Close"].pct_change()

    volume_values = cleaned["Volume"].to_numpy(dtype=float)
    logged = np.log1p(volume_values)
    frame["Volume"] = np.diff(logged, prepend=logged[0]) if logged.size else logged

    norm_cols = ["Open", "High", "Low", "Close", "Volume"]
    frame[norm_cols] = frame[norm_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["RawClose"] = cleaned["Close"].to_numpy()
    frame["RawVolume"] = cleaned["Volume"].to_numpy()

    return frame


# HANDOFF.md Branch E Phase E1 (2026-07-27): which secondary product's cross-rate return to
# add when training on a given primary product. Only BTC-USD has one -- ETH/SOL trading is
# blocked until BTC promotes (CLAUDE.md), so there's no cross-rate hypothesis defined for
# those as the primary yet.
CROSS_RATE_SECONDARY_PRODUCT = {"BTC-USD": "ETH-USD"}


def compute_cross_rate_feature(primary_normalized: pd.DataFrame, primary_product_id: str, interval: str) -> np.ndarray:
    """
    EthBtcRelReturn = secondary product's log return minus the primary's own log return,
    aligned to the primary's Date grid. Requires the secondary already backfilled
    (data/raw/<SECONDARY>.parquet) -- raises the same FileNotFoundError as load_raw_ohlcv if not.

    Secondary dates are ffilled onto the primary's grid across any gaps (rare -- see
    src/backfill.py's own gap-ratio warning) rather than dropping rows, since dropping would
    silently shrink the training window in a way unrelated to the primary product's own data
    quality.

    Returns an all-zero (neutral) array when no secondary is configured for this primary, so
    callers don't need a branch.
    """
    secondary_id = CROSS_RATE_SECONDARY_PRODUCT.get(primary_product_id)
    if secondary_id is None:
        return np.zeros(len(primary_normalized))

    secondary_raw = load_raw_ohlcv(secondary_id)
    if interval == "4h":
        secondary_raw = resample_to_4h(secondary_raw)

    secondary_close = secondary_raw.set_index("Date")["Close"].sort_index()
    primary_dates = pd.to_datetime(primary_normalized["Date"])
    aligned_close = secondary_close.reindex(primary_dates).ffill()
    secondary_log_return = np.log(aligned_close / aligned_close.shift(1)).fillna(0.0).to_numpy()

    primary_close = primary_normalized["RawClose"]
    primary_log_return = np.log(primary_close / primary_close.shift(1)).fillna(0.0).to_numpy()

    return secondary_log_return - primary_log_return


def get_crypto_training_data(
    product: str = "btc-usd",
    interval: str = "1h",
    use_stationary_features: bool = True,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load, normalize, and feature-engineer crypto OHLCV for training/evaluation.

    Returns a DataFrame with: Date, RawClose, plus either
      - stationary features (14 indicator cols) + 4 cyclical cols  (use_stationary_features=True), or
      - normalized OHLCV (5 cols) + 4 cyclical cols                (use_stationary_features=False)
    """
    interval = normalize_interval(interval)
    product_id = resolve_product(product)

    raw = load_raw_ohlcv(product_id)
    if interval == "4h":
        raw = resample_to_4h(raw)

    if start:
        raw = raw[raw["Date"] >= pd.to_datetime(start)]
    if end:
        raw = raw[raw["Date"] <= pd.to_datetime(end)]
    raw = raw.reset_index(drop=True)

    normalized = build_normalized_frame(raw)
    cyclical = compute_cyclical_time_features(normalized)
    vol_seasonality = compute_vol_and_seasonality_features(normalized)

    if use_stationary_features:
        indicators = compute_stationary_features(normalized)
        training_data = indicators.copy()
        training_data["RawClose"] = normalized["RawClose"].to_numpy()
    else:
        training_data = normalized.copy()

    for col in ["HourSin", "HourCos", "DowSin", "DowCos"]:
        training_data[col] = cyclical[col].to_numpy()

    for col in ["VolRatio_6_48", "RelVolumeByHourOfWeek"]:
        training_data[col] = vol_seasonality[col].to_numpy()

    training_data["EthBtcRelReturn"] = compute_cross_rate_feature(normalized, product_id, interval)

    training_data = training_data.sort_values("Date").reset_index(drop=True)
    _validate_training_data_quality(training_data, interval=interval)
    return training_data


def load_benchmark_close_prices(start: str | pd.Timestamp, end: str | pd.Timestamp, interval: str = "1h") -> pd.DataFrame:
    """
    Buy-and-hold BTC benchmark series (CLAUDE.md G3 swap: QQQ -> B&H BTC), aligned to the
    same bar grid as the strategy being evaluated. Always BENCHMARK_PRODUCT (BTC-USD),
    even when the traded product is BTC-USD itself -- for the BTC baseline this collapses
    to "alpha vs its own buy-and-hold," which is exactly the degenerate-always-long catch
    G3 is meant to be; for future ETH/SOL runs it becomes a genuine market-beta benchmark.
    """
    interval = normalize_interval(interval)
    raw = load_raw_ohlcv(BENCHMARK_PRODUCT)
    if interval == "4h":
        raw = resample_to_4h(raw)

    window = raw[(raw["Date"] >= pd.to_datetime(start) - pd.Timedelta(hours=1)) & (raw["Date"] <= pd.to_datetime(end) + pd.Timedelta(hours=1))]
    return window[["Date", "Close"]].sort_values("Date").reset_index(drop=True)


def benchmark_equity_curve(period_df: pd.DataFrame, benchmark_prices: pd.DataFrame, initial_balance: float = 1000.0) -> pd.Series:
    """Align the BTC benchmark close price to `period_df`'s Date grid and scale to an equity curve."""
    if "Date" not in period_df.columns:
        raise ValueError("Benchmark equity curve requires a Date column in the period dataframe.")

    aligned = pd.DataFrame({"Date": pd.to_datetime(period_df["Date"])}).merge(benchmark_prices, on="Date", how="left")
    aligned["Close"] = aligned["Close"].ffill().bfill()
    if aligned["Close"].isna().any():
        raise ValueError(
            "Unable to align BTC benchmark prices to experiment dates. "
            "Ensure data/raw/BTC-USD.parquet covers the same date range as the traded product."
        )

    first_price = max(float(aligned["Close"].iloc[0]), 1e-8)
    return float(initial_balance) * (aligned["Close"] / first_price)

"""
Historical OHLCV backfill via the Coinbase Advanced Trade REST candles endpoint.

Per CLAUDE.md's process flow, data/raw/<PRODUCT>.parquet always stores 1h bars -- 4h bars
(the alternative fee-floor config) are derived on the fly by src/market_data.py via
resampling, not stored separately. This keeps exactly one on-disk source of truth per
product.

Usage:
    .venv/Scripts/python.exe -m src.backfill --product btc-usd --start 2020-01-01
    .venv/Scripts/python.exe -m src.backfill --product btc-usd            # top up to now
    .venv/Scripts/python.exe -m src.backfill --product btc-usd --refresh  # full refetch

Uses the public (unauthenticated) candles endpoint -- historical OHLCV is public market
data and does not need the CDP key at all.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.coinbase_client import (
    GRANULARITY_MAP,
    MAX_CANDLES_PER_REQUEST,
    NATIVE_GRANULARITY_SECONDS,
    get_rest_client,
)
from src.products import PRODUCT_PRESETS, resolve_product

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

REQUEST_SLEEP_SECONDS = 0.25  # polite pacing for the public candles endpoint


def raw_parquet_path(product_id: str) -> Path:
    return RAW_DATA_DIR / f"{product_id}.parquet"


def _candles_to_frame(candles) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    rows = [
        {
            "Date": pd.to_datetime(int(c.start), unit="s", utc=True).tz_localize(None),
            "Open": float(c.open),
            "High": float(c.high),
            "Low": float(c.low),
            "Close": float(c.close),
            "Volume": float(c.volume),
        }
        for c in candles
    ]
    return pd.DataFrame(rows)


def fetch_hourly_candles(product_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Paginate the public ONE_HOUR candles endpoint across [start, end] (UTC, inclusive-ish).
    Coinbase caps each response at ~350 candles; we request MAX_CANDLES_PER_REQUEST at a time
    to stay comfortably under that.
    """
    client = get_rest_client(require_credentials=False)
    granularity = GRANULARITY_MAP["1h"]
    step_seconds = NATIVE_GRANULARITY_SECONDS["1h"] * MAX_CANDLES_PER_REQUEST

    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    frames: list[pd.DataFrame] = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + step_seconds, end_ts)
        response = client.get_public_candles(
            product_id=product_id,
            start=str(chunk_start),
            end=str(chunk_end),
            granularity=granularity,
        )
        frame = _candles_to_frame(response.candles)
        if not frame.empty:
            frames.append(frame)
        chunk_start = chunk_end
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not frames:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return combined


def _warn_on_gaps(df: pd.DataFrame) -> None:
    if len(df) < 3:
        return
    diffs = df["Date"].diff().dropna()
    expected = pd.Timedelta(hours=1)
    gap_count = int((diffs > expected * 1.5).sum())
    if gap_count > 0:
        total_missing_hours = int((diffs[diffs > expected * 1.5] / expected).sum()) - gap_count
        print(
            f"WARNING: {gap_count} gap(s) detected in {df['Date'].min()}..{df['Date'].max()} "
            f"(~{total_missing_hours} missing hourly bars). Exchange downtime or a thin period "
            f"is the usual cause on a 24/7 market; re-run with --refresh if it looks wrong."
        )


def backfill(product_key: str, start: str | None, end: str | None, refresh: bool) -> Path:
    product_id = resolve_product(product_key)
    out_path = raw_parquet_path(product_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    end_ts = pd.to_datetime(end) if end else now_utc

    existing = None
    if out_path.exists() and not refresh:
        existing = pd.read_parquet(out_path)
        if not existing.empty:
            latest = pd.to_datetime(existing["Date"]).max()
            fetch_start = latest + pd.Timedelta(hours=1)
            print(f"Existing cache found: {len(existing)} rows through {latest}. Topping up from {fetch_start}.")
        else:
            existing = None

    if existing is None:
        if not start:
            raise ValueError(
                f"No cached data at {out_path} and --start not provided. "
                "Provide --start YYYY-MM-DD for the initial backfill."
            )
        fetch_start = pd.to_datetime(start)

    if fetch_start >= end_ts:
        print(f"Already up to date through {end_ts}. Nothing to fetch.")
        return out_path

    print(f"Fetching {product_id} 1h candles: {fetch_start} -> {end_ts}")
    new_data = fetch_hourly_candles(product_id, fetch_start, end_ts)
    print(f"Fetched {len(new_data)} new candles.")

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = (
        combined.drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if combined.empty:
        raise RuntimeError(f"No candle data returned for {product_id}. Check the product_id and date range.")

    _warn_on_gaps(combined)

    combined.to_parquet(out_path, index=False)
    print(f"Saved {len(combined)} rows ({combined['Date'].min()} .. {combined['Date'].max()}) to {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill 1h OHLCV candles from Coinbase Advanced Trade.")
    parser.add_argument("--product", default="btc-usd", choices=list(PRODUCT_PRESETS.keys()),
                        help="Product preset to backfill.")
    parser.add_argument("--start", default=None, help="Backfill start date (YYYY-MM-DD, UTC). Required for a fresh cache.")
    parser.add_argument("--end", default=None, help="Backfill end date (YYYY-MM-DD, UTC). Defaults to now.")
    parser.add_argument("--refresh", action="store_true", help="Ignore existing cache and refetch the full range.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backfill(args.product, args.start, args.end, args.refresh)


if __name__ == "__main__":
    main()

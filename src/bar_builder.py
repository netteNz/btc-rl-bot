"""
Live bar continuation: WS `ticker` channel -> hourly OHLCV bars appended to
data/raw/<PRODUCT>.parquet, so the same parquet file backfill.py writes stays current
between REST backfill runs.

Design note on volume: the `ticker` channel reports last-trade price plus a rolling
`volume_24_h` figure -- it does NOT include a per-tick trade size. We approximate a bar's
volume as the delta of `volume_24_h` between the first and last tick observed inside that
bar's hour window. Because volume_24_h is a rolling sum, this is an approximation (it can
be slightly off right at the rolling edge) but it is the only volume signal the ticker
channel exposes; CLAUDE.md specifies the ticker channel (not `candles` or `market_trades`)
as the source, so this tradeoff is accepted rather than switched to a different channel.

A bar is only finalized once a tick from the *next* hour arrives, so the last bar of a
run stays open until the next tick crosses the boundary. During low-liquidity hours this
can delay finalization by longer than usual, but never fabricates a bar from silence.

Usage:
    .venv/Scripts/python.exe -m src.bar_builder --product btc-usd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from coinbase.websocket.types.websocket_response import WebsocketResponse

from src.backfill import raw_parquet_path
from src.coinbase_client import get_ws_client
from src.products import PRODUCT_PRESETS, resolve_product


class BarBuilder:
    """Aggregates `ticker` channel messages for a single product into hourly OHLCV bars."""

    def __init__(self, product_id: str, on_bar_complete: Callable[[dict], None]):
        self.product_id = product_id
        self.on_bar_complete = on_bar_complete
        self._bucket_start: Optional[pd.Timestamp] = None
        self._open = self._high = self._low = self._close = None
        self._first_volume_24h: Optional[float] = None
        self._last_volume_24h: Optional[float] = None

    def handle_message(self, raw_message: str) -> None:
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return
        if data.get("channel") != "ticker":
            return

        response = WebsocketResponse(data)
        timestamp = response.timestamp
        for event in response.events:
            for ticker in (event.tickers or []):
                if ticker.product_id != self.product_id or ticker.price is None:
                    continue
                self._ingest_tick(timestamp, ticker)

    def _ingest_tick(self, timestamp_str: str, ticker) -> None:
        ts = pd.to_datetime(timestamp_str, utc=True).tz_localize(None)
        bucket = ts.floor("h")
        price = float(ticker.price)
        volume_24h = float(ticker.volume_24_h) if ticker.volume_24_h is not None else None

        if self._bucket_start is None:
            self._start_bucket(bucket, price, volume_24h)
            return

        if bucket > self._bucket_start:
            self._finalize_bar()
            self._start_bucket(bucket, price, volume_24h)
            return

        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        if volume_24h is not None:
            self._last_volume_24h = volume_24h

    def _start_bucket(self, bucket: pd.Timestamp, price: float, volume_24h: Optional[float]) -> None:
        self._bucket_start = bucket
        self._open = self._high = self._low = self._close = price
        self._first_volume_24h = volume_24h
        self._last_volume_24h = volume_24h

    def _finalize_bar(self) -> None:
        volume = 0.0
        if self._first_volume_24h is not None and self._last_volume_24h is not None:
            volume = max(self._last_volume_24h - self._first_volume_24h, 0.0)
        bar = {
            "Date": self._bucket_start,
            "Open": self._open,
            "High": self._high,
            "Low": self._low,
            "Close": self._close,
            "Volume": volume,
        }
        self.on_bar_complete(bar)


def append_bar_to_parquet(product_id: str, bar: dict) -> None:
    path = raw_parquet_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([bar])

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined = (
        combined.drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    combined.to_parquet(path, index=False)
    print(
        f"Bar closed [{product_id}] {bar['Date']}  "
        f"O={bar['Open']:.2f} H={bar['High']:.2f} L={bar['Low']:.2f} C={bar['Close']:.2f} V~={bar['Volume']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live WS ticker -> hourly bar continuation.")
    parser.add_argument("--product", default="btc-usd", choices=list(PRODUCT_PRESETS.keys()))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    product_id = resolve_product(args.product)

    builder = BarBuilder(product_id, on_bar_complete=lambda bar: append_bar_to_parquet(product_id, bar))
    client = get_ws_client(on_message=builder.handle_message)

    print(f"Connecting to Coinbase WS ticker channel for {product_id}...")
    client.open()
    client.ticker([product_id])
    print("Subscribed. Streaming bars (Ctrl+C to stop)...")
    try:
        client.run_forever_with_exception_check()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.close()


if __name__ == "__main__":
    main()

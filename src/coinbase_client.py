"""
Coinbase Advanced Trade client factory.

Centralizes credential loading (.env -> CDP_API_KEY / CDP_API_SECRET) and the handful of
constants (product presets, candle granularity mapping) shared by the backfill, bar_builder,
and portfolio_tracker scripts.

Security posture (see CLAUDE.md):
- The CDP key configured in .env is View-only. Nothing in this repo requests Trade/Transfer
  scope, and get_rest_client() does not validate or elevate scope -- it just authenticates.
- Credentials are never logged. Do not add print/log statements that include api_key or
  api_secret anywhere in this module or its callers.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from coinbase.rest import RESTClient
from coinbase.websocket import WSClient

from src.products import DEFAULT_PRODUCT, PRODUCT_PRESETS, resolve_product

__all__ = [
    "DEFAULT_PRODUCT",
    "PRODUCT_PRESETS",
    "resolve_product",
    "GRANULARITY_MAP",
    "NATIVE_GRANULARITY_SECONDS",
    "MAX_CANDLES_PER_REQUEST",
    "get_rest_client",
    "get_ws_client",
]

# Coinbase Advanced Trade candle granularities that exist natively. Notably there is no
# FOUR_HOUR granularity -- CLAUDE.md's "4h bars" alternative is built by resampling four
# consecutive ONE_HOUR candles in src/market_data.py, not by requesting it directly.
GRANULARITY_MAP: dict[str, str] = {
    "1h": "ONE_HOUR",
    "4h": "ONE_HOUR",  # resampled client-side; see src/market_data.py resample_to_4h()
}
NATIVE_GRANULARITY_SECONDS: dict[str, int] = {
    "1h": 3600,
}

# Coinbase caps candle responses at 350 rows per request.
MAX_CANDLES_PER_REQUEST = 300


def get_rest_client(require_credentials: bool = True) -> RESTClient:
    """
    Build a RESTClient from .env credentials.

    require_credentials=False allows calling public endpoints (get_public_candles, etc.)
    without CDP_API_KEY/CDP_API_SECRET set, which is useful for historical backfill in
    environments that haven't been provisioned with a key yet.
    """
    api_key = os.getenv("CDP_API_KEY")
    api_secret = os.getenv("CDP_API_SECRET")
    if require_credentials and (not api_key or not api_secret):
        raise RuntimeError(
            "CDP_API_KEY / CDP_API_SECRET not found in environment. "
            "Add them to .env (see CLAUDE.md Security Posture) or pass require_credentials=False "
            "for public-only endpoints."
        )
    return RESTClient(api_key=api_key or None, api_secret=api_secret or None)


def get_ws_client(on_message, on_open=None, on_close=None) -> WSClient:
    """Build a WSClient from .env credentials for the public ticker channel."""
    api_key = os.getenv("CDP_API_KEY")
    api_secret = os.getenv("CDP_API_SECRET")
    return WSClient(
        api_key=api_key or None,
        api_secret=api_secret or None,
        on_message=on_message,
        on_open=on_open,
        on_close=on_close,
    )

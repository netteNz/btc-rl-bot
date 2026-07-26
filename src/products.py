"""
Product presets shared by the data layer (coinbase_client, backfill, bar_builder) and the
training pipeline (market_data, experiments). Deliberately has zero dependency on the
`coinbase` SDK or python-dotenv -- the training pipeline should never need network
credentials just to resolve a product id from a CLI flag.
"""

from __future__ import annotations

# BTC-USD is the only promoted/locked product today; ETH/SOL are pre-wired per CLAUDE.md
# ("ETH/SOL expansion only after BTC promotes") but must not be used for training until
# that gate is cleared.
PRODUCT_PRESETS: dict[str, str] = {
    "btc-usd": "BTC-USD",
    "eth-usd": "ETH-USD",
    "sol-usd": "SOL-USD",
}
DEFAULT_PRODUCT = "btc-usd"

# The product always benchmarked against for G3 alpha (buy-and-hold BTC), regardless of
# which product is being traded -- see CLAUDE.md "G3 benchmark rewrites: QQQ -> B&H BTC".
# This plays the same role QQQ played for individual stock tickers: a market-beta
# benchmark, not the traded asset's own price series.
BENCHMARK_PRODUCT = "BTC-USD"


def resolve_product(product_key: str) -> str:
    """Map a CLI-friendly key (btc-usd) or a literal Coinbase product_id (BTC-USD) to the
    canonical product_id."""
    key = str(product_key).strip()
    if key.lower() in PRODUCT_PRESETS:
        return PRODUCT_PRESETS[key.lower()]
    return key.upper()

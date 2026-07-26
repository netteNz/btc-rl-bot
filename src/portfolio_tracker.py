"""
Paper-trading portfolio tracker.

Per CLAUDE.md, this runs parallel to the RL pipeline, not inside it: WS `ticker` marks
positions to market locally between reconciliations, REST `get_portfolio_breakdown()`
reconciles authoritative balances on a slow interval, and WS `user` channel fills trigger
an immediate out-of-cycle reconciliation. `get_accounts()` is paginated to exhaustion on
every reconciliation -- `has_next`/`cursor` mean a single unpaginated call can silently
miss balances on later pages, which would corrupt the net-worth estimate.

This module only reads account/portfolio state. The CDP key is View-only (see CLAUDE.md
Security Posture) and nothing here places, previews, or cancels orders -- there is no
live-order capability in this repo until paper-trading validation is complete and
Trade/Transfer scope is explicitly approved.

Usage:
    .venv/Scripts/python.exe -m src.portfolio_tracker --product btc-usd
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

from coinbase.websocket.types.websocket_response import WebsocketResponse

from src.coinbase_client import get_rest_client, get_ws_client
from src.products import PRODUCT_PRESETS, resolve_product

RECONCILE_INTERVAL_SECONDS = 60


class PortfolioTracker:
    def __init__(self, product_ids: list[str]):
        self.product_ids = product_ids
        self.rest = get_rest_client(require_credentials=True)
        self.portfolio_uuid: Optional[str] = None
        self.balances: dict[str, float] = {}
        self.last_prices: dict[str, float] = {}
        self._last_reconcile_at: float = 0.0

    def _resolve_default_portfolio(self) -> Optional[str]:
        response = self.rest.get_portfolios(portfolio_type="DEFAULT")
        portfolios = response.portfolios or []
        return portfolios[0].uuid if portfolios else None

    def fetch_all_accounts(self) -> list:
        """Paginate get_accounts() to exhaustion via has_next/cursor. Never call this
        without draining the cursor -- a single page can silently omit balances."""
        accounts = []
        cursor = None
        while True:
            response = self.rest.get_accounts(limit=250, cursor=cursor)
            accounts.extend(response.accounts or [])
            if not getattr(response, "has_next", False):
                break
            cursor = getattr(response, "cursor", None)
            if not cursor:
                break
        return accounts

    def reconcile(self) -> None:
        """Slow-interval authoritative refresh: paginated accounts + portfolio breakdown."""
        accounts = self.fetch_all_accounts()
        self.balances = {}
        for account in accounts:
            balance = getattr(account, "available_balance", None)
            if not balance or balance.value is None:
                continue
            self.balances[account.currency] = self.balances.get(account.currency, 0.0) + float(balance.value)

        if self.portfolio_uuid is None:
            self.portfolio_uuid = self._resolve_default_portfolio()

        self._last_reconcile_at = time.time()
        print(f"[reconcile] {len(accounts)} accounts across {len(self.balances)} currencies: {self.balances}")

    def maybe_reconcile(self, interval_seconds: int = RECONCILE_INTERVAL_SECONDS) -> None:
        if time.time() - self._last_reconcile_at >= interval_seconds:
            self.reconcile()

    def mark_to_market(self, product_id: str, price: float) -> None:
        self.last_prices[product_id] = price

    def net_worth_estimate(self) -> float:
        """Cash (USD/USDC) + mark-to-market value of held base currencies. Approximate
        between reconciliations -- authoritative balances only come from reconcile()."""
        total = self.balances.get("USD", 0.0) + self.balances.get("USDC", 0.0)
        for product_id, price in self.last_prices.items():
            base_currency = product_id.split("-")[0]
            qty = self.balances.get(base_currency, 0.0)
            total += qty * price
        return total

    def handle_ws_message(self, raw_message: str) -> None:
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return

        channel = data.get("channel")
        if channel == "ticker":
            response = WebsocketResponse(data)
            for event in response.events:
                for ticker in (event.tickers or []):
                    if ticker.product_id in self.product_ids and ticker.price is not None:
                        self.mark_to_market(ticker.product_id, float(ticker.price))

        elif channel == "user":
            # Any order/fill update triggers an immediate out-of-cycle reconciliation
            # rather than waiting for the slow REST interval.
            response = WebsocketResponse(data)
            for event in response.events:
                if event.orders:
                    print(f"[user] {len(event.orders)} order update(s) received -- reconciling now.")
                    self.reconcile()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-trading portfolio tracker (read-only, View-only key).")
    parser.add_argument("--product", action="append", default=None,
                        help="Product to mark-to-market (repeatable). Default: all presets.")
    parser.add_argument("--reconcile-interval", type=int, default=RECONCILE_INTERVAL_SECONDS,
                        help="Seconds between REST reconciliations.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    products = args.product or list(PRODUCT_PRESETS.keys())
    product_ids = [resolve_product(p) for p in products]

    tracker = PortfolioTracker(product_ids)
    print("Running initial reconciliation...")
    tracker.reconcile()

    client = get_ws_client(on_message=tracker.handle_ws_message)
    client.open()
    client.ticker(product_ids)
    client.user(product_ids)
    print(f"Subscribed to ticker + user channels for {product_ids}. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            client.raise_background_exception()
            tracker.maybe_reconcile(args.reconcile_interval)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.close()


if __name__ == "__main__":
    main()

"""
Fee application and position-sizing invariant tests.

Added after btc-baseline-v1's degenerate always-flat collapse (turnover-penalty/fee
double-counting -- see CLAUDE.md Known Failure Patterns) and a second, more fundamental
bug found while designing btc-baseline-v2: position sizing was ported from the stock bot
as whole-share flooring (`int(net_worth * weight // price)`). On BTC-USD (observed range
~$16.5K-$126K in this repo's backfill) with the env's un-configurable $1,000 default
balance, floor(1000 / price) is 0 for every bar in the dataset -- the agent was
structurally incapable of ever entering a position, independent of reward shaping. This
alone plausibly explains v1's 0% trade rate; the reward fix was necessary but may not
have been sufficient.

Invariants locked in here:
  1. A flat-forever episode incurs zero total fees.
  2. A single round trip incurs exactly 2x the taker rate (on the traded notional).
  3. Fees apply on position *change* only -- holding a position across many bars, through
     real (moving) prices, never accrues fees beyond the single entry trade. Fractional
     sizing recomputes target_shares from net_worth on every bar unless this is explicitly
     guarded; a nonzero cash residual from a prior fee (fees are additive on top of
     notional, not deducted from it) will otherwise drift in and out of a tradeable
     notional as price moves and fire economically meaningless "hold" trades.
  4. Position sizing is fractional: an account balance smaller than the asset price must
     still be able to enter a position (the whole-share bug above, pinned down so it can't
     silently return).

These exercise PositionManager accounting through CryptoTradingEnv.step(), not the reward
shaping on top of it -- reward_turnover_penalty_scale is irrelevant to fee *accounting*
(info["execution_fee"]), only to the reward signal derived from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.env.trading_env import CryptoTradingEnv

TAKER_RATE = 0.006
INITIAL_BALANCE = 1000.0
FLAT_PRICE = 97.13

# Real BTC-USD hourly closes are in this ballpark (this repo's backfill spans roughly
# $16.5K-$126K) -- comfortably above INITIAL_BALANCE, which is exactly the regime that
# broke whole-share sizing.
BTC_SCALE_PRICES = [
    60123.45, 60310.20, 60287.90, 60455.10, 60398.75, 60510.33, 60622.80, 60580.15,
    60701.40, 60655.90, 60789.20, 60844.55, 60712.30, 60899.10, 60950.40,
]


def _make_price_df(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Date": dates,
        "RawClose": np.array(prices, dtype=float),
        "Open": np.zeros(n),
        "High": np.zeros(n),
        "Low": np.zeros(n),
        "Close": np.zeros(n),
        "Volume": np.zeros(n),
    })


def _make_flat_price_df(n: int, price: float = FLAT_PRICE) -> pd.DataFrame:
    return _make_price_df([price] * n)


def _run_env(df: pd.DataFrame, actions: list[int], initial_balance: float = INITIAL_BALANCE) -> tuple[float, int, CryptoTradingEnv]:
    """Returns (total_fees_paid, trades_executed, env)."""
    env = CryptoTradingEnv(
        df,
        initial_balance=initial_balance,
        transaction_cost_rate=TAKER_RATE,
        trade_penalty=0.0,
        execution_mode="same_bar",
        binary_actions=True,
        min_hold_bars=0,
        reward_turnover_penalty_scale=0.0,
        min_trade_notional=1.0,
    )
    obs, _ = env.reset()
    total_fees = 0.0
    trades_executed = 0
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        total_fees += float(info["execution_fee"])
        if info["execution_fee"] > 0:
            trades_executed += 1
        if terminated or truncated:
            break
    return total_fees, trades_executed, env


def test_flat_forever_incurs_zero_fees():
    df = _make_flat_price_df(n=30)
    actions = [0] * (len(df) - 1)  # never go long
    total_fees, trades_executed, env = _run_env(df, actions)
    assert total_fees == 0.0
    assert trades_executed == 0
    assert env.pm.shares_held == 0


def test_single_round_trip_charges_exactly_two_times_taker_rate():
    df = _make_flat_price_df(n=30, price=FLAT_PRICE)
    n_steps = len(df) - 1
    actions = [1] * n_steps
    for i in range(5, n_steps):
        actions[i] = 0  # flip flat at step 5 and stay flat

    total_fees, trades_executed, _ = _run_env(df, actions)

    # Fractional sizing spends exactly the target notional (no whole-share rounding loss),
    # so a flat-price round trip costs exactly 2x the taker rate on the entry notional,
    # which (fully invested, starting from cash) equals INITIAL_BALANCE exactly.
    expected_fee = 2 * TAKER_RATE * INITIAL_BALANCE

    assert trades_executed == 2  # exactly one entry, one exit
    assert total_fees == pytest.approx(expected_fee, rel=1e-9)


def test_holding_position_longer_does_not_accrue_additional_fees():
    """Fees apply on position change only -- never per bar held."""
    df_short = _make_flat_price_df(n=10)
    df_long = _make_flat_price_df(n=25)

    actions_short = [1] * (len(df_short) - 1)  # go long once, keep signaling long
    actions_long = [1] * (len(df_long) - 1)    # same, but held for many more bars

    fees_short, trades_short, _ = _run_env(df_short, actions_short)
    fees_long, trades_long, _ = _run_env(df_long, actions_long)

    assert trades_short == 1
    assert trades_long == 1
    assert fees_short > 0.0
    assert fees_short == pytest.approx(fees_long)


def test_holding_through_moving_price_does_not_drift_or_rebalance():
    """
    Regression test for the phantom-rebalancing bug: continuously signaling "long" through
    a moving (not flat) price series must fire exactly one fee event -- the entry -- and
    shares_held must not drift afterward. Before the fix, a nonzero post-fee cash residual
    combined with real price movement caused target_shares to be silently re-derived from
    net_worth on every bar, firing dozens of economically meaningless "hold" trades.
    """
    df = _make_price_df(BTC_SCALE_PRICES)
    actions = [1] * (len(df) - 1)  # signal long every bar, never flip

    total_fees, trades_executed, env = _run_env(df, actions)

    assert trades_executed == 1
    entry_notional = INITIAL_BALANCE  # fully invested from cash, fractional sizing
    assert total_fees == pytest.approx(TAKER_RATE * entry_notional, rel=1e-9)


def test_fractional_sizing_allows_entry_when_balance_is_below_asset_price():
    """
    Pins down the whole-share-flooring bug directly: with a $1,000 balance against a
    BTC-scale price (~$60K), int(net_worth // price) floors to 0 forever -- the agent
    could never enter a position regardless of reward shaping or trade decisions. Fractional
    sizing must allow a real, nonzero position here.
    """
    df = _make_price_df(BTC_SCALE_PRICES)
    actions = [1] * (len(df) - 1)

    _, _, env = _run_env(df, actions, initial_balance=1000.0)

    assert env.pm.shares_held > 0.0
    # Sanity: at these prices, a whole-share-floor implementation would size to exactly 0.
    assert env.pm.shares_held < 1.0

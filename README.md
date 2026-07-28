# coinbase-rl-bot

Reinforcement-learning trading bot for spot crypto on Coinbase Advanced Trade. Binary PPO
(long/flat), fee-aware reward, 24/7 calendar. BTC-USD baseline first; ETH/SOL only after BTC
promotes. Sibling project to `reinforcement-learning-stocks` — same pipeline, crypto-specific
economics (fees, calendar, benchmark). Full architecture and ground rules: [CLAUDE.md](CLAUDE.md).
Current experiment status and next steps: [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md).

## Layout

```
src/
  coinbase_client.py     Coinbase Advanced Trade API wrapper
  backfill.py             REST candle history -> data/raw/<PRODUCT>.parquet
  bar_builder.py           WebSocket ticker -> live bar continuation
  market_data.py           parquet loading / preprocessing
  feature_engineering.py   27-feature stationary observation space + cyclical time features
  env/trading_env.py       CryptoTradingEnv (fee model, 24/7 calendar, B&H BTC benchmark)
  experiments.py           sweep runner (PPO training, multi-seed)
  signal_analytics.py      accuracy / actionable-signal analytics
  ensemble.py              ensemble model support
  portfolio_tracker.py     paper-trading mark-to-market (view-only key, no live orders)
  products.py               Coinbase product metadata

scripts/
  evaluate_sweep.py            cross-seed 6-gate promotion check (authoritative)
  sanity_scan.py                signal integrity scan
  generate_ensemble_config.py   ensemble config generator (verify seed pins manually)

tests/
  test_fee_model.py    locks in the fee-application invariant (flat-forever = 0 fees,
                        one round trip = 2x taker rate); run before any sweep batch

data/
  raw/                       OHLCV parquet, one file per product (gitignored)
  experiment_leaderboard.csv           per-run results, all sweeps appended here
  experiment_summary.json              latest run's top-3 + snapshot pointers
  experiment_snapshots/                per-run model .zip + leaderboard snapshot

docs/
  NEXT_EXPERIMENTS.md    branch-by-branch experiment plan and results log
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Credentials in `.env` (`CDP_API_KEY`, `CDP_API_SECRET`) — never committed, never logged. API key is
**view-only**; no live order placement until paper-trading validation is complete and explicitly
approved.

## Running the pipeline

Always invoke Python as `.\.venv\Scripts\python.exe`, never bare `python`.

```powershell
# 1. Backfill historical candles
.\.venv\Scripts\python.exe -m src.backfill --product BTC-USD --interval 1h

# 2. Run a sweep (example: baseline template, see docs/NEXT_EXPERIMENTS.md for current batch)
.\.venv\Scripts\python.exe -m src.experiments --product BTC-USD --interval 1h --binary-actions --min-hold-bars 6 --reward-mode sharpe --transaction-cost-rate 0.006 --initial-balance 10000 --min-trade-notional 1.0 --ent-coefs 0.01,0.02,0.05 --timesteps 40000 --seeds 3,7,13,21,42 --execution-mode next_bar --reward-turnover-penalty-scale 0.0 --max-weight-delta-per-step 0.10 --use-stationary-features --n-envs 1 --run-label "<label>" --append

# 3. Evaluate against the 6-gate promotion framework (cross-seed, authoritative)
.\.venv\Scripts\python.exe scripts\evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv --label "<label>"

# 4. Signal integrity check
.\.venv\Scripts\python.exe scripts\sanity_scan.py

# 5. Only after a labeled sweep clears all 6 gates: generate ensemble config (verify seed pins manually)
.\.venv\Scripts\python.exe scripts\generate_ensemble_config.py
```

Before any sweep batch:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fee_model.py
```

## Status

No BTC-USD config has cleared the 6-gate promotion bar yet. See
[CLAUDE.md → Handoff Status](CLAUDE.md#handoff-status-2026-07-27) for the full experiment history
and [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md) for the current diagnosis and next batch.

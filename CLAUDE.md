# CLAUDE.md — coinbase-rl-bot

Crypto RL trading bot. Sibling project to `reinforcement-learning-stocks` — same methodology, different market economics. Read this entire file before designing experiments or modifying the env.

## Project Ground Truth

**Mission:** Adapt the Binary PPO gold-standard architecture to spot crypto on Coinbase Advanced Trade. BTC-USD baseline first; ETH/SOL expansion only after BTC promotes.

**Architecture:** Binary PPO — long/flat discrete actions. On spot crypto this is not a constraint, it is the market's actual action space (no shorting on spot). Do not propose continuous sizing or SAC.

**Data source:** Coinbase Advanced Trade API via `coinbase-advanced-py`. Historical backfill via REST candles endpoint; live continuation via WebSocket `ticker` channel → `bar_builder.py`. OHLCV parquet in `data/raw/`, one file per product, same schema convention as the stock bot's store.

## Non-Negotiable: The Three Crypto Deltas

These are the only structural differences from the stock bot. Everything not listed here carries over unchanged.

### 1. Fees dominate (the big one)
- Retail tier: ~0.40% maker / 0.60% taker under $10K 30-day volume. Taker/taker round trip ≈ **1.2% of notional per flip**.
- **Fee-aware reward is mandatory.** Per-flip taker penalty modeled explicitly in the env. A leaderboard built on fee-free reward is fiction — never report or compare fee-free results.
- Bar interval floor: **1h bars, min-hold ≥ 6** (or 4h bars, min-hold 3). Never design sweeps at minute bars — expected edge per trade cannot clear the fee floor.
- Fee params live in env config, parameterized (maker-assumption is a future sweep variable, not a baseline setting).

### 2. Calendar & benchmark
- Market is 24/7/365. No sessions, no overnight gaps, no open/close semantics.
- Annualization: hourly crypto = √8760 ≈ 93.6 (vs √1638 ≈ 40.5 for hourly stock bars). **Never reuse stock-calibrated Sharpe/alpha gate thresholds numerically.** G3/G5 thresholds are provisional until the first BTC baseline provides crypto-native distributions.
- G3 benchmark rewrites: QQQ → **buy-and-hold BTC**. Alpha vs B&H BTC is the promotion bar. This is deliberate — it is the gate-level catch for degenerate always-long policies, which look brilliant in every bull regime.
- Keep hour-of-day + day-of-week cyclical features (real intraday/weekly seasonality: Asia/US session flows, weekend liquidity thinning). Drop all market-session/open-close features.

### 3. Regime coverage
- Training/eval data **must** span at least one full drawdown cycle (2022-style leg) and one bull leg. A model trained on one regime is a bull-market artifact that G5 will not catch when all seeds saw the same regime.
- This is the crypto analog of the stock bot's "CV > 4.0 → rebuild parquet from 2015" failure pattern.

## Known Failure Patterns (crypto-native)

- **Trade rate 0% + G3 pass = degenerate always-flat.** Alpha ≈ −(benchmark return) with zero trades means the policy never entered; G3 passes spuriously in any net-down test window. A G3 pass is meaningless unless G6 > 0. Mirror image of the always-long trap — the gate stack catches it via G1/G2/G6, but never read G3 in isolation. `evaluate_sweep.py` now prints an explicit degenerate-policy diagnostic (always-flat/always-long rows, flagged separately from the 6-gate table) so this isn't missed on a quick read.
- **Cost double-counting.** `reward_turnover_penalty_scale` MUST be 0 in this repo — it now defaults to `0.0` in `src/experiments.py`. The turnover penalty was a proxy for costs the stock env didn't model; explicit fees now model them for real. Stacking both makes no-trade the learned optimum. Whipsaw control is min-hold's job, not the reward's.
- **Fee application invariant**, locked in by `tests/test_fee_model.py` (passing): a flat-forever episode incurs zero total fees; a single round trip incurs exactly 2× taker rate. Fees apply on position *change* only — never per bar held. Run this test before any sweep batch.
- **Position quantization (whole-share sizing) — the dominant cause of v1 AND v2's collapse.** Position sizing was ported from the stock bot as whole-share flooring (`int(net_worth * weight // price)`). `initial_balance` was never even exposed as a CLI flag and silently used the env class default of $1,000. Against this repo's BTC-USD backfill (~$16.5K–$126K), `floor(1000 / price)` is **0 for every single bar in the dataset** — the agent was structurally incapable of ever entering a position, regardless of reward shaping. Confirmed directly: btc-baseline-v2 (run after the reward fix below) still showed 0% trade rate on all 15 rows, identical to v1. **Fixed:** `PositionManager` now sizes fractionally (Coinbase supports ~1e-8 BTC precision); `--initial-balance` (default `10000.0`) and `--min-trade-notional` (default `1.0`, mirrors Coinbase's practical order minimum) are now real CLI flags in `src/experiments.py`. If a future sweep on any product shows 0% trade rate again, **check position sizing before reward shaping** — it has now been the actual cause twice.
- **Phantom rebalancing on hold.** A second bug surfaced while fixing the one above: `target_shares` was unconditionally re-derived from `net_worth` on *every* bar, even when the desired weight hadn't changed. Fees are additive on top of notional (not deducted from it), so a small nonzero cash residual is always left in `balance` after a trade; under fractional sizing, real price movement then drifted that residual in and out of a tradeable notional and fired dozens of economically meaningless "hold" trades per episode (whole-share flooring hid this by accident — too coarse to notice). **Fixed:** `PositionManager.step` now only re-derives position size from net worth when the weight target has actually changed; a genuine hold leaves `shares_held` untouched. Locked in by `tests/test_fee_model.py::test_holding_through_moving_price_does_not_drift_or_rebalance`.
- **Baseline invalidated by reward mis-specification or position-quantization** re-runs on the same template with an incremented version label. It is a correction, not the next batch, and does not trigger the 1h→4h horizon escalation — that rule applies only to an agent that trades and fails to clear fees.

## Process Flow (end to end)

```
DATA LAYER (new, this repo)
  REST candles backfill ─────┐
                             ├──► data/raw/<PRODUCT>.parquet  (OHLCV, 1h)
  WS ticker → bar_builder ───┘         │
                                       ▼
ENV LAYER (adapted)                CryptoTradingEnv
  fee model (per-flip taker)  ◄────  loads parquet
  24/7 calendar                      27-feature stationary obs
  B&H BTC benchmark                  + hour/day cyclical features
                                       │
                                       ▼
EXPERIMENT LAYER (carried over unchanged)
  experiments.py (sweep)
      → evaluate_sweep.py      (cross-seed 6-gate check — authoritative)
      → sanity_scan.py         (signal integrity)
      → generate_ensemble_config.py  (verify seed pins MANUALLY)
                                       │
                                       ▼
POST-PROMOTION
  walkforward confirmation → paper trading via portfolio_tracker
  (no live orders — View-only key until explicitly approved)
```

Portfolio tracking runs parallel to this flow, not inside it: WS ticker marks-to-market locally, REST `get_portfolio_breakdown()` reconciles on a slow interval, `user` channel fills trigger immediate reconciliation. Paginate `get_accounts()` to exhaustion.

## What Carries Over Unchanged

- `experiments.py` → `evaluate_sweep.py` → `sanity_scan.py` → `generate_ensemble_config.py` pipeline, in that order. `evaluate_sweep.py` cross-seed gates are authoritative; per-run gates are not sufficient.
- 6-gate promotion framework (with G3 benchmark swap and provisional G3/G5 thresholds per above).
- All sweep flag discipline: `--binary-actions`, `--min-hold-bars`, `--max-weight-delta-per-step 0.10`, `--use-stationary-features`, `--append`, ≥5 seeds, explicit `--n-envs`.
- Known gotchas: SubprocVecEnv FD leak (`--n-envs 1` fallback), legacy `sac_trading_bot_*.zip` naming (not a bug, do not rename), unreliable label filter in `generate_ensemble_config.py` (verify seed pins manually).
- One variable family per batch. Never >20 runs without written justification.

## What Is New (build order)

1. **Env fee model** — per-flip taker penalty, config-driven rates.
2. **24/7 calendar handling** — parameterize the existing env; do not fork it.
3. **Benchmark series swap** — B&H BTC replaces QQQ everywhere alpha is computed.
4. **Pagination-aware portfolio fetch** — `get_accounts()` returns `has_next`/`cursor`; always paginate to exhaustion or mark-to-market silently misses balances on later pages.

## First Experiment (locked)

Falsifiable question: *Does BTC-USD pass a fee-aware Binary PPO baseline at 1h bars with positive alpha vs buy-and-hold BTC?*

- Standard new-ticker template: 3 ent_coefs × 5 seeds = 15 runs.
- Fee-aware reward and 1h bars are **baked into the baseline config, not swept**.
- Batch 2 variable family (only after a valid, trading baseline clears fees): min-hold (6 vs 12). If a trading baseline still fails alpha at 1h, next variable is horizon (4h bars), **not hyperparameters** — fee economics dominate before entropy coefficients matter.

## Handoff Status (2026-07-26)

Full pipeline built and smoke-tested end to end: data layer, `CryptoTradingEnv`, feature engineering, `experiments.py` sweep runner, `evaluate_sweep.py`/`sanity_scan.py`/`generate_ensemble_config.py`, `portfolio_tracker.py`.

- **btc-baseline-v1 (INVALIDATED):** 15/15 runs, 0% trade rate on every run — degenerate always-flat. Reward-side cause identified at the time: `reward_turnover_penalty_scale` defaulted to `0.05` and stacked with the explicit fee model.
- **btc-baseline-v2 (INVALIDATED):** same template, `reward_turnover_penalty_scale` fixed to `0.0` — **still 15/15 runs, 0% trade rate, identical to v1.** The reward fix was necessary but not sufficient. Root cause confirmed by direct inspection of the v2 leaderboard: position sizing (see Known Failure Patterns → "Position quantization") had the agent structurally unable to afford even 1 BTC on the env's un-configurable $1,000 default balance against this repo's $16.5K–$126K price range. **This is now the confirmed primary cause for both v1 and v2** — do not assume a repeat 0% trade rate is reward-side without checking sizing first.
- **Fixes applied this session, round 1 (reward):** `reward_turnover_penalty_scale` defaults to `0.0` in `src/experiments.py`. `tests/test_fee_model.py` added, locking in the fee-application invariant. `pytest` added to `requirements.txt`. `evaluate_sweep.py` prints a degenerate-policy diagnostic (flags always-flat/always-long rows and any spurious G3 pass) and no longer suggests nudging turnover penalty in its "no champion" message. `--product` argparse no longer rejects uppercase/mixed-case product ids. `.claude/skills/crypto-experiment-strategist/SKILL.md` flag names corrected (`--interval`, `--transaction-cost-rate`) and its product-status line updated.
- **Fixes applied this session, round 2 (sizing, after v2 also collapsed):** `PositionManager` sizes fractionally instead of flooring to whole shares; `PositionManager.step` no longer re-derives position size from net worth on bars where the weight target hasn't changed (phantom-rebalancing fix); `--initial-balance` (new flag, default `10000.0`) and `--min-trade-notional` (new flag, default `1.0`) added to `src/experiments.py` and wired into `env_kwargs`. `tests/test_fee_model.py` expanded to 5 tests, including two regression tests pinning each bug down directly (`test_fractional_sizing_allows_entry_when_balance_is_below_asset_price`, `test_holding_through_moving_price_does_not_drift_or_rebalance`). All passing. Verified directly against the real BTC-USD parquet (not just synthetic data) that a $1,000–$10,000 balance can now enter and hold a position through real price movement with exactly one fee event.
- **btc-baseline-v3 (READY, NOT YET RUN):** same template, both fixes in place. Run command:

  ```powershell
  .\.venv\Scripts\python.exe -m src.experiments --product BTC-USD --interval 1h --binary-actions --min-hold-bars 6 --reward-mode sharpe --transaction-cost-rate 0.006 --initial-balance 10000 --min-trade-notional 1.0 --ent-coefs 0.01,0.02,0.05 --timesteps 40000 --seeds 3,7,13,21,42 --execution-mode next_bar --reward-turnover-penalty-scale 0.0 --max-weight-delta-per-step 0.10 --use-stationary-features --n-envs 1 --run-label "btc-baseline-v3" --append
  ```

  Then `python scripts\evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv --label btc-baseline-v3`. v1/v2 rows are **not comparable** to v3 (different position sizing, different `initial_balance`) — never rank them together.
- **Next-session entry point:** if v3 hasn't run yet, run it. If it has, read the degenerate-policy diagnostic first — if trade rate is still 0%, do not default back to reward-side diagnosis; re-verify sizing (`tests/test_fee_model.py`) and the actual `initial_balance`/`min_trade_notional` values recorded in the leaderboard row before looking anywhere else. Otherwise branch per the decision rules in Known Failure Patterns / the `crypto-experiment-strategist` skill.

## Security Posture

- CDP API key is **View-only**. Do not add Trade/Transfer scope until live execution is explicitly approved — never preemptively.
- Credentials in `.env` only (`CDP_API_KEY`, `CDP_API_SECRET`). Never hardcoded, never committed, never echoed into logs or tool output.
- `.gitignore` covers `.env`, `.venv/`, `data/raw/*.parquet`.
- IP allowlist intentionally opted out (dynamic residential IP). Revisit if deployed to Azure Container Apps (stable outbound IP).
- No live orders of any kind in this repo until paper-trading validation through the pipeline is complete.

## Environment

- Windows / PowerShell. Venv: `.\.venv\Scripts\Activate.ps1`. Always invoke as `.\.venv\Scripts\python.exe`, never bare `python`.
- Stack: `coinbase-advanced-py`, `gymnasium`, `stable-baselines3`, `pandas`, `pyarrow`, `python-dotenv`.
- Layout: `src/` (clients, bar_builder, portfolio_tracker, `env/`), `data/raw/` (parquet), project root `CLAUDE.md` (this file).

## Working Conventions

- Validate approach before writing code. Explain non-obvious concepts inline as built.
- `scaffold` keyword = architecture overview + key/non-obvious scripts + run commands only. No full file trees, no packaged downloads.
- Docs concise, standard Markdown. Diagrams: visual flowchart-style, never Mermaid.
- Security-by-default on everything.
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
- **Majority-seed degenerate collapse survives the sizing fix.** btc-baseline-v3 (post sizing+phantom-rebalance fix, see Handoff Status) still shows 13/15 rows collapsing to always-flat or always-long; only 2/15 rows trade in a normal range, and those fail alpha/accuracy/win-rate anyway. `btc-minhold-12` and `btc-4h-baseline-v1` — which change min-hold and bar interval respectively, i.e. different levers — show the **same majority-degenerate pattern** (13/15 and 14/15 collapsed). Since changing hold length or horizon didn't move the collapse rate, the lever is probably neither of those. **Diagnosed 2026-07-27** — resolved to "no extractable edge in the current feature set," not an exploration/shaping pathology. See the signal-probe finding below and [HANDOFF.md](HANDOFF.md) §5 Steps 3–4.
- **Reward-shaping defaults audit (2026-07-27):** `reward_hold_penalty_scale` and `reward_drawdown_penalty_scale` both default to `0.10` and were left at that default in v3/A/B/all three (confirmed by leaderboard inspection — none of the three commands overrode them). Unlike `reward_turnover_penalty_scale`, `--initial-balance`, and `--min-trade-notional`, these two flags (plus `reward_action_bonus_scale=0.02`, `reward_direction_scale=0.35`) carry no documented crypto-specific rationale in their `argparse` help text — one-line descriptions only, no note on why these values or whether they were re-derived for crypto vs carried over. Not confirmed as the cause of the collapse above; superseded as the leading hypothesis by the signal-probe finding below (features > shaping) but still flagged as unexamined for whenever a features batch reopens the reward-shaping question.
- **No extractable BTC edge confirmed in the current stationary feature set at the fee floor (2026-07-27).** `scripts/signal_probe.py` fit LogisticRegression + GradientBoostingClassifier directly on the env's 18 stationary market features (same walk-forward split, forward 1-bar and 6-bar log-return sign as target) — independent of PPO/reward mechanics entirely. Result: AUC 0.52–0.55, weak but real (replicates across both model families and both val/test windows, well under the 0.60 leakage-suspicion line). A threshold×min-hold sweep (thresholds 0.55–0.75, min_hold ∈ {1,6}, includes this repo's actual `min_hold_bars=6`) found **zero operating points that are both fee-positive at a non-trivial trade count (≥20 flips) and agree in sign across val and test** — the only candidate with a meaningful sample size (~70 flips, positive on test) was negative on validation at the identical threshold/min-hold, i.e. did not replicate. Conclusion: the corner-collapse pattern above is consistent with PPO correctly finding there's nothing to trade at this fee level with these features, not failing to explore. **Next variable family is features, not entropy/reward-shaping** — see NEXT_EXPERIMENTS.md Branch E. Re-run `scripts/signal_probe.py` against any new feature set before committing a full PPO sweep to it; it's near-free relative to a training batch.
- **Branch E Phase E0 (vol term structure + volume weekly seasonality) also found no confirmed edge (2026-07-27).** Added `VolRatio_6_48` and `RelVolumeByHourOfWeek` (`src/feature_engineering.py`, wired through `src/market_data.py`; both env and probe now import a single shared `STATIONARY_FEATURE_COLUMNS` constant instead of two hand-duplicated lists — fix which itself closes a duplication risk this session introduced). Re-probed: same AUC range, no change. One sweep cell passed the split-agreement bar but failed a **cross-model** check — GradientBoosting was negative at the identical (horizon, threshold, min_hold) where LogisticRegression was positive, and the cell was an isolated spike between negative neighbors, not a plateau — a textbook artifact of a 40-cell threshold sweep, not real structure. `find_confirmed_edge` in `scripts/signal_probe.py` now requires agreement across both model families as well as both splits before calling anything confirmed; this is the third time this session a naive positive read on this probe turned out not to survive a stricter check (n_flips minimum, then val/test agreement, now cross-model agreement) — **read the sweep table itself, not just the printed verdict line, before trusting a "signal may exist" result.**
- **Branch E Phase E1 (ETH/BTC cross-rate) also found no confirmed edge (2026-07-27) — Branch E is now exhausted.** `data/raw/ETH-USD.parquet` backfilled (57,586 rows). `EthBtcRelReturn` added via `src/market_data.py::compute_cross_rate_feature` (`CROSS_RATE_SECONDARY_PRODUCT = {"BTC-USD": "ETH-USD"}` — returns an all-zero/neutral array for any primary product without a configured secondary, so it degrades gracefully rather than needing a branch at every call site). Sanity-checked clean. Probe result: AUC 0.55 max, zero cells clear the same three-layer bar (min flips, val/test agreement, cross-model agreement) as E0. **Both candidate feature families (free and data-cost) are now exhausted with the same null result.** Per NEXT_EXPERIMENTS.md Branch E's own failure interpretation, this is no longer a quick-batch decision — do not keep adding feature families ad hoc. Two escalation paths on the table, presented to the user rather than decided unilaterally: (a) maker-fee execution modeling (test whether the 0.60% taker fee floor itself exceeds extractable edge — Branch B was meant to test this but never did, since it collapsed degenerate instead of trading-and-losing), or (b) reconsider as a scope question whether 1h/4h BTC-USD Binary PPO can clear fees with any readily available feature set at all.
- **Per-run provenance gap.** `n_envs`, `initial_balance`, and `min_trade_notional` are used at runtime (`env_kwargs`, `SubprocVecEnv` setup in `src/experiments.py`) but are never written to a leaderboard column, and no run-log/manifest file exists anywhere in the repo. Once a run completes, these three values are unrecoverable unless the invocation was recorded in CLAUDE.md by hand (as v3's is). This is why `btc-minhold-12`/`btc-4h-baseline-v1`'s exact `n_envs` is unknown despite both having full leaderboard rows — record the full command in this file at run time, every time.

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

## Handoff Status (2026-07-27)

Full pipeline built and smoke-tested end to end: data layer, `CryptoTradingEnv`, feature engineering, `experiments.py` sweep runner, `evaluate_sweep.py`/`sanity_scan.py`/`generate_ensemble_config.py`, `portfolio_tracker.py`.

- **btc-baseline-v1 (INVALIDATED):** 15/15 runs, 0% trade rate on every run — degenerate always-flat. Reward-side cause identified at the time: `reward_turnover_penalty_scale` defaulted to `0.05` and stacked with the explicit fee model.
- **btc-baseline-v2 (INVALIDATED):** same template, `reward_turnover_penalty_scale` fixed to `0.0` — **still 15/15 runs, 0% trade rate, identical to v1.** The reward fix was necessary but not sufficient. Root cause confirmed by direct inspection of the v2 leaderboard: position sizing (see Known Failure Patterns → "Position quantization") had the agent structurally unable to afford even 1 BTC on the env's un-configurable $1,000 default balance against this repo's $16.5K–$126K price range. **This is now the confirmed primary cause for both v1 and v2** — do not assume a repeat 0% trade rate is reward-side without checking sizing first.
- **Fixes applied 2026-07-26, round 1 (reward):** `reward_turnover_penalty_scale` defaults to `0.0` in `src/experiments.py`. `tests/test_fee_model.py` added, locking in the fee-application invariant. `pytest` added to `requirements.txt`. `evaluate_sweep.py` prints a degenerate-policy diagnostic (flags always-flat/always-long rows and any spurious G3 pass) and no longer suggests nudging turnover penalty in its "no champion" message. `--product` argparse no longer rejects uppercase/mixed-case product ids. `.claude/skills/crypto-experiment-strategist/SKILL.md` flag names corrected (`--interval`, `--transaction-cost-rate`) and its product-status line updated.
- **Fixes applied 2026-07-26, round 2 (sizing, after v2 also collapsed):** `PositionManager` sizes fractionally instead of flooring to whole shares; `PositionManager.step` no longer re-derives position size from net worth on bars where the weight target hasn't changed (phantom-rebalancing fix); `--initial-balance` (new flag, default `10000.0`) and `--min-trade-notional` (new flag, default `1.0`) added to `src/experiments.py` and wired into `env_kwargs`. `tests/test_fee_model.py` expanded to 5 tests, including two regression tests pinning each bug down directly (`test_fractional_sizing_allows_entry_when_balance_is_below_asset_price`, `test_holding_through_moving_price_does_not_drift_or_rebalance`). All passing. Verified directly against the real BTC-USD parquet (not just synthetic data) that a $1,000–$10,000 balance can now enter and hold a position through real price movement with exactly one fee event.
- **Ordering anomaly:** `btc-minhold-12` (NEXT_EXPERIMENTS.md Branch A) and `btc-4h-baseline-v1` (Branch B) both ran and were committed to the repo **before** btc-baseline-v3 itself ran — out of the documented sequence, which requires reading v3's diagnostic before choosing a branch, and running only one. Cause not established (likely a prior session got ahead of the written plan). Flagging so this isn't mistaken for a clean sequential handoff — treat all three results below as concurrent inputs to diagnose together, not as a linear A/B/C decision tree that was followed correctly.
- **btc-baseline-v3 (RAN 2026-07-26T07:10–07:15Z):** both fixes in place, template per the command below. **Result: no champion, 0/15 rows pass all 6 gates.** Degenerate-policy diagnostic: 9/15 rows always-flat, 4/15 always-long — 13/15 collapsed. Only 2/15 rows traded in a normal range, and both failed anyway (test alpha ≈ −0.02 to −0.04, actionable accuracy ≈ 0.49 < 0.53 threshold, trade win rate ≈ 0.49 < 0.52 threshold). The sizing fix worked — the non-degenerate rows show real, economically coherent trading activity — but it did not solve the collapse; most seed/ent_coef configs still degenerate.

  ```powershell
  .\.venv\Scripts\python.exe -m src.experiments --product BTC-USD --interval 1h --binary-actions --min-hold-bars 6 --reward-mode sharpe --transaction-cost-rate 0.006 --initial-balance 10000 --min-trade-notional 1.0 --ent-coefs 0.01,0.02,0.05 --timesteps 40000 --seeds 3,7,13,21,42 --execution-mode next_bar --reward-turnover-penalty-scale 0.0 --max-weight-delta-per-step 0.10 --use-stationary-features --n-envs 1 --run-label "btc-baseline-v3" --append
  ```

  v1/v2 rows are **not comparable** to v3 (different position sizing, different `initial_balance`) — never rank them together.
- **btc-minhold-12 (Branch A, RAN):** min-hold 12 vs v3's 6. **No champion.** Same collapse pattern, inverted skew: 4/15 always-flat, 9/15 always-long, 2/15 trade normally and fail alpha. Doubling min-hold did not reduce the degenerate rate — if anything it shifted mass toward always-long instead of always-flat.
- **btc-4h-baseline-v1 (Branch B, RAN):** 4h bars, min-hold 3. **No champion.** 10/15 always-flat, 4/15 always-long, 1/15 trades normally and fails alpha. Changing horizon did not reduce the degenerate rate either.
- **Reading across all three:** changing hold length (A) and changing horizon (B) each targeted a different lever than v3, and neither moved the degenerate-collapse rate. That's evidence the collapse isn't caused by min-hold or bar interval — see Known Failure Patterns → "Majority-seed degenerate collapse survives the sizing fix." Most likely remaining suspects: entropy coefficient / exploration, or reward shaping (hold penalty, drawdown penalty) pushing the policy into a corner solution. Not yet isolated as its own experiment.
- **btc-minhold-12 / btc-4h-baseline-v1 recorded provenance (2026-07-27 audit):** reward-shaping params ARE recoverable from the leaderboard and are **identical across v3/A/B, all at `src/experiments.py`'s argparse defaults** — `reward_hold_penalty_scale=0.10`, `reward_drawdown_penalty_scale=0.10`, `reward_action_bonus_scale=0.02`, `reward_direction_scale=0.35`, `reward_return_scale=1.0`, `reward_pnl_scale=0.0`, `reward_turnover_penalty_scale=0.0`, `reward_clip=1.0`. None of the three commands overrode any reward-shaping flag. **`n_envs`, `initial_balance`, and `min_trade_notional` are NOT recoverable** — they're used in `env_kwargs`/training setup but never written to a leaderboard column, and no run-log/manifest file exists anywhere in the repo. This is a real provenance gap, not just an A/B-specific one: any future run's exact `n_envs`/`initial_balance`/`min_trade_notional` is unknowable after the fact unless the invocation itself is recorded in CLAUDE.md at run time (as v3's is, above). Consider adding these as logged leaderboard columns.
- **`--n-envs` correction:** a prior draft of this handoff asserted `--n-envs 4` was "empirically verified clean" and should be locked in as the repo default. That claim doesn't check out against anything in this repo: the argparse default is `--n-envs 8` (`src/experiments.py`), v3's documented command explicitly used `--n-envs 1`, and there is no test, log, or artifact anywhere showing a 4-envs run was ever executed here. Do not treat n_envs=4 as verified or locked without an actual test backing it up — the existing gotcha (`--n-envs 1` fallback on `Errno 24`) stands unchanged.
- **Signal probe result (2026-07-27) — Steps 1–4 of HANDOFF.md §5 all complete.** `scripts/signal_probe.py` found no confirmed extractable edge in the current feature set at the fee floor (AUC 0.52–0.55, weak but real; zero threshold×min-hold operating points both fee-positive at a non-trivial trade count and replicating in sign across val/test — see Known Failure Patterns above for full detail). **Branch D (entropy/reward-shaping) is cancelled**, not just gated — do not run it against the current feature set. Do not re-run A (min-hold) or B (horizon) either; both are ruled out as levers.
- **Next-session entry point (2026-07-27): Branch E is exhausted — both phases ran, neither found a confirmed edge.** E0 (vol term structure + volume seasonality) and E1 (ETH/BTC cross-rate, `data/raw/ETH-USD.parquet` now backfilled) both probed clean against the three-layer confirmed-edge bar (min flips, val/test agreement, cross-model agreement) and both came back empty. Do not add a Phase E2 feature ad hoc. Next decision is a scope choice between (a) maker-fee execution modeling and (b) reconsidering whether 1h/4h BTC-USD Binary PPO can clear fees with any readily available feature set — presented to the user, not decided here. See Known Failure Patterns above and NEXT_EXPERIMENTS.md Branch E for full detail.

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
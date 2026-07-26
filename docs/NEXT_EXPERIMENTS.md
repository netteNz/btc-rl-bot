# NEXT_EXPERIMENTS.md — btc post-v3 program

Branch on v3's degenerate-policy diagnostic. Run ONE branch. Never two concurrently.
v1/v2 rank against nothing. Read trade rate before any other gate.

## Constants (all branches)
`--binary-actions --transaction-cost-rate 0.006 --initial-balance 10000 --min-trade-notional 1.0 --reward-turnover-penalty-scale 0.0 --max-weight-delta-per-step 0.10 --use-stationary-features --seeds 3,7,13,21,42 --n-envs 4 --reward-mode sharpe --execution-mode next_bar --append`

## Branch 0 — v3 trade rate 0%
NO sweep. Diagnose: (1) `pytest tests/test_fee_model.py` (2) verify initial_balance/min_trade_notional in leaderboard rows (3) reward decomposition on forced-trade rollout. Sizing before reward — it's been the cause twice.

## Branch A — v3 trades, passes/near-passes → min-hold isolation
v3 rows ARE the min-hold-6 control (no rerun). New arm: min-hold 12. 15 runs. Confirmatory, medium impact (rankable vs v3, note min-hold).
Q: does halving flip frequency improve fee-adjusted alpha, or lag exits enough to give it back?
```powershell
.\.venv\Scripts\python.exe -m src.experiments --product BTC-USD --interval 1h --min-hold-bars 12 --ent-coefs 0.01,0.02,0.05 --timesteps 40000 --run-label "btc-minhold-12" [constants]
python scripts\evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv --label btc-minhold-12
```

## Branch B — v3 trades, fails alpha at 1h → 4h horizon (locked rule: horizon, not hyperparams)
PREREQ 1: resample 1h parquet → 4h (deterministic OHLCV agg; no re-backfill).
PREREQ 2: verify eval annualization switches to √2190 for 4h before reading Sharpe.
GUARD: if v3 alpha fails AND actionable accuracy ≈ 0, problem is signal not horizon — stop, do not launch B.
15 runs. Exploratory, HIGH impact — new comparison universe, never rank vs 1h rows.
```powershell
.\.venv\Scripts\python.exe -m src.experiments --product BTC-USD --interval 4h --min-hold-bars 3 --ent-coefs 0.01,0.02,0.05 --timesteps 40000 --run-label "btc-4h-baseline-v1" [constants]
python scripts\evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv --label btc-4h-baseline-v1
```

## Branch C — v3 trades, in-band, alpha fails, 2–3/5 seeds collapsed → timesteps 60k
Champion ent_coef from v3 only. 5 runs. Confirmatory, low impact. Label `btc-60k-<entcoef>`.
Seeds still collapse at 60k → env-fit diagnosis, not more compute.

## Success criteria (all)
Trade rate > 0 first. Then: alpha ≥ 0 vs B&H BTC · accuracy ≥ 0.53 · drift ≤ 0.05 · CV < 1.0 over active seeds (≥5 seeds).
G6 band 0.40–0.80 is PROVISIONAL (stock-calibrated): record distribution; 30–40% trade rate + positive alpha = calibration question, not auto-kill.

## Failure interpretation
A fails both directions → hold length isn't the lever → B.
B fails with trading agent at 4h → fee floor may exceed extractable edge at retail tier. Finding, not bug. Next family = maker-fee execution modeling = NEW proposal, stop here.
C seeds collapse at 60k → env-fit diagnosis.
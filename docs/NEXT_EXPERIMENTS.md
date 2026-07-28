# NEXT_EXPERIMENTS.md — btc post-v3 program

**Superseded by [HANDOFF.md](../HANDOFF.md) §4–5 for next steps** (Branches C and D below). This file
stays as the results log for Branches 0/A/B; read HANDOFF.md for the current diagnosis fork and
execution sequence.

Status (2026-07-27): Branches 0, A, and B have all run. All three show no champion, dominated by
degenerate-policy collapse (always-flat/always-long). **Ordering anomaly:** A and B were run and
committed before v3 itself ran, out of the sequence this doc originally prescribed ("branch on v3's
diagnostic, run ONE branch"). Cause not established. Treat the three results below as concurrent
inputs to one diagnosis, not a linear decision tree that was followed correctly. Full detail in
CLAUDE.md → Handoff Status (2026-07-27).

**`n_envs` note:** not recoverable for any historical run — never logged as a leaderboard column, no
run-manifest exists. The argparse default is `--n-envs 8`; v3's documented command used `--n-envs 1`;
A/B's actual value is unknown. Do not assume A/B ran at the same `n_envs` as v3 or as each other.

v1/v2 rank against nothing. v3/A/B are cross-comparable (same sizing fix, same fee/reward config).

## Constants (all branches)
`--binary-actions --transaction-cost-rate 0.006 --initial-balance 10000 --min-trade-notional 1.0 --reward-turnover-penalty-scale 0.0 --max-weight-delta-per-step 0.10 --use-stationary-features --seeds 3,7,13,21,42 --n-envs 1 --reward-mode sharpe --execution-mode next_bar --append`

`--n-envs 1` is the only value with a documented working run (v3). Do not substitute a higher value
without recording the actual invocation in CLAUDE.md — see the `n_envs` note above.

## Branch 0 — btc-baseline-v3 (RAN, min-hold 6, 1h) — COMPLETE, no champion
9/15 rows always-flat, 4/15 always-long, 2/15 trade normally and fail alpha (≈ −0.02 to −0.04),
accuracy (≈0.49 < 0.53), win rate (≈0.49 < 0.52). Sizing fix confirmed working (non-degenerate rows
show coherent trading) but does not prevent collapse in most configs.

## Branch A — btc-minhold-12 (RAN, min-hold 12, 1h) — COMPLETE, no champion
v3 rows are the min-hold-6 control. 4/15 always-flat, 9/15 always-long, 2/15 trade normally and fail
alpha. Doubling min-hold shifted the degenerate skew toward always-long but did not reduce the
degenerate rate. **Finding: min-hold is not the lever.**

## Branch B — btc-4h-baseline-v1 (RAN, 4h bars, min-hold 3) — COMPLETE, no champion
10/15 always-flat, 4/15 always-long, 1/15 trades normally and fails alpha. Changing horizon did not
reduce the degenerate rate either. **Finding: horizon is not the lever** — do not read this as the
"fee floor exceeds extractable edge" case this branch was originally designed to catch, because the
agent mostly never traded rather than trading and losing to fees.

## Branch C — timesteps 60k — NOT RECOMMENDED as originally scoped
Original precondition was "2–3/5 seeds collapsed." Actual result across Branches 0/A/B is majority
collapse (9–10/15 rows), a more severe failure mode. More compute is unlikely to fix a corner-solution
collapse; diagnose the cause first (see below) rather than running this branch on the old rationale.

## Branch D — entropy / reward-shaping isolation — CANCELLED 2026-07-27
Was gated on the HANDOFF.md §5 Step 3 signal probe showing AUC materially > 0.52 with a confirmed,
fee-surviving edge. **Probe ran 2026-07-27** (`scripts/signal_probe.py`, LogisticRegression +
GradientBoostingClassifier, same features/splits as the env, horizons 1 and 6): AUC came in at
0.52–0.55 — real (replicates across 2 model families and both val/test windows) but weak. A
threshold×min-hold sweep (thresholds 0.55–0.75, min_hold ∈ {1,6}, includes this repo's actual
`min_hold_bars=6` policy) found **zero operating points that are both fee-positive with a non-trivial
trade count (≥20 flips) and agree in sign across val and test** — the one candidate with a meaningful
sample (~70 flips, positive on test) was negative on validation at the identical threshold/min-hold.
**Conclusion: no confirmed extractable edge in the current feature set at the fee floor.**
Reward-shaping/entropy tuning can force a policy off the flat/long corners, but cannot manufacture
edge that isn't there — Branch D stays cancelled until the observation space changes. See Branch E.

Original rationale (kept for reference, superseded by the probe result): A (min-hold) and B (horizon)
each changed a different lever from v3's baseline and neither moved the degenerate-collapse rate.
That ruled out hold length and bar interval as the cause and pointed at whatever v3/A/B held in
common instead — `ent_coef` range or reward shaping (`reward_hold_penalty_scale`,
`reward_drawdown_penalty_scale`, both at 0.10, both undocumented for crypto-specific rationale). The
probe result above supersedes this: the shared cause is more likely an observation space with no
fee-clearing edge, not shaping/entropy.

## Branch E — feature family
Per HANDOFF.md §5 Step 4 resolution: next variable family is **features**, not entropy/reward-shaping.
New observation space = new comparison universe (HIGH impact). Designed via the
`crypto-experiment-strategist` skill as two sequential, gated phases. Funding-proxy basis dropped from
candidates — no compatible data source in this repo's spot-only Coinbase Advanced Trade architecture.

**Phase E0 — realized-vol term structure + volume weekly seasonality. RAN 2026-07-27. No confirmed
edge.** Implemented `VolRatio_6_48` (short/long realized-vol ratio) and `RelVolumeByHourOfWeek`
(volume relative to trailing historical average for the same day-of-week/hour-of-day bucket,
strictly trailing — no leakage) in `src/feature_engineering.py`, wired into
`src/market_data.py::get_crypto_training_data` and the shared `STATIONARY_FEATURE_COLUMNS` constant
(now imported by both `src/env/trading_env.py` and `scripts/signal_probe.py` instead of two
hand-duplicated lists). Re-ran the signal probe: AUC 0.51–0.55, same range as before. One sweep cell
(`horizon=6, LogisticRegression, threshold=0.60, min_hold=6`) initially passed the ≥20-flips +
val/test-sign-agreement bar — but `GradientBoosting` was **negative** at that identical operating
point, and the cell was an isolated spike (negative neighbors at threshold 0.55 and 0.65 on both
splits), not a plateau. `find_confirmed_edge` was tightened to require agreement across **both model
families**, not just both splits — with that bar, zero cells clear. **E0 does not change the
conclusion: no confirmed extractable edge.**

**Phase E1 — ETH/BTC cross-rate returns. RAN 2026-07-27. No confirmed edge.** `ETH-USD` backfilled
(57,586 rows, gap ratio well under the 2% warning threshold). Implemented `EthBtcRelReturn` (ETH-USD
log return − BTC-USD log return, aligned on Date with trailing ffill across any ETH gaps) in
`src/market_data.py::compute_cross_rate_feature`, wired into `get_crypto_training_data` and the shared
`STATIONARY_FEATURE_COLUMNS` constant. Sanity-checked clean (no NaNs, real variation, std ≈ 0.0038).
Re-ran the probe: AUC 0.55 max, same range as E0 and the original baseline. **Zero cells clear the
confirmed-edge bar** (≥20 flips, val+test sign agreement, both model families agreeing) — same result
as E0.

**Branch E conclusion (2026-07-27): both candidate feature families exhausted, neither found a
confirmed extractable edge.** Per the design's own failure interpretation, this stops being a
quick-batch decision. Do not add more feature families ad hoc. Escalate to one of:
(a) maker-fee execution modeling — CLAUDE.md already flags the 0.60% taker assumption as "a future
sweep variable, not a baseline setting"; the fee floor itself may exceed extractable edge at the
retail taker tier, which Branch B was originally meant to test but never actually did (it collapsed
degenerate instead of trading-and-losing); or
(b) reconsidering whether 1h/4h BTC-USD Binary PPO can clear fees with any readily available feature
set at all — a scope question for CLAUDE.md, not a batch design question.
See HANDOFF.md for the full state and this decision point.

## Success criteria (all)
Trade rate > 0 first. Then: alpha ≥ 0 vs B&H BTC · accuracy ≥ 0.53 · drift ≤ 0.05 · CV < 1.0 over active seeds (≥5 seeds).
G6 band 0.40–0.80 is PROVISIONAL (stock-calibrated): record distribution; 30–40% trade rate + positive alpha = calibration question, not auto-kill.

## Failure interpretation (superseded by 2026-07-27 results above, kept for reference)
A fails both directions → hold length isn't the lever → B. (Confirmed: A failed both directions.)
B fails with trading agent at 4h → fee floor may exceed extractable edge at retail tier. (Did not
apply — B's failure mode was degenerate collapse, not a trading agent losing to fees.)
C seeds collapse at 60k → env-fit diagnosis. (Superseded — see Branch C note above.)

# HANDOFF.md — coinbase-rl-bot (2026-07-27)

Read this before CLAUDE.md's Handoff Status. This supersedes NEXT_EXPERIMENTS.md's decision tree and the CLAUDE.md next-session entry point. Apply the doc amendments in §6 as the first commit of the session.

## 1. State in one paragraph

Pipeline is built, tested, and mechanically sound: fractional sizing, fee invariants, phantom-rebalance fix all locked by `tests/test_fee_model.py` (5 passing). Five batches have run (v1, v2, v3, minhold-12, 4h-baseline-v1), zero champions. v1/v2 were structural failures (whole-share sizing, $1K default balance — agent could not trade; rank against nothing). v3/A/B are real results: **majority degenerate collapse (13–14 of 15 rows each) into always-flat or always-long corners, and every row that did trade (5 of 45) sits at ~0.49 accuracy / ~0.49 win rate / negative alpha.** Min-hold (A) and horizon (B) are ruled out as levers. The open question was no longer "is the env broken" — it was "is there any extractable signal in the current feature set at all." **§5 Step 3 (2026-07-27) answered that: no.** A supervised probe (LogisticRegression + GradientBoostingClassifier, same features/splits as the env, threshold×min-hold sweep) found AUC 0.52–0.55 — real but weak, and **zero operating points cleared fees with a non-trivial trade count that also replicated in sign across val and test.** Entropy/reward-shaping is off the table until the feature set changes. See §5 Steps 3–4 below.

## 2. Batch ledger

| Label | Config delta | Result | Standing |
|---|---|---|---|
| btc-baseline-v1 | mh6, 1h | 15/15 no-trade | invalid (sizing bug) — rank vs nothing |
| btc-baseline-v2 | + turnover 0.0 | 15/15 no-trade | invalid (sizing bug) — rank vs nothing |
| btc-baseline-v3 | + sizing fix | 9 flat / 4 long / 2 trade-and-fail | valid; the baseline |
| btc-minhold-12 (A) | mh12 | 4 flat / 9 long / 2 trade-and-fail | valid; min-hold ≠ lever |
| btc-4h-baseline-v1 (B) | 4h, mh3 | 10 flat / 4 long / 1 trade-and-fail | valid; horizon ≠ lever |

Caveat on A/B: they ran before v3 (ordering anomaly, cause unestablished) and possibly at `--n-envs 4` vs v3's recorded 1. Coarse finding (majority collapse) is robust across that; fine-grained skew comparisons (flat↔long shifts between batches) are two-variable and should not be over-read.

## 3. Resolved this session

- **n-envs / FD leak:** empirically clean at `--n-envs 4` on this Windows setup (spawn-based multiprocessing; the leak gotcha was stock-repo/platform history). **`--n-envs 4` is now the locked repo value** — record it in every command. Remaining task is provenance only (§5 step 1), not safety.
- **Sizing, fees, rebalancing:** settled, tested, closed. If any future batch shows 0% trade rate: check sizing first (twice the actual cause), reward second.

## 4. The diagnosis fork (do NOT skip to a sweep)

All 5 trading rows at coin-flip accuracy means the corner collapse may be the *correct* policy for a signal-free observation space under a 1.2% round-trip fee — not an exploration pathology. Entropy/shaping sweeps (proposed Branch D) can force trading; they cannot create edge. Therefore Branch D is **gated** on the signal probe below. Branch C (60k timesteps) stays cancelled — majority collapse is not the partial-seed-collapse case it was scoped for.

**Resolved 2026-07-27:** the signal probe (§5 Step 3) confirms the corner collapse is consistent with a signal-free-at-current-fees observation space, not an exploration failure. Branch D stays cancelled. Do not revisit it until a features batch changes the observation space.

## 5. Execution sequence (in order; steps 1–3 cost less than one training run)

**Step 1 — A/B provenance. RAN 2026-07-27, partially closed.** Reward-shaping params ARE recoverable from the leaderboard and are identical across v3/A/B, all at argparse defaults (recorded in CLAUDE.md Handoff Status). `n_envs`/`initial_balance`/`min_trade_notional` are **not** recoverable — never logged as leaderboard columns, no run-manifest exists anywhere in the repo. This is a real, ongoing provenance gap (see CLAUDE.md Known Failure Patterns), not fully closable after the fact. Also corrected: the `--n-envs 4` "verified clean, locked" claim in §3 above does not check out against anything in this repo (argparse default is 8, v3 used 1, no test of 4 exists here) — do not treat it as settled.

**Step 2 — reward-shaping defaults audit. RAN 2026-07-27.** All `reward_*` defaults in `src/experiments.py` were confirmed against the leaderboard: `reward_hold_penalty_scale=0.10` and `reward_drawdown_penalty_scale=0.10` (not 0.01 as this section originally assumed — that number doesn't match anything in this repo), `reward_action_bonus_scale=0.02`, `reward_direction_scale=0.35`, all left at default in v3/A/B. These four carry no documented crypto-specific rationale in their help text (unlike `reward_turnover_penalty_scale`/`--initial-balance`/`--min-trade-notional`, which do). Flagged as unexamined in CLAUDE.md Known Failure Patterns — candidate for a future isolation batch, but not the immediate next step (see Step 4 resolution below).

**Step 3 — signal probe (`scripts/signal_probe.py`) — RAN 2026-07-27.**
- Inputs: same BTC-USD 1h parquet, same stationary feature pipeline (18 columns — 14 indicators + 4 cyclical; CLAUDE.md's "27-feature" figure appears to include env account-state features not part of the market observation, not reconciled further here), same walk-forward split ratios as `src/experiments.py`.
- Targets: sign of forward 1-bar and forward 6-bar log return, one script invocation covering both via `--horizons 1,6`.
- Models: `LogisticRegression` (with `StandardScaler`) and `sklearn.ensemble.GradientBoostingClassifier`. `scikit-learn` was **not** already installed — added to `requirements.txt`, correcting this section's original "no new heavy deps" assumption.
- **Result: AUC 0.52–0.55 across both horizons/both models/both splits** — real (replicates across two independent model families and two independent chronological windows, well under the 0.60 leakage-suspicion line) but weak.
- Original design used a single naive `prob>0.55` no-min-hold rule and found fee-adjusted expectancy negative everywhere. Extended with a threshold×min-hold sweep (thresholds 0.55–0.75, min_hold ∈ {1, 6}) to give the signal a fair shot at the trade frequency this repo's actual policies use (`min_hold_bars=6`). **Still no confirmed edge:** the only fee-positive cells have single-digit-to-low-double-digit flip counts (noise from a handful of trades, not a sample), and the one cell with a non-trivial trade count (~70 flips) that was positive on test was **negative on validation at the identical threshold/min-hold** — does not replicate out-of-sample. Verdict logic now requires ≥20 flips per split AND val/test sign agreement at the same operating point before calling anything a candidate edge; zero cells clear that bar.
- No shuffling anywhere; strictly chronological.

**Step 4 — branch on probe result. RESOLVED 2026-07-27: no confirmed edge → features branch.**
- ~~AUC ≤ ~0.52 (both horizons)~~ — actual outcome was AUC modestly *above* 0.52 but with **no operating point that is both fee-positive at a meaningful trade count and replicates in sign across val/test.** Functionally the same conclusion as the "no edge" branch: feature set has no confirmed extractable BTC edge at the fee floor. Next variable family = **features** (candidates: ETH/BTC cross-rate returns, realized-vol term structure, volume/liquidity weekly seasonality, funding-proxy basis). Entropy/reward-shaping sweep (Branch D) stays cancelled — see NEXT_EXPERIMENTS.md. Design the feature batch via `crypto-experiment-strategist` skill — it is a HIGH-impact change (new observation space = new comparison universe).
- (AUC materially > 0.52 with a confirmed fee-positive, replicating operating point did not occur — that branch is moot for this session.)

## 6. Doc amendments to apply (first commit)

1. CLAUDE.md next-session entry point → replace with pointer to this file's §5.
2. CLAUDE.md gotcha line: "SubprocVecEnv FD leak (--n-envs 1 fallback)" → "FD-leak was stock-repo/platform history; verified clean at n-envs 4 on this Windows setup. n-envs locked at 4, recorded per command."
3. NEXT_EXPERIMENTS.md constants: confirm `--n-envs 4`; mark Branches C and D as superseded by HANDOFF.md §4–5.
4. Record A/B actual commands (Step 1 output) in CLAUDE.md Handoff Status.

## 7. Standing guardrails (unchanged)

- Never re-run A or B expecting different results. No Branch C. No ETH/SOL until BTC promotes.
- Trade rate > 0 before reading any other gate. G3 meaningless when G6 = 0.
- G6 band 0.40–0.80 provisional (stock-calibrated) — record distribution, don't auto-kill selective-but-profitable.
- `reward_turnover_penalty_scale` stays 0.0. Fee-aware reward always on. No minute bars.
- One variable family per batch; >20 runs needs written justification; ≥5 seeds; `--append`.
- v1/v2 rank against nothing. Cross-interval and cross-fee-assumption rows never rank together.
- CDP key View-only; no live orders; credentials in `.env` only.

## 8. Definition of done for next session

Steps 1–3 complete, probe numbers recorded in CLAUDE.md, fork chosen with rationale, and the chosen batch *designed* (via strategist skill, full output format) — running it is optional; designing it correctly is not.

**Status 2026-07-27: Steps 1–4 complete.** Fork resolved to the features branch (no confirmed edge in current observation space). Feature batch (NEXT_EXPERIMENTS.md Branch E) designed via the `crypto-experiment-strategist` skill as two gated phases (E0: vol term structure + volume seasonality, no new data; E1: ETH/BTC cross-rate, requires ETH-USD backfill).

**Phase E0 implemented and probed 2026-07-27 — still no confirmed edge.** Same AUC range (0.51–0.55) with the two new features added. One sweep cell initially looked positive on both val and test, but a second model family (GradientBoosting) was negative at that exact operating point and the cell was an isolated spike, not a plateau — classic threshold-fishing artifact from a 40-cell sweep, not real structure. `scripts/signal_probe.py`'s confirmed-edge check now requires agreement across both model families in addition to both splits; with that bar, zero cells clear.

**Phase E1 also run 2026-07-27 — still no confirmed edge.** ETH-USD backfilled, `EthBtcRelReturn` implemented and sanity-checked (clean, no NaNs, real variation). Probe result: AUC 0.55 max, zero cells clear the same three-layer bar (min flips, val/test agreement, cross-model agreement).

**Branch E is exhausted — both candidate feature families (free and data-cost) found nothing.** Per §5's own failure interpretation, this is no longer a quick-batch decision. Two escalation paths, mutually exclusive for the next session: (a) maker-fee execution modeling — test whether the fee floor itself (0.60% taker) exceeds extractable edge, which Branch B was originally meant to test but never did (collapsed degenerate instead of trading-and-losing); or (b) reconsider, as a scope question, whether 1h/4h BTC-USD Binary PPO can clear fees with any readily available feature set at all. **This choice is presented to the user rather than decided unilaterally** — it changes the shape of the next phase of work, not just the next batch.

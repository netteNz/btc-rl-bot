---
name: crypto-experiment-strategist
description: 'Design tightly scoped experiment batches for the coinbase-rl-bot crypto RL project after the research question has already been identified. Use to isolate variables, define controls, set success criteria, and produce execution-ready sweep commands. Adapted for Binary PPO long/flat on spot crypto with fee-aware reward, 24/7 calendar, B&H-BTC benchmark, and the 6-gate promotion framework.'
argument-hint: 'What validated research question, failure mode, or follow-up hypothesis should be turned into an experiment batch? (e.g. btc-baseline-v3 corrected baseline, min-hold isolation, 4h horizon escalation)'
user-invocable: true
---

# Crypto Experiment Strategist

Turn a validated research question into a controlled experiment batch for spot crypto RL.

## Objective
Design the next batch so the maximum is learned with the minimum compute and noise — under crypto's fee and calendar economics, which dominate before hyperparameters matter.

## Project Context (read before designing)
- **Algorithm: Binary PPO, long/flat.** On spot crypto this is the market's actual action space — no shorting on spot. Never propose SAC, continuous sizing, or short legs.
- **Market: 24/7/365.** No sessions, no gaps. Hourly annualization = √8760 ≈ 93.6 (vs √1638 ≈ 40.5 stock hourly). **Never reuse stock-calibrated Sharpe/alpha thresholds numerically.**
- **Benchmark: buy-and-hold BTC** everywhere alpha is computed. QQQ does not exist in this repo.
- **Fee economics set the floor:** ~0.60% taker at entry tier → ~1.2% round trip. Bar interval 1h with min-hold ≥ 6 (or 4h with min-hold 3). Never design sweeps at minute bars.
- **Standard sweep template:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  .\.venv\Scripts\python.exe src\experiments.py `
      --product BTC-USD `
      --interval 1h `
      --binary-actions `
      --min-hold-bars 6 `
      --reward-mode sharpe `
      --transaction-cost-rate 0.006 `
      --initial-balance 10000 `
      --min-trade-notional 1.0 `
      --ent-coefs 0.01,0.02,0.05 `
      --timesteps 40000 `
      --seeds 3,7,13,21,42 `
      --execution-mode next_bar `
      --reward-turnover-penalty-scale 0.0 `
      --max-weight-delta-per-step 0.10 `
      --use-stationary-features `
      --run-label "your_label_here" `
      --append
  ```
- **Non-negotiable flags:** `--binary-actions`, `--min-hold-bars` (≥6 at 1h), `--transaction-cost-rate` (fee-aware reward always on), `--initial-balance` (must be set explicitly, not left to accident), `--reward-turnover-penalty-scale 0.0`, `--max-weight-delta-per-step 0.10`, `--use-stationary-features`, `--append`, minimum 5 seeds.
- **`reward-turnover-penalty-scale` MUST be 0.** The turnover penalty was a proxy for costs the stock env didn't model; explicit fees model them for real. Stacking both was one of two causes of the btc-baseline-v1/v2 always-flat collapse. Whipsaw control is min-hold's job, not the reward's.
- **`--initial-balance` MUST be passed explicitly.** It was not even a CLI flag until btc-baseline-v2 also collapsed at 0% trade rate -- every prior sweep silently used the env's $1,000 class default against BTC-USD prices of $16.5K-$126K, where whole-share position flooring made `floor(balance / price) == 0` for every bar in the dataset. Position sizing is fractional now (Coinbase supports ~1e-8 BTC precision), so this is no longer a structural blocker at any balance, but it is still a real fee-tier realism knob (CLAUDE.md's 0.60% taker assumption is for < $10K 30-day volume) -- do not let it default silently again.
- **Post-sweep evaluation:** Always `python scripts/evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv --label <label>`.
- **Promotion pipeline (in order):** evaluate_sweep.py → sanity_scan.py → generate_ensemble_config.py (verify seed pins manually) → walkforward confirmation → paper trading via portfolio_tracker. No live orders — View-only key until explicitly approved.
- **6 promotion gates required.** G3 = alpha vs B&H BTC. **A G3 pass is meaningless unless G6 > 0** — always-flat passes G3 spuriously in any net-down test window. G3/G5 thresholds are provisional until the first valid (trading) BTC baseline provides crypto-native distributions; never silently loosen them — update project CLAUDE.md when calibrating.
- **Product status:** BTC-USD — baseline v1 invalidated (reward mis-specification → always-flat collapse; 0% trade rate across all 15 runs). **baseline v2 ALSO invalidated** with the reward fix already in place -- still 0% trade rate across all 15 runs, root-caused to whole-share position sizing against BTC-scale prices (see failure modes below). Both bugs are now fixed (`reward_turnover_penalty_scale` defaults to 0.0; `PositionManager` sizes fractionally; `--initial-balance`/`--min-trade-notional` are real flags) — **btc-baseline-v3 is unblocked**. ETH-USD / SOL-USD — blocked until BTC promotes.
- **Known failure modes (crypto-native):**
  - Always-flat collapse (0% trade rate + spurious G3 pass): check position sizing (`initial_balance` vs asset price scale) *and* reward cost double-counting before touching hyperparameters -- both have independently caused this exact symptom in this repo (v1: reward; v2: sizing)
  - Position quantization: whole-share flooring against a balance smaller than the asset price silently zeroes out every trade, independent of reward or policy quality -- always sanity-check `floor(initial_balance / typical_price)` before trusting a 0% trade rate diagnosis is reward-side
  - Fee application bug: fee charged per bar held instead of per position change → flat becomes the only survivable policy
  - Phantom rebalancing on hold: recomputing target position size from net worth on every bar (instead of only when the weight target changes) fires meaningless "hold" trades as a post-fee cash residual drifts against a moving price
  - "Single config group" in eval header + identical rows across ent_coefs: verify from run logs that the sweep actually varied configs before interpreting anything
  - Seed collapse (0.0/0.0): expected in small numbers, filter with `filter_active_seeds`; ALL seeds collapsed = env/reward specification issue, not a seed issue

## Use this skill when
- The next research question is already known
- A corrected baseline (version increment) or follow-up batch needs design
- A gate failure needs focused diagnosis
- The 1h→4h horizon escalation decision is on the table

## Do not use this skill when
- The fee application unit test has not passed (flat-forever episode → zero fees; one round trip → exactly 2× taker). Nothing runs before this invariant holds.
- The main problem is still figuring out what happened (diagnose first, then design)
- The batch would target ETH/SOL before BTC promotes

## Core Procedure

### 1. Restate the exact research question
One explicit, falsifiable question. Examples:
- Does BTC-USD pass a fee-aware Binary PPO baseline at 1h bars with positive alpha vs B&H BTC? (btc-baseline-v3, still open -- v1/v2 both invalidated pre-trade)
- Does min-hold 12 vs 6 change the fee-clearance economics at 1h?
- Does 4h horizon clear the fee floor where 1h could not?

### 2. Choose the minimum informative batch
- New/corrected baseline: 3 ent_coefs × 5 seeds = 15 runs
- Variable isolation: fix all but one family, ≤ 15 runs
- Never > 20 runs without written justification

### 3. Define the experiment structure
For each experiment specify: goal, exact variable(s) changed, exact variables held constant (always include binary-actions, min-hold, transaction-cost-rate, turnover-scale 0, max-weight-delta, stationary features, seeds), and evaluation artifacts to inspect.

### 4. Define success and failure interpretation
**Universal success criteria:**
- G6 trade rate in target zone (60–75%); gate band (0.40, 0.80)
- Trade rate > 0 before any other gate is read — otherwise the run is diagnostic, not evidential
- Alpha ≥ 0.00 vs B&H BTC
- Actionable accuracy ≥ 0.53, val/test drift ≤ 0.05
- CV < 1.0 over active seeds (requires ≥ 5 seeds; CV with 3–4 seeds is a seed-count artifact)

**Decision rules:**
- Baseline invalidated by reward mis-specification → same template, increment version label (vN+1). A correction, not batch 2.
- Valid trading baseline fails alpha at 1h → next variable is **horizon (4h)**, not hyperparameters. Fee economics dominate before entropy coefficients matter.
- Trade rate in band but alpha fails → now, and only now, hyperparameters are on the table.

### 5. Protect comparability
Hold the standard template constant unless the deviation is the variable under test. Note the fee tier assumption in the run label context — fee rates change with 30-day volume, and results at 0.60% taker are not comparable to results at 0.40% without saying so.

### 6. Produce execution-ready run plans
Always include: venv activation, full one-liner sweep command, post-sweep evaluate command, expected leaderboard label.

## Required Output Format
1. **Research question**
2. **Why this batch is the right next step**
3. **Controlled experiment batch**
4. **Variables changed**
5. **Variables held constant**
6. **Success criteria**
7. **Failure interpretation**
8. **Execution-ready run plans**
9. **Priority order**
10. **Leaderboard comparability impact (REQUIRED)**

## Leaderboard Comparability Rule
- Low impact: same base config, only timesteps or seeds changed
- Medium impact: min-hold or ent_coef range changed
- High impact: bar interval, fee rate, or observation space changed — results are a new comparison universe; never rank across bar intervals or fee assumptions in one leaderboard view

Always note whether the batch is exploratory or confirmatory.

## Constraints
- Never omit `--binary-actions`, `--transaction-cost-rate`, `--min-hold-bars`, `--initial-balance`, `--max-weight-delta-per-step 0.10`, `--use-stationary-features`, or `--append` from any sweep command
- Never assume a 0% trade rate is reward-side without first checking `initial_balance` against the product's price scale (whole-share sizing was the actual cause of btc-baseline-v2's collapse, not the reward)
- Never set `--reward-turnover-penalty-scale` to anything but 0.0
- Never design minute-bar sweeps
- Never recommend ETH/SOL batches before BTC promotes
- Never recommend promoting with < 5 seeds
- Never read G3 in isolation — trade rate first
- Never recalibrate G3/G5 thresholds silently — update project CLAUDE.md in the same change
- One variable family at a time
- Always include the post-sweep evaluate command in the run plan
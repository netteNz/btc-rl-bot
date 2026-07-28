# Fee Sensitivity Sweep — Resolving the Branch E Fork (2026-07-27)

**Authorship note:** this document is Fable's response to the HANDOFF.md decision fork (maker-fee
modeling vs. reconsidering scope), relayed into the repo by the user. It is not Claude's analysis —
recorded here verbatim in substance so the reasoning and its authorship stay attached to each other,
rather than drifting into unattributed "the docs say" fact the way the n-envs claim did.

## Correction to HANDOFF.md §3 (self-correction, from Fable)

The original HANDOFF.md asserted `--n-envs 4` was "empirically verified clean" and should be locked
in as the repo default. That claim was converted from a one-line report into a locked repo constant
without any run in this repo backing it — argparse default is 8, v3 ran at 1, A/B's actual value is
unknown, and no test of 4 exists anywhere in the repo. This is precisely the doc-drift-into-fact
failure pattern this project's handoffs exist to prevent, and this time Fable was the source, not a
prior session's carelessness. The session's walk-back (see CLAUDE.md, HANDOFF.md §5 Step 1) stands.
The run-manifest gap it exposed — `n_envs`/`initial_balance`/`min_trade_notional` never logged per
leaderboard row — is worth fixing regardless of which direction below is chosen.

## The actual finding: economic, not informational

The probe results (original 18 features, then Branch E Phases E0 and E1) are a clean scientific
finding, and the finding is economic rather than informational. AUC 0.52–0.55 replicating across two
model families, two horizons, and two independent chronological windows — through three feature
families — says BTC 1h direction *is* weakly predictable from the observation space tested. The
signal is real. The problem is arithmetic: a 0.53–0.55 AUC at 1h horizons translates to a per-trade
gross edge on the order of single-digit-to-low-double-digit bps, against a 120bp taker round trip.
Roughly an order-of-magnitude gap. This is why E0 and E1 landed in the identical AUC band —
incremental features at the same horizon move edge by bps when the deficit is measured in tens of
bps. More feature families won't close that gap; NEXT_EXPERIMENTS.md and HANDOFF.md are right to stop
that line of experimentation.

## The real next decision isn't (a) vs (b) — it's a measurement neither doc has run yet

The (a) maker-fee-modeling vs. (b) reconsider-scope fork, as posed in HANDOFF.md, isn't actually the
next decision. It's the *conclusion* of a measurement that hasn't been run: the fee-sensitivity sweep.
`scripts/signal_probe.py` already computes fee-adjusted expectancy at a fixed 0.6% taker rate.
Parameterize the fee and re-run the same three-layer confirmed-edge criterion (min flips, val/test
sign agreement, cross-model agreement) across a grid of round-trip costs — say 1.2%, 0.8%, 0.5%, 0.3%,
0.1%, 0% — and find the **breakeven round trip**: the fee level where cells first clear the bar.
Roughly 30 minutes of work against an existing script. It converts the (a)/(b) fork from a judgment
call into a lookup.

### Interpretation zones

- **Breakeven ≥ ~0.8%:** maker/maker at entry tier (0.80% round trip) might already suffice → option
  (a) is live. Caveat from the start: maker fills are contingent and adversely selected — you get
  filled preferentially when price is moving against you — so any maker-fee model needs a
  fill-probability haircut, or it's optimism dressed as realism.
- **Breakeven ~0.3–0.7%:** (a) alone fails at entry tier; the real question becomes whether fee-tier
  progression is realistic. At a $10K balance, sustaining the 30-day volume needed for the 0.25/0.40
  tier is plausible only if the bot itself is trading actively enough to get there — circular at best.
  Honest read: thin path.
- **Breakeven ≤ ~0.2%:** no retail spot path exists at intraday horizons, full stop, and (b) resolves
  itself without further debate.

## If breakeven lands low: one more cheap test before accepting the scope conclusion

If the breakeven lands in the third zone (Fable's prior: it will), there's exactly one cheap test left
before accepting the scope conclusion — the **daily-horizon probe**. Same `scripts/signal_probe.py`,
resampled to 1d bars, horizons 1 and 5. Daily BTC moves run 2–3%, so the fee-to-edge ratio improves
roughly 10x purely from horizon arithmetic — it's the only same-data configuration where a 0.53 AUC
could conceivably clear a 1.2% round trip. Cost: ~3,200 daily bars is thin for PPO training, so even a
positive probe result there buys a harder RL problem, not a solved one.

The difference this test resolves: "this scope is dead" vs. "this scope is dead *at intraday
frequencies specifically*."

## If both come back empty: this is not project failure

If the fee-sensitivity sweep and the daily-horizon probe both come back empty, that should not be
framed as failure. The sibling stock bot's economics had zero commissions as a *load-bearing*
assumption, and this project just demonstrated — with proper controls, replication criteria, and no
self-deception — that the assumption doesn't port to crypto. "TA-derived features cannot clear retail
crypto fees at intraday horizons" is a true, hard-won result that most retail bot builders never
establish, which is exactly why most retail bots bleed out on fees instead. The methodology refused to
promote fiction across five consecutive batches. That's the system working, not failing. The pipeline,
the probe harness, and the fee-aware eval framework all remain assets for whatever the reframed scope
becomes — regime-scale allocation, cross-sectional alts, or shelving crypto until the fee structure or
available data sources change.

## Sequence

1. Fee-sensitivity sweep (extend `scripts/signal_probe.py` with a `--fee-rates` grid; re-run the
   existing `find_confirmed_edge` bar at each level; ~30 min of work against the existing script).
2. Breakeven high (≥ ~0.8%) → design the maker-fee execution model, with a fill-probability haircut
   built in from the start, not bolted on after an optimistic result.
3. Breakeven low (≤ ~0.2–0.3%) → run the daily-horizon probe (resample to 1d, horizons 1 and 5).
4. Both empty → the scope decision is genuinely the user's to make at that point, not a batch design
   question. Write it up as a proposal per HANDOFF.md's original framing, informed by real breakeven
   numbers instead of a guess.

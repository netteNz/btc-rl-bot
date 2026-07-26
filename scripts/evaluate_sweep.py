"""
scripts/evaluate_sweep.py

Post-sweep evaluation script. Loads a leaderboard CSV, applies all 6 promotion gates,
surfaces the overtrade diagnostic, ranks survivors by Sharpe, and prints a champion
summary with copy-paste ensemble config command.

Ported from the sibling stock bot's scripts/evaluate_sweep.py. Per CLAUDE.md, this
pipeline stage "carries over unchanged" except for the G3 benchmark swap (QQQ -> B&H BTC)
and the "ticker" -> "product" column rename (a crypto pair like BTC-USD isn't a ticker).
Gate thresholds below are numerically identical to the stock bot's and are explicitly
PROVISIONAL until the first BTC baseline gives a crypto-native return/Sharpe distribution
(see CLAUDE.md "Calendar & benchmark").

Gate 5 (CV Stability) recomputes CV using only active seeds (Sharpe > 0,
trade_rate > 10%) to prevent collapsed seeds from artificially inflating
the config-level CV and blocking otherwise stable configurations.

Usage:
    python scripts/evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv
    python scripts/evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv --label btc-baseline-v1
    python scripts/evaluate_sweep.py --leaderboard data/experiment_leaderboard.csv --label btc-baseline-v1 --promote
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Gate definitions -- must match CLAUDE.md's 6-gate promotion framework
# ---------------------------------------------------------------------------
GATES = [
    {
        "id": 1,
        "name": "Actionable Accuracy",
        "col": "test_actionable_accuracy",
        "op": ">=",
        "threshold": 0.53,
    },
    {
        "id": 2,
        "name": "Trade Win Rate",
        "col": "test_trade_win_rate",
        "op": ">=",
        "threshold": 0.52,
    },
    {
        "id": 3,
        "name": "Alpha vs B&H BTC",
        "col": "test_alpha_vs_btc",
        "op": ">=",
        "threshold": 0.00,
    },
    {
        "id": 4,
        "name": "Val/Test Drift",
        "col": None,          # computed: |val_acc - test_acc|
        "op": "<=",
        "threshold": 0.05,
    },
    {
        "id": 5,
        "name": "CV Stability",
        "col": "test_return_cv_by_config",
        "op": "<",
        "threshold": 0.50,
    },
    {
        "id": 6,
        "name": "Trade Rate",
        "col": "test_trade_rate",
        "op": "between",
        "threshold": (0.40, 0.80),
    },
]

# Overtrade diagnostic -- the fix target
TRADE_RATE_TARGET_LOW = 0.60
TRADE_RATE_TARGET_HIGH = 0.75
TRADE_RATE_COL = "test_trade_rate"

# Column used for final ranking among gate passers
RANK_COL = "test_sharpe_ratio"

# Degenerate-policy diagnostic (CLAUDE.md Failure Patterns, learned from btc-baseline-v1):
# a near-0% trade rate (always-flat) or near-100% trade rate (always-long) is a collapsed
# policy, not a real strategy. G6 already blocks these from `all_gates`, but a spurious
# G3 pass on a degenerate row is easy to miss on a quick read of the leaderboard --
# "alpha >= 0 with zero trades" just means the benchmark happened to be down over the
# window, not that the policy found any edge. Surface it explicitly.
DEGENERATE_TRADE_RATE_LOW = 0.02
DEGENERATE_TRADE_RATE_HIGH = 0.98


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate_pass(row: pd.Series, gate: dict) -> bool:
    if gate["id"] == 4:
        val_col = _find_col(row.index, ["val_actionable_accuracy", "val_accuracy"])
        test_col = _find_col(row.index, ["test_actionable_accuracy", "test_accuracy"])
        if val_col is None or test_col is None:
            return False
        drift = abs(row[val_col] - row[test_col])
        return drift <= gate["threshold"]

    col = gate["col"]
    if col not in row.index or pd.isna(row[col]):
        return False
    v = row[col]
    if gate["op"] == ">=":
        return v >= gate["threshold"]
    if gate["op"] == "<=":
        return v <= gate["threshold"]
    if gate["op"] == "<":
        return v < gate["threshold"]
    if gate["op"] == ">":
        return v > gate["threshold"]
    if gate["op"] == "between":
        return gate["threshold"][0] <= v <= gate["threshold"][1]
    return False


def _find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _gate_symbol(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _trade_rate_symbol(rate: float) -> str:
    if pd.isna(rate):
        return "N/A"
    if TRADE_RATE_TARGET_LOW <= rate <= TRADE_RATE_TARGET_HIGH:
        return f"OK {rate:.1%}"
    if rate < TRADE_RATE_TARGET_LOW:
        return f"LOW {rate:.1%} (under-trade)"
    return f"HIGH {rate:.1%} (overtrade)"


def _bar(passed: int, total: int) -> str:
    filled = "#" * passed
    empty = "." * (total - passed)
    return f"[{filled}{empty}] {passed}/{total}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate sweep leaderboard against promotion gates.")
    parser.add_argument("--leaderboard", required=True, help="Path to leaderboard CSV")
    parser.add_argument("--label", default=None, help="Filter by run_label substring")
    parser.add_argument("--product", default=None, help="Filter by product (e.g. BTC-USD)")
    parser.add_argument("--top", type=int, default=10, help="How many rows to show in ranked table (default 10)")
    parser.add_argument("--promote", action="store_true", help="Run generate_ensemble_config.py for the champion after evaluation")
    parser.add_argument("--g6-min-trade-rate", type=float, default=None, help="Override G6 lower bound for trade rate (default: 0.40).")
    parser.add_argument("--g6-max-trade-rate", type=float, default=None, help="Override G6 upper bound for trade rate (default: 0.80).")
    args = parser.parse_args()

    if args.g6_min_trade_rate is not None or args.g6_max_trade_rate is not None:
        for gate in GATES:
            if gate["id"] == 6:
                lo = args.g6_min_trade_rate if args.g6_min_trade_rate is not None else gate["threshold"][0]
                hi = args.g6_max_trade_rate if args.g6_max_trade_rate is not None else gate["threshold"][1]
                gate["threshold"] = (lo, hi)
                break
        global TRADE_RATE_TARGET_LOW, TRADE_RATE_TARGET_HIGH
        if args.g6_min_trade_rate is not None:
            TRADE_RATE_TARGET_LOW = args.g6_min_trade_rate
        if args.g6_max_trade_rate is not None:
            TRADE_RATE_TARGET_HIGH = args.g6_max_trade_rate

    # ---- Load ---------------------------------------------------------------
    lb_path = Path(args.leaderboard)
    if not lb_path.exists():
        print(f"ERROR: leaderboard not found: {lb_path}")
        sys.exit(1)

    df = pd.read_csv(lb_path)
    print(f"\n{'=' * 70}")
    print(f"  SWEEP EVALUATION  --  {lb_path.name}")
    print(f"{'=' * 70}")
    print(f"  Total rows loaded : {len(df)}")

    # ---- Filter -------------------------------------------------------------
    if args.label:
        label_col = _find_col(df.columns, ["run_label", "label", "experiment_label"])
        if label_col:
            df = df[df[label_col].str.contains(args.label, na=False)]
            print(f"  Filtered by label : '{args.label}'  ->  {len(df)} rows")
        else:
            print("  WARNING: No run_label column found -- skipping label filter")

    if args.product:
        product_col = _find_col(df.columns, ["product", "ticker", "symbol"])
        if product_col:
            df = df[df[product_col].str.upper() == args.product.upper()]
            print(f"  Filtered by product: {args.product.upper()}  ->  {len(df)} rows")

    if df.empty:
        print("\n  No rows match the filter criteria. Exiting.")
        sys.exit(0)

    # ---- Recompute CV using active seeds only --------------------------------
    ACTIVE_SHARPE_MIN = 0.0
    ACTIVE_TRADE_MIN = 0.10
    ret_col = _find_col(df.columns, ["test_cumulative_return", "test_cumulative_signal_return"])
    shr_col = _find_col(df.columns, ["test_sharpe_ratio"])
    tr_col_ = _find_col(df.columns, [TRADE_RATE_COL, "trade_rate", "trade_frequency"])

    if ret_col and shr_col and tr_col_:
        active_mask = (df[shr_col] > ACTIVE_SHARPE_MIN) & (df[tr_col_] > ACTIVE_TRADE_MIN)
        active_df = df[active_mask]
        n_active = active_mask.sum()

        config_keys = [c for c in ["run_label", "ent_coef"] if c in df.columns]
        if config_keys and not active_df.empty:
            cv_map = (
                active_df.groupby(config_keys)[ret_col]
                .apply(lambda x: x.std() / x.mean() if len(x) > 1 and x.mean() != 0 else float("nan"))
                .reset_index()
                .rename(columns={ret_col: "clean_cv"})
            )
            df = df.merge(cv_map, on=config_keys, how="left")
            print(f"\n  CV recomputed over active seeds only ({n_active}/{len(df)} rows active, "
                  f"Sharpe > {ACTIVE_SHARPE_MIN}, trade_rate > {ACTIVE_TRADE_MIN:.0%})")
        else:
            if not active_df.empty and ret_col in active_df.columns:
                ret_vals = active_df[ret_col]
                single_cv = ret_vals.std() / ret_vals.mean() if ret_vals.mean() != 0 else float("nan")
            else:
                single_cv = float("nan")
            df["clean_cv"] = single_cv
            print(f"\n  CV recomputed over active seeds only ({n_active}/{len(df)} rows active) -- "
                  f"single config group, clean_cv={single_cv:.4f}")
    else:
        df["clean_cv"] = df.get("test_return_cv_by_config", float("nan"))

    for gate in GATES:
        if gate["id"] == 5:
            gate["col"] = "clean_cv"

    # ---- Per-row gate evaluation --------------------------------------------
    gate_cols = []
    for gate in GATES:
        col_name = f"gate_{gate['id']}_pass"
        df[col_name] = df.apply(lambda row, g=gate: _gate_pass(row, g), axis=1)
        gate_cols.append(col_name)

    df["gates_passed"] = df[gate_cols].sum(axis=1).astype(int)
    df["all_gates"] = df["gates_passed"] == len(GATES)

    tr_col = _find_col(df.columns, [TRADE_RATE_COL, "trade_rate", "trade_frequency"])

    # ---- Summary stats -------------------------------------------------------
    n_total = len(df)
    gate_pass_rates = {f"Gate {g['id']} ({g['name']})": df[f"gate_{g['id']}_pass"].mean() for g in GATES}

    print(f"\n{'-' * 70}")
    print("  GATE PASS RATES")
    print(f"{'-' * 70}")
    for name, rate in gate_pass_rates.items():
        bar = "#" * int(rate * 20)
        empty = "." * (20 - int(rate * 20))
        print(f"  {name:<35}  [{bar}{empty}]  {rate:.0%}")

    print(f"\n  Configs with {len(GATES)}/{len(GATES)} gates : {df['all_gates'].sum()} / {n_total}")

    if tr_col:
        print(f"\n{'-' * 70}")
        print(f"  TRADE RATE DISTRIBUTION  (target: {TRADE_RATE_TARGET_LOW:.0%}-{TRADE_RATE_TARGET_HIGH:.0%})")
        print(f"{'-' * 70}")
        bands = {
            f"Overtrade   (> {TRADE_RATE_TARGET_HIGH:.0%})": df[tr_col] > TRADE_RATE_TARGET_HIGH,
            f"Target zone ({TRADE_RATE_TARGET_LOW:.0%}-{TRADE_RATE_TARGET_HIGH:.0%})": (df[tr_col] >= TRADE_RATE_TARGET_LOW) & (df[tr_col] <= TRADE_RATE_TARGET_HIGH),
            f"Under-trade (< {TRADE_RATE_TARGET_LOW:.0%})": df[tr_col] < TRADE_RATE_TARGET_LOW,
        }
        for label, mask in bands.items():
            count = mask.sum()
            pct = count / n_total if n_total else 0
            bar = "#" * int(pct * 20)
            empty = "." * (20 - int(pct * 20))
            print(f"  {label:<40}  [{bar}{empty}]  {count:3d} rows  ({pct:.0%})")

    # ---- Degenerate-policy diagnostic ----------------------------------------
    if tr_col:
        always_flat = df[tr_col] <= DEGENERATE_TRADE_RATE_LOW
        always_long = df[tr_col] >= DEGENERATE_TRADE_RATE_HIGH
        degenerate = always_flat | always_long
        df["is_degenerate"] = degenerate
        n_degenerate = int(degenerate.sum())

        gate3_col = "gate_3_pass"
        spurious_g3 = degenerate & df.get(gate3_col, False)
        n_spurious = int(spurious_g3.sum())

        print(f"\n{'-' * 70}")
        print("  DEGENERATE-POLICY DIAGNOSTIC")
        print(f"{'-' * 70}")
        print(f"  Always-flat rows (trade_rate <= {DEGENERATE_TRADE_RATE_LOW:.0%}) : {int(always_flat.sum())}")
        print(f"  Always-long rows (trade_rate >= {DEGENERATE_TRADE_RATE_HIGH:.0%}) : {int(always_long.sum())}")
        if n_spurious > 0:
            print(f"\n  WARNING: {n_spurious} degenerate row(s) show a passing Gate 3 (alpha vs B&H BTC).")
            print("  A degenerate always-flat/always-long policy did not find an edge -- G3 passing")
            print("  just means the benchmark moved against a static position over this window.")
            print("  G6 (trade rate) already excludes these from 'all_gates', but do not read G3 in")
            print("  isolation when triaging a sweep. See CLAUDE.md Failure Patterns.")
            label_col_deg = _find_col(df.columns, ["run_label", "label"])
            cols_to_show = [c for c in [label_col_deg, "seed", tr_col, "test_alpha_vs_btc"] if c and c in df.columns]
            print(df.loc[spurious_g3, cols_to_show].head(10).to_string(index=False))
        elif n_degenerate > 0:
            print(f"\n  {n_degenerate} degenerate row(s) found, none with a spurious Gate 3 pass. G6 will exclude them.")
        else:
            print("\n  No degenerate always-flat/always-long rows detected.")

    # ---- Ranked table of top configs ----------------------------------------
    rank_col = _find_col(df.columns, [RANK_COL, "test_sharpe_ratio"])
    sort_cols = ["all_gates", rank_col] if rank_col else ["gates_passed"]
    sort_asc = [False, False] if rank_col else [False]

    ranked = df.sort_values(sort_cols, ascending=sort_asc).head(args.top)

    print(f"\n{'-' * 70}")
    print(f"  TOP {args.top} CONFIGS  (sorted: {len(GATES)}/{len(GATES)} gates first, then Sharpe)")
    print(f"{'-' * 70}")

    display_always = ["run_label", "product", "seed"]
    display_metrics = [
        "test_actionable_accuracy", "test_trade_win_rate",
        "test_alpha_vs_btc", "test_return_cv_by_config", "clean_cv",
        rank_col, tr_col,
    ]
    display_gate = ["gates_passed"]

    show_cols = [c for c in display_always if c in df.columns]
    show_cols += [c for c in display_metrics if c and c in df.columns]
    show_cols += display_gate

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(ranked[show_cols].to_string(index=False))

    # ---- Gate detail for top 3 ----------------------------------------------
    print(f"\n{'-' * 70}")
    print("  GATE BREAKDOWN -- TOP 3")
    print(f"{'-' * 70}")
    for _, row in ranked.head(3).iterrows():
        label_col = _find_col(ranked.columns, ["run_label", "label"])
        row_label = row[label_col] if label_col else "?"
        seed_col = _find_col(ranked.columns, ["seed"])
        seed_val = f"  seed={int(row[seed_col])}" if seed_col and not pd.isna(row[seed_col]) else ""

        gates_passed = int(row["gates_passed"])
        print(f"\n  [{row_label}{seed_val}]  {_bar(gates_passed, len(GATES))}")
        for gate in GATES:
            passed = bool(row[f"gate_{gate['id']}_pass"])
            sym = _gate_symbol(passed)

            if gate["id"] == 4:
                val_col = _find_col(row.index, ["val_actionable_accuracy", "val_accuracy"])
                test_col = _find_col(row.index, ["test_actionable_accuracy", "test_accuracy"])
                drift = abs(row[val_col] - row[test_col]) if (val_col and test_col) else float("nan")
                detail = f"drift={drift:.4f}"
            elif gate["id"] == 5:
                raw_cv = row.get("test_return_cv_by_config", float("nan"))
                clean_cv = row.get("clean_cv", float("nan"))
                detail = f"clean_cv={clean_cv:.4f}  (raw_cv={raw_cv:.4f})"
            else:
                col = gate["col"]
                val = row[col] if col and col in row.index else float("nan")
                detail = f"value={val:.4f}" if not pd.isna(val) else "N/A"

            print(f"    [{sym}]  Gate {gate['id']}: {gate['name']:<22}  {detail}  (thresh {gate['op']} {gate['threshold']})")

        if tr_col and tr_col in row.index:
            print(f"    Trade rate: {_trade_rate_symbol(row[tr_col])}")

    # ---- Champion -----------------------------------------------------------
    champion_df = ranked[ranked["all_gates"] == True]  # noqa: E712

    print(f"\n{'=' * 70}")
    if champion_df.empty:
        print(f"  NO CHAMPION -- no config passed all {len(GATES)} gates.")
        print("  Recommendation: check the degenerate-policy diagnostic above first -- a 0%")
        print("  or ~100% trade rate means the config never really tried. reward_turnover_penalty_scale")
        print("  should stay at 0 in this repo (fees already price trading); if trade rate is in range")
        print("  but alpha/win-rate still fail, try reward_hold_penalty_scale or ent_coef next,")
        print("  not turnover penalty. Remember G3/G5")
        print("  thresholds are provisional -- do not silently loosen them without")
        print("  updating CLAUDE.md once the first BTC baseline distribution exists.")
        print(f"{'=' * 70}\n")
        sys.exit(0)

    champion = champion_df.iloc[0]
    print("  CHAMPION IDENTIFIED")
    print(f"{'=' * 70}")

    label_col = _find_col(champion.index, ["run_label", "label"])
    seed_col = _find_col(champion.index, ["seed"])
    product_col2 = _find_col(champion.index, ["product", "ticker", "symbol"])

    champ_label = champion[label_col] if label_col else "unknown"
    champ_seed = int(champion[seed_col]) if seed_col else "unknown"
    champ_product = champion[product_col2].upper() if product_col2 else "BTC-USD"

    sharpe_val = champion[rank_col] if rank_col and rank_col in champion.index else "N/A"
    alpha_val = champion["test_alpha_vs_btc"] if "test_alpha_vs_btc" in champion.index else "N/A"
    acc_val = champion["test_actionable_accuracy"] if "test_actionable_accuracy" in champion.index else "N/A"
    wr_val = champion["test_trade_win_rate"] if "test_trade_win_rate" in champion.index else "N/A"
    tr_val = champion[tr_col] if tr_col and tr_col in champion.index else None

    print(f"  Label   : {champ_label}")
    print(f"  Product : {champ_product}   Seed : {champ_seed}")
    print(f"  Sharpe  : {sharpe_val}")
    print(f"  Alpha vs B&H BTC : {alpha_val}")
    print(f"  Acc     : {acc_val}    WinRate : {wr_val}")
    if tr_val is not None:
        print(f"  Trade rate: {_trade_rate_symbol(tr_val)}")

    # ---- Next steps ---------------------------------------------------------
    print(f"\n{'-' * 70}")
    print("  NEXT STEPS")
    print(f"{'-' * 70}")
    print("\n  1. Lock baseline snapshot before any further changes:")
    print(f"       python scripts/sanity_scan.py --leaderboard {lb_path}")
    print("\n  2. Regenerate ensemble config with champion:")
    print("       python scripts/generate_ensemble_config.py \\")
    print(f"           --leaderboard {lb_path} \\")
    print(f"           --product {champ_product} \\")
    print(f"           --label {champ_label}")
    print("\n  3. Validate ensemble config was written correctly:")
    print("       cat staging/models/ensemble_config.json")
    print("\n  4. Walk-forward confirmation, then paper trading via portfolio_tracker")
    print("     (no live orders -- View-only key until explicitly approved).")
    print()

    # ---- Auto-promote -------------------------------------------------------
    if args.promote:
        print(f"{'-' * 70}")
        print("  --promote flag set. Running generate_ensemble_config.py ...")
        print(f"{'-' * 70}")
        cmd = [
            sys.executable, "scripts/generate_ensemble_config.py",
            "--leaderboard", str(lb_path),
            "--product", champ_product,
            "--label", champ_label,
        ]
        print(f"  CMD: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            print("\n  generate_ensemble_config.py completed successfully.")
        else:
            print(f"\n  generate_ensemble_config.py exited with code {result.returncode}.")


if __name__ == "__main__":
    main()

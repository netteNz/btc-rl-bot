"""
scripts/generate_ensemble_config.py

Generate or update staging/models/ensemble_config.json from a leaderboard CSV.

Ported from the sibling stock bot's scripts/generate_ensemble_config.py with the
"ticker" -> "product" rename (CLAUDE.md pipeline carries this script over unchanged
otherwise). Per CLAUDE.md's known gotchas, the run_label filter below is unreliable --
always verify seed pins manually in staging/models/ensemble_config.json after running this.
"""

import json
import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ensemble import SparseEnsemble


def main():
    parser = argparse.ArgumentParser(description="Generate or update ensemble configuration.")
    parser.add_argument("--leaderboard", type=str, help="Path to leaderboard CSV for a specific product.")
    parser.add_argument("--product", type=str, help="Product to update (e.g., BTC-USD). Required if --leaderboard is provided.")
    parser.add_argument("--label", type=str, help="Optional run_label filter for the leaderboard.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top seeds to include in the ensemble (default: 3).")
    args = parser.parse_args()

    data_dir = ROOT_DIR / "data"
    staging_dir = ROOT_DIR / "staging" / "models"
    staging_dir.mkdir(parents=True, exist_ok=True)
    config_out = staging_dir / "ensemble_config.json"

    config = {}
    if config_out.exists():
        with open(config_out, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode {config_out}. Starting fresh.")

    if args.leaderboard and args.product:
        products_to_process = {args.product.lower(): (args.leaderboard, args.label)}
    elif args.leaderboard or args.product:
        print("Error: Both --leaderboard and --product must be provided together.")
        sys.exit(1)
    else:
        products_to_process = {
            "btc-usd": ("experiment_leaderboard.csv", None),
        }

    for product, (lb_file, label_filter) in products_to_process.items():
        lb_file_path = Path(lb_file)
        if lb_file_path.exists():
            leaderboard_path = lb_file_path
        elif (data_dir / lb_file).exists():
            leaderboard_path = data_dir / lb_file
        else:
            print(f"Warning: {lb_file} not found locally or in {data_dir}. Skipping {product}.")
            continue

        print(f"Processing {product}...")
        ensemble = SparseEnsemble(str(leaderboard_path))

        if label_filter:
            initial_len = len(ensemble.active_seeds_df)
            label_col = next((c for c in ensemble.active_seeds_df.columns if c in ["run_label", "label"]), None)
            if label_col:
                ensemble.active_seeds_df = ensemble.active_seeds_df[ensemble.active_seeds_df[label_col] == label_filter]
                print(f"  Filtered by label '{label_filter}': {initial_len} -> {len(ensemble.active_seeds_df)} rows.")
            else:
                print("  Warning: No run_label column found. Skipping label filter.")

        dropped = ensemble.filter_active_seeds(min_test_trades=20)
        print(f"  Dropped {dropped} collapsed seeds.")

        active_seeds_count = len(ensemble.active_seeds_df)
        if active_seeds_count == 0:
            print(f"  Error: No active seeds found for {product}. Skipping.")
            continue

        top_n = min(args.top_n, active_seeds_count)
        ensemble.load_top_n_models(n=top_n, run_label_filter=label_filter)

        metrics = ensemble.aggregate_metrics()
        top_n_sharpe = metrics.get("ensemble_mean_test_sharpe", 0.0)
        top_n_gap = metrics.get("ensemble_mean_val_test_gap", 1.0)

        active_seed_list = [int(info["seed"]) for info in ensemble.top_models_info]

        # Production-readiness heuristic carried over from the stock bot. Thresholds here
        # are PROVISIONAL -- crypto Sharpe is annualized with sqrt(8760), not sqrt(252), so
        # a "0.20 Sharpe" bar calibrated on daily stock bars is not directly comparable.
        # Revisit once the first BTC baseline gives a crypto-native distribution.
        if active_seeds_count >= 2 and top_n_sharpe >= 0.20 and top_n_gap <= 0.05:
            ready = True
            notes = "production ready (PROVISIONAL threshold)"
        elif active_seeds_count >= 2 or (0.15 <= top_n_sharpe < 0.20):
            ready = "monitor"
            notes = "borderline ensemble or marginal alpha"
        else:
            ready = False
            notes = "Sharpe below threshold or insufficient active seeds"

        first_row = ensemble.active_seeds_df.iloc[0] if not ensemble.active_seeds_df.empty else {}
        min_hold = int(first_row.get("min_hold_bars", 0)) if "min_hold_bars" in first_row else 0
        interval = str(first_row.get("interval", "1h")) if "interval" in first_row else "1h"
        use_cooldown = bool(first_row.get("use_cooldown_obs", 0)) if "use_cooldown_obs" in first_row else False

        config[product] = {
            "active_seeds": active_seed_list,
            "ensemble_method": "voting",
            "top_n_mean_sharpe": round(top_n_sharpe, 3),
            "top_n_mean_val_test_gap": round(top_n_gap, 3),
            "production_ready": ready,
            "notes": f"{active_seeds_count} active seeds. {notes}",
            "run_label": label_filter if label_filter else "N/A",
            "leaderboard_csv": str(leaderboard_path.relative_to(ROOT_DIR)) if leaderboard_path.is_relative_to(ROOT_DIR) else str(leaderboard_path),
            "interval": interval,
            "min_hold_bars": min_hold,
            "use_cooldown_obs": use_cooldown,
        }

    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved master config to {config_out}")
    print("Reminder (CLAUDE.md known gotcha): the run_label filter above is unreliable -- "
          "verify seed pins manually before treating this config as canonical.")


if __name__ == "__main__":
    main()

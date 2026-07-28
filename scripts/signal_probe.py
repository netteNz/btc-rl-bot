"""
Signal probe (HANDOFF.md Step 3): does a supervised model find ANY forward-return
edge in the crypto stationary feature set, independent of PPO/env mechanics?

Every trading row across btc-baseline-v3 / btc-minhold-12 / btc-4h-baseline-v1 sits at
~0.49 accuracy on the actionable signal -- coin-flip. Before spending a batch on
entropy/reward-shaping to force the policy off its degenerate corners, check whether
there is any edge in the observation space at all for those corners to find.

Same data loader, same feature set, and the same walk-forward split ratios as
src/experiments.py -- results are directly comparable to the leaderboard's
train/val/test regime split. Strictly chronological; no shuffling anywhere.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.market_data import get_crypto_training_data  # noqa: E402
from src.feature_engineering import STATIONARY_FEATURE_COLUMNS as FEATURE_COLUMNS  # noqa: E402

TAKER_FEE = 0.006  # CLAUDE.md baseline: 0.60% retail taker tier
ENTRY_THRESHOLD = 0.55  # naive long/flat rule probed for fee-adjusted expectancy


def split_walk_forward(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """Mirror src/experiments.py::_split_walk_forward exactly (chronological, no shuffling)."""
    ordered = df.sort_values("Date").reset_index(drop=True)
    n = len(ordered)
    train_end = max(int(n * train_ratio), 10)
    val_end = max(train_end + int(n * val_ratio), train_end + 5)
    val_end = min(val_end, n - 5)
    return (
        ordered.iloc[:train_end].reset_index(drop=True),
        ordered.iloc[train_end:val_end].reset_index(drop=True),
        ordered.iloc[val_end:].reset_index(drop=True),
    )


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    fwd_log_return = np.log(df["RawClose"].shift(-horizon) / df["RawClose"])
    df["fwd_log_return"] = fwd_log_return
    df["target"] = (fwd_log_return > 0).astype(int)
    return df.iloc[:-horizon].reset_index(drop=True)


def apply_min_hold(raw_positions: np.ndarray, min_hold: int) -> np.ndarray:
    """Lock a position for at least min_hold bars after each accepted flip -- mirrors the
    env's min_hold_bars gate: a flip signal is ignored while the hold timer is active."""
    if min_hold <= 1 or len(raw_positions) == 0:
        return raw_positions.copy()
    out = np.empty_like(raw_positions)
    current = raw_positions[0]
    out[0] = current
    last_flip_idx = 0
    for i in range(1, len(raw_positions)):
        if raw_positions[i] != current and (i - last_flip_idx) >= min_hold:
            current = raw_positions[i]
            last_flip_idx = i
        out[i] = current
    return out


def expectancy_for_positions(positions: np.ndarray, fwd_returns: np.ndarray, fee: float):
    """One taker fee per flip (proxy for the env's per-flip taker penalty)."""
    prev = np.concatenate([[0], positions[:-1]])
    flips = positions != prev
    raw_pnl = positions * fwd_returns
    fee_drag = flips.astype(float) * fee
    net_pnl = raw_pnl - fee_drag
    return float(net_pnl.mean()), float(flips.mean())


def fee_adjusted_expectancy(probs: np.ndarray, fwd_returns: np.ndarray, threshold: float, fee: float):
    """Naive rule: long when prob > threshold, else flat. No min-hold constraint --
    used only for the headline classification table; see the threshold/min-hold sweep
    for the fairer comparison against this repo's actual min_hold_bars=6 policies."""
    positions = (probs > threshold).astype(int)
    return expectancy_for_positions(positions, fwd_returns, fee)


def run_probe(horizon: int, product: str, interval: str, train_ratio: float, val_ratio: float):
    df = get_crypto_training_data(product=product, interval=interval, use_stationary_features=True)
    df = add_targets(df, horizon)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    train_df, val_df, test_df = split_walk_forward(df, train_ratio, val_ratio)

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["target"].to_numpy()

    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "GradientBoosting": GradientBoostingClassifier(random_state=0),
    }

    rows = []
    raw_predictions = {}  # (horizon, model_name, split) -> (probs, fwd_returns)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            X = split_df[feature_cols].to_numpy()
            y = split_df["target"].to_numpy()
            probs = model.predict_proba(X)[:, 1]
            fwd_returns = split_df["fwd_log_return"].to_numpy()
            raw_predictions[(horizon, model_name, split_name)] = (probs, fwd_returns)

            preds = (probs > 0.5).astype(int)
            auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
            acc = accuracy_score(y, preds)
            expectancy, trade_rate = fee_adjusted_expectancy(probs, fwd_returns, ENTRY_THRESHOLD, TAKER_FEE)
            rows.append({
                "horizon_bars": horizon,
                "model": model_name,
                "split": split_name,
                "n": len(y),
                "auc": round(auc, 4),
                "accuracy": round(acc, 4),
                "fee_adj_expectancy_per_bar": round(expectancy, 6),
                "flip_rate_at_p>0.55": round(trade_rate, 4),
            })
    return pd.DataFrame(rows), raw_predictions


def run_expectancy_sweep(raw_predictions: dict, thresholds: list[float], min_holds: list[int], fee: float) -> pd.DataFrame:
    """For every fitted (horizon, model, split), sweep entry threshold x min-hold and report
    fee-adjusted expectancy. Answers: does ANY operating point turn this signal fee-positive,
    including at this repo's actual min_hold_bars=6 floor -- not just the naive threshold=0.55,
    no-hold rule from the headline table."""
    rows = []
    for (horizon, model_name, split_name), (probs, fwd_returns) in raw_predictions.items():
        for threshold in thresholds:
            raw_positions = (probs > threshold).astype(int)
            for min_hold in min_holds:
                positions = apply_min_hold(raw_positions, min_hold)
                expectancy, flip_rate = expectancy_for_positions(positions, fwd_returns, fee)
                n_flips = int(round(flip_rate * len(positions)))
                rows.append({
                    "horizon_bars": horizon,
                    "model": model_name,
                    "split": split_name,
                    "threshold": threshold,
                    "min_hold": min_hold,
                    "fee_adj_expectancy_per_bar": round(expectancy, 6),
                    "flip_rate": round(flip_rate, 4),
                    "n_flips": n_flips,
                })
    return pd.DataFrame(rows)


def find_confirmed_edge(sweep: pd.DataFrame, min_flips: int) -> pd.DataFrame:
    """A cell only counts as a candidate edge if: (1) it has at least min_flips flips (a
    handful of trades landing on the right side of zero by chance is not an edge), (2) val
    AND test agree in sign at the SAME (horizon, threshold, min_hold) -- two independent
    chronological windows finding the same thing is the bar for 'this replicates', not just
    picking whichever split happened to come out positive, and (3) BOTH model families agree
    in sign at that same (horizon, threshold, min_hold) too. (3) matters as much as (2): a
    5-threshold x 2-min_hold x 2-model x 2-split sweep has enough cells that one isolated
    positive island surrounded by negative neighbors, found by only one of the two models,
    is exactly what you'd expect from noise fishing -- not evidence a second, differently
    structured model would also find."""
    candidates = sweep[sweep["n_flips"] >= min_flips]
    if candidates.empty:
        return candidates

    key_cols = ["horizon_bars", "threshold", "min_hold"]
    pivot = candidates.pivot_table(index=key_cols, columns=["model", "split"], values="fee_adj_expectancy_per_bar")
    required = [("LogisticRegression", "val"), ("LogisticRegression", "test"),
                ("GradientBoosting", "val"), ("GradientBoosting", "test")]
    if not all(col in pivot.columns for col in required):
        return pivot.iloc[0:0].reset_index()

    agrees_positive = pivot[
        (pivot[("LogisticRegression", "val")] > 0) & (pivot[("LogisticRegression", "test")] > 0)
        & (pivot[("GradientBoosting", "val")] > 0) & (pivot[("GradientBoosting", "test")] > 0)
    ]
    return agrees_positive.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizons", default="1,6", help="Comma-separated forward-return horizons in bars.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sweep-thresholds", default="0.55,0.60,0.65,0.70,0.75",
                        help="Comma-separated entry-probability thresholds for the fee-adjusted expectancy sweep.")
    parser.add_argument("--sweep-min-holds", default="1,6",
                        help="Comma-separated min-hold-bar values for the sweep. 1 = no constraint (naive rule); "
                             "6 = this repo's actual fee-floor policy (CLAUDE.md: 1h bars, min-hold >= 6).")
    parser.add_argument("--min-flips-for-edge", type=int, default=20,
                        help="Minimum flip count for a sweep cell to be eligible as a candidate edge. Below this, "
                             "a positive average is indistinguishable from a handful of trades landing well by "
                             "chance -- not evidence of anything.")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    thresholds = [float(t) for t in args.sweep_thresholds.split(",")]
    min_holds = [int(m) for m in args.sweep_min_holds.split(",")]

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    all_metrics = []
    all_predictions = {}
    for h in horizons:
        metrics, predictions = run_probe(h, args.product, args.interval, args.train_ratio, args.val_ratio)
        all_metrics.append(metrics)
        all_predictions.update(predictions)
    report = pd.concat(all_metrics, ignore_index=True)

    print("=== Classification metrics (naive threshold=0.55, no min-hold) ===")
    print(report.to_string(index=False))

    sweep = run_expectancy_sweep(all_predictions, thresholds, min_holds, TAKER_FEE)
    print()
    print("=== Fee-adjusted expectancy sweep (threshold x min-hold) ===")
    print(sweep.to_string(index=False))

    max_auc = report["auc"].max()
    confirmed = find_confirmed_edge(sweep, args.min_flips_for_edge)

    print()
    print(f"=== Confirmed-edge candidates (>= {args.min_flips_for_edge} flips per split, "
          f"BOTH models x BOTH splits all positive at the same threshold/min-hold) ===")
    if confirmed.empty:
        print("(none)")
    else:
        print(confirmed.to_string(index=False))

    print()
    print(f"Max AUC across all horizons/splits/models: {max_auc:.4f}")

    if max_auc > 0.60:
        print("WARNING: AUC > 0.60 -- suspect leakage before celebrating. Audit feature/target "
              "alignment first (HANDOFF.md Step 3 note).")
    elif confirmed.empty:
        print("No operating point clears the confirmed-edge bar: either every cell is fee-negative, "
              "the only positive cells have too few flips to mean anything, a positive result doesn't "
              "replicate on val at the same threshold/min-hold, or it only shows up in one of the two "
              "model families (an isolated threshold spike one model happened to land on, not "
              "structure a second model would also find). Treat as no extractable edge. Next variable "
              "family = features, not entropy/reward-shaping. Design via crypto-experiment-strategist skill.")
    else:
        print("At least one operating point clears fees with a non-trivial trade count AND agrees "
              "in sign across val and test AND across both model families independently. Signal may "
              "be economically extractable. The entropy/reward-shaping batch (NEXT_EXPERIMENTS.md "
              "Branch D) is justified -- but PPO's own action space (binary, min_hold=6) must be able "
              "to reach a similar operating point to the one(s) listed above.")


if __name__ == "__main__":
    main()

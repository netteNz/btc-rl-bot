# ============================================================================ #
# COINBASE-RL-BOT - EXPERIMENT RUNNER
# ============================================================================ #
#
# Adapted from the sibling stock bot's src/experiments.py. Per CLAUDE.md, the pipeline
# shape (sweep -> evaluate_sweep -> sanity_scan -> generate_ensemble_config) and most
# knobs carry over unchanged. The deltas actually made here are exactly the "Three
# Crypto Deltas" from CLAUDE.md:
#   1. Fee-aware reward is mandatory -> default transaction_cost_rate models the
#      ~0.60% taker/taker round trip; min_hold_bars defaults to 6 at 1h bars.
#   2. 24/7 calendar + BTC annualization -> bars_per_year comes from
#      src.market_data.get_interval_bars_per_year (8760 @ 1h, 2190 @ 4h), and alpha is
#      computed against buy-and-hold BTC (src.market_data.load_benchmark_close_prices),
#      not QQQ.
#   3. Regime coverage -> --start/--end let a sweep explicitly target a range spanning
#      both a drawdown leg and a bull leg; there is no automatic guarantee of this, it
#      is an experiment-design responsibility (see CLAUDE.md "Regime coverage").
#
# Deliberately dropped relative to the stock bot's runner (out of scope per CLAUDE.md):
# news features, the daily/intraday_5m experiment-preset switch, and Phase 1 telemetry
# instrumentation. All three were stock-bot-specific refinements, not part of this
# project's build order.
#
# G3/G5 gate thresholds below are carried over NUMERICALLY UNCHANGED from the stock bot
# and are explicitly PROVISIONAL per CLAUDE.md, pending the first BTC baseline's
# crypto-native return/Sharpe distribution.
# ============================================================================ #
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.trading_env import LEADERBOARD_VERSION, CryptoTradingEnv
from src.market_data import (
    benchmark_equity_curve,
    get_crypto_training_data,
    get_interval_bars_per_year,
    load_benchmark_close_prices,
)
from src.products import resolve_product
from src.signal_analytics import compute_metrics, enrich_with_truth_labels

DEFAULT_LEADERBOARD_PATH = ROOT_DIR / "data" / "experiment_leaderboard.csv"
DEFAULT_REWARD_LEADERBOARD_PATH = ROOT_DIR / "data" / "experiment_reward_leaderboard.csv"
DEFAULT_SUMMARY_PATH = ROOT_DIR / "data" / "experiment_summary.json"
DEFAULT_SNAPSHOT_DIR = ROOT_DIR / "data" / "experiment_snapshots"

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _split_walk_forward(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 30:
        raise ValueError("Dataset is too small for walk-forward split (need at least 30 rows).")

    ordered = df.sort_values("Date").reset_index(drop=True)
    n = len(ordered)
    train_end = max(int(n * train_ratio), 10)
    val_end = max(train_end + int(n * val_ratio), train_end + 5)
    val_end = min(val_end, n - 5)

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    val_df = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test_df = ordered.iloc[val_end:].reset_index(drop=True)

    if len(val_df) < 5 or len(test_df) < 5:
        raise ValueError("Split produced too few rows in validation/test. Adjust ratios or provide more data.")

    return train_df, val_df, test_df


# ============================================================================ #
# SIMULATION ENGINE
# ============================================================================ #
def _simulate_with_model(model, df: pd.DataFrame, env_kwargs: dict) -> pd.DataFrame:
    """
    Executes a deterministic out-of-sample simulation using the trained agent.
    Execution is bar-by-bar and strictly deterministic; features accessed at step T
    never include information from T+1 or later (no look-ahead).
    """
    eval_kwargs = env_kwargs.copy()
    eval_kwargs["max_episode_steps"] = 0
    eval_kwargs["random_start"] = False
    env = CryptoTradingEnv(df, **eval_kwargs)
    obs, _ = env.reset()
    rows: list[dict] = []

    is_maskable = "MaskablePPO" in str(type(model))

    while True:
        step_idx = env.current_step
        if is_maskable and hasattr(env, "action_masks"):
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        price = float(df.loc[step_idx, env.price_column])
        date_value = pd.to_datetime(df.loc[step_idx, "Date"]) if "Date" in df.columns else step_idx

        obs, reward, terminated, truncated, info = env.step(action)
        discrete_pos = env.position
        rows.append(
            {
                "step": step_idx,
                "date": date_value,
                "price": price,
                "action": discrete_pos,
                "reward": float(reward),
                "net_worth": float(env.net_worth),
                "reward_portfolio_return": float(info.get("reward_portfolio_return", 0.0)),
                "reward_direction": float(info.get("reward_direction", 0.0)),
                "reward_pnl": float(info.get("reward_pnl", 0.0)),
                "reward_hold_penalty": float(info.get("reward_hold_penalty", 0.0)),
                "reward_action_bonus": float(info.get("reward_action_bonus", 0.0)),
                "reward_drawdown_penalty": float(info.get("reward_drawdown_penalty", 0.0)),
                "reward_drawdown": float(info.get("reward_drawdown", 0.0)),
                "realized_return": float(info.get("realized_return", 0.0)),
            }
        )
        if terminated or truncated:
            break

    return pd.DataFrame(rows)


def _summarize_rewards(signal_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    if signal_df.empty:
        return {
            f"{prefix}_reward_total_mean": 0.0,
            f"{prefix}_reward_total_sum": 0.0,
            f"{prefix}_reward_portfolio_return_mean": 0.0,
            f"{prefix}_reward_direction_mean": 0.0,
            f"{prefix}_reward_pnl_mean": 0.0,
            f"{prefix}_reward_hold_penalty_mean": 0.0,
            f"{prefix}_reward_action_bonus_mean": 0.0,
            f"{prefix}_reward_turnover_penalty_mean": 0.0,
            f"{prefix}_reward_drawdown_penalty_mean": 0.0,
            f"{prefix}_reward_drawdown_mean": 0.0,
        }

    return {
        f"{prefix}_reward_total_mean": float(signal_df["reward"].mean()),
        f"{prefix}_reward_total_sum": float(signal_df["reward"].sum()),
        f"{prefix}_reward_portfolio_return_mean": float(signal_df["reward_portfolio_return"].mean()),
        f"{prefix}_reward_direction_mean": float(signal_df["reward_direction"].mean()),
        f"{prefix}_reward_pnl_mean": float(signal_df.get("reward_pnl", pd.Series([0.0])).mean()),
        f"{prefix}_reward_hold_penalty_mean": float(signal_df["reward_hold_penalty"].mean()),
        f"{prefix}_reward_action_bonus_mean": float(signal_df.get("reward_action_bonus", pd.Series([0.0])).mean()),
        f"{prefix}_reward_turnover_penalty_mean": float(signal_df.get("reward_turnover_penalty", pd.Series([0.0])).mean()),
        f"{prefix}_reward_drawdown_penalty_mean": float(signal_df["reward_drawdown_penalty"].mean()),
        f"{prefix}_reward_drawdown_mean": float(signal_df["reward_drawdown"].mean()),
    }


def _ranking_score(metrics_obj) -> float:
    cumulative_clipped = max(min(metrics_obj.cumulative_signal_return, 1.0), -1.0)
    return (
        0.50 * metrics_obj.actionable_accuracy
        + 0.30 * metrics_obj.trade_win_rate
        + 0.20 * cumulative_clipped
    )


def _robustness_score(
    *,
    ranking_score: float,
    test_alpha_vs_btc: float,
    val_actionable_accuracy: float,
    test_actionable_accuracy: float,
    test_return_cv_by_config: float,
) -> float:
    test_alpha_clipped = max(min(float(test_alpha_vs_btc), 1.0), -1.0)
    val_test_gap = abs(float(val_actionable_accuracy) - float(test_actionable_accuracy))
    cv_penalty = min(float(test_return_cv_by_config), 5.0) / 5.0
    return float(ranking_score + (0.10 * test_alpha_clipped) - (0.05 * val_test_gap) - (0.05 * cv_penalty))


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def _annualized_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    std = float(clean.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(np.sqrt(max(periods_per_year, 1)) * clean.mean() / std)


def _annualized_sortino(returns: pd.Series, periods_per_year: int) -> float:
    clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    downside = clean[clean < 0.0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    if downside_std <= 1e-12:
        return 0.0
    return float(np.sqrt(max(periods_per_year, 1)) * clean.mean() / downside_std)


def _risk_metrics_from_equity(equity: pd.Series, prefix: str, periods_per_year: int) -> dict[str, float]:
    curve = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    if curve.empty:
        return {
            f"{prefix}_cumulative_return": 0.0,
            f"{prefix}_sharpe_ratio": 0.0,
            f"{prefix}_sortino_ratio": 0.0,
            f"{prefix}_max_drawdown": 0.0,
        }

    returns = curve.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    peak = curve.cummax().replace(0.0, np.nan)
    drawdown = (curve / peak) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    cumulative_return = float((curve.iloc[-1] / max(curve.iloc[0], 1e-8)) - 1.0)

    return {
        f"{prefix}_cumulative_return": cumulative_return,
        f"{prefix}_sharpe_ratio": _annualized_sharpe(returns, periods_per_year=periods_per_year),
        f"{prefix}_sortino_ratio": _annualized_sortino(returns, periods_per_year=periods_per_year),
        f"{prefix}_max_drawdown": max_drawdown,
    }


def _attach_config_stability_metrics(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard

    leaderboard = leaderboard.copy()
    if "leaderboard_version" not in leaderboard.columns:
        leaderboard["leaderboard_version"] = 1
    else:
        leaderboard["leaderboard_version"] = leaderboard["leaderboard_version"].fillna(1).astype(int)

    config_keys = [
        "leaderboard_version",
        "product",
        "interval",
        "timesteps",
        "learning_rate",
        "gamma",
        "ent_coef",
        "use_stationary_features",
        "threshold",
        "horizon",
        "transaction_cost_rate",
        "trade_penalty",
        "execution_mode",
        "spread_bps",
        "slippage_bps",
        "max_weight_delta_per_step",
        "reward_mode",
        "rolling_reward_window",
        "reward_epsilon",
        "reward_return_scale",
        "reward_pnl_scale",
        "reward_direction_scale",
        "reward_hold_penalty_scale",
        "reward_drawdown_penalty_scale",
        "reward_action_bonus_scale",
        "reward_turnover_penalty_scale",
        "reward_clip",
        "reward_ignore_transaction_cost",
        "binary_actions",
        "min_hold_bars",
    ]

    grouped = (
        leaderboard.groupby(config_keys, dropna=False)["test_cumulative_signal_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "test_return_mean_by_config",
                "std": "test_return_std_by_config",
                "count": "config_seed_count",
            }
        )
    )
    grouped["test_return_std_by_config"] = grouped["test_return_std_by_config"].fillna(0.0)
    denom = grouped["test_return_mean_by_config"].abs()
    grouped["test_return_cv_by_config"] = np.where(
        denom > 1e-8,
        grouped["test_return_std_by_config"] / denom,
        np.inf,
    )
    grouped["high_return_cv_risk"] = (grouped["test_return_cv_by_config"] >= 1.0).astype(int)

    merged = leaderboard.merge(grouped, on=config_keys, how="left")
    return merged


def _passes_promotion_gates(row: pd.Series, args: argparse.Namespace) -> bool:
    test_actionable = float(row.get("test_actionable_accuracy", 0.0))
    test_win_rate = float(row.get("test_trade_win_rate", 0.0))
    test_alpha = float(row.get("test_alpha_vs_btc", float("-inf")))
    val_actionable = float(row.get("val_actionable_accuracy", 0.0))
    test_cv = float(row.get("test_return_cv_by_config", float("inf")))
    test_trade_count = int(row.get("test_trade_count", 0))
    test_actionable_support = int(row.get("test_actionable_support", 0))

    return (
        test_actionable >= float(args.promote_min_test_actionable)
        and test_win_rate >= float(args.promote_min_test_win_rate)
        and test_alpha >= float(args.promote_min_test_alpha)
        and abs(val_actionable - test_actionable) <= float(args.promote_max_val_test_gap)
        and test_cv < float(args.promote_max_test_cv)
        and test_trade_count >= int(args.promote_min_test_trade_count)
        and test_actionable_support >= int(args.promote_min_test_actionable_support)
    )


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def _safe_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return sanitized[:80] if sanitized else "run"


def make_env(df, env_kwargs):
    def _init():
        return CryptoTradingEnv(df, **env_kwargs)
    return _init


def _format_duration(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


class ProgressCallback(BaseCallback):
    """Lightweight print-based progress indicator (no extra UI dependency)."""
    def __init__(self, total_timesteps: int, print_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_every = print_every
        self._last_printed = 0
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_printed >= self.print_every or self.num_timesteps >= self.total_timesteps:
            elapsed = time.time() - self._start_time
            pct = 100.0 * self.num_timesteps / max(self.total_timesteps, 1)
            print(f"    ...{self.num_timesteps}/{self.total_timesteps} steps ({pct:.0f}%) [{elapsed:.0f}s]", end="\r")
            self._last_printed = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        print()


def write_experiment_outputs(
    leaderboard: pd.DataFrame,
    leaderboard_path: Path,
    reward_leaderboard_path: Path,
    summary_path: Path,
    snapshot_dir: Path | None = DEFAULT_SNAPSHOT_DIR,
    run_label: str | None = None,
    append_results: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    reward_leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    new_results = leaderboard.copy()
    if "leaderboard_version" not in new_results.columns:
        new_results["leaderboard_version"] = LEADERBOARD_VERSION
    else:
        new_results["leaderboard_version"] = new_results["leaderboard_version"].fillna(LEADERBOARD_VERSION).astype(int)

    historical_leaderboard_path = leaderboard_path.with_name(f"{leaderboard_path.stem}_history{leaderboard_path.suffix}")
    historical_reward_leaderboard_path = reward_leaderboard_path.with_name(f"{reward_leaderboard_path.stem}_history{reward_leaderboard_path.suffix}")

    cumulative_history = new_results.copy()
    if historical_leaderboard_path.exists():
        try:
            existing_history = pd.read_csv(historical_leaderboard_path)
            cumulative_history = pd.concat([existing_history, new_results], ignore_index=True)
        except Exception as e:
            print(f"Warning: could not read existing history: {e}")

    cumulative_history.sort_values(["leaderboard_version", "ranking_score"], ascending=[False, False]).to_csv(historical_leaderboard_path, index=False)
    cumulative_history.sort_values("val_reward_total_mean", ascending=False).to_csv(historical_reward_leaderboard_path, index=False)

    if append_results and leaderboard_path.exists():
        try:
            existing = pd.read_csv(leaderboard_path)
            leaderboard = pd.concat([existing, new_results], ignore_index=True)
        except Exception as e:
            print(f"Warning: could not append to existing leaderboard: {e}")
            leaderboard = new_results.copy()
    else:
        leaderboard = new_results.copy()

    leaderboard = leaderboard.sort_values(["leaderboard_version", "ranking_score"], ascending=[False, False]).reset_index(drop=True)
    comparable_leaderboard = leaderboard[leaderboard["leaderboard_version"] == LEADERBOARD_VERSION].copy()
    if comparable_leaderboard.empty:
        comparable_leaderboard = leaderboard.copy()

    comparable_leaderboard.to_csv(leaderboard_path, index=False)
    reward_leaderboard = comparable_leaderboard.sort_values("val_reward_total_mean", ascending=False).reset_index(drop=True)
    reward_leaderboard.to_csv(reward_leaderboard_path, index=False)

    timestamp = _timestamp_slug()
    summary: dict[str, object] = {
        "rows": int(len(comparable_leaderboard)),
        "generated_at_utc": timestamp,
        "leaderboard_path": str(leaderboard_path),
        "reward_leaderboard_path": str(reward_leaderboard_path),
        "leaderboard_history_path": str(historical_leaderboard_path),
        "reward_leaderboard_history_path": str(historical_reward_leaderboard_path),
        "leaderboard_version": LEADERBOARD_VERSION,
        "top3": leaderboard.head(3).to_dict(orient="records"),
    }

    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{timestamp}_{_safe_label(run_label)}" if run_label else timestamp
        snapshot_leaderboard_path = snapshot_dir / f"experiment_leaderboard_{suffix}.csv"
        snapshot_reward_leaderboard_path = snapshot_dir / f"experiment_reward_leaderboard_{suffix}.csv"
        snapshot_summary_path = snapshot_dir / f"experiment_summary_{suffix}.json"

        comparable_leaderboard.to_csv(snapshot_leaderboard_path, index=False)
        reward_leaderboard.to_csv(snapshot_reward_leaderboard_path, index=False)

        snapshot_history_leaderboard_path = snapshot_dir / f"experiment_leaderboard_history_{suffix}.csv"
        snapshot_history_reward_path = snapshot_dir / f"experiment_reward_leaderboard_history_{suffix}.csv"

        cumulative_history.to_csv(snapshot_history_leaderboard_path, index=False)
        cumulative_history.sort_values("val_reward_total_mean", ascending=False).to_csv(
            snapshot_history_reward_path,
            index=False,
        )

        snapshot_summary = {
            **summary,
            "leaderboard_path": str(snapshot_leaderboard_path),
            "reward_leaderboard_path": str(snapshot_reward_leaderboard_path),
            "leaderboard_history_path": str(snapshot_history_leaderboard_path),
            "reward_leaderboard_history_path": str(snapshot_history_reward_path),
        }
        snapshot_summary_path.write_text(json.dumps(snapshot_summary, indent=2), encoding="utf-8")
        summary["snapshot_paths"] = {
            "leaderboard": str(snapshot_leaderboard_path),
            "reward_leaderboard": str(snapshot_reward_leaderboard_path),
            "leaderboard_history": str(snapshot_history_leaderboard_path),
            "reward_leaderboard_history": str(snapshot_history_reward_path),
            "summary": str(snapshot_summary_path),
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return reward_leaderboard, summary


# ============================================================================ #
# MAIN EXPERIMENT PIPELINE
# ============================================================================ #
def run_experiments(args: argparse.Namespace) -> pd.DataFrame:
    interval = str(args.interval)
    bars_per_year = get_interval_bars_per_year(interval)
    seeds = _parse_int_list(args.seeds)
    learning_rates = _parse_float_list(args.learning_rates)
    gammas = _parse_float_list(args.gammas)
    ent_coefs = _parse_float_list(args.ent_coefs)
    timesteps_list = _parse_int_list(args.timesteps)

    df = get_crypto_training_data(
        product=args.product,
        interval=interval,
        use_stationary_features=args.use_stationary_features,
        start=args.start,
        end=args.end,
    )
    train_df, val_df, test_df = _split_walk_forward(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    benchmark_prices = load_benchmark_close_prices(
        start=pd.to_datetime(val_df["Date"]).min(),
        end=pd.to_datetime(test_df["Date"]).max(),
        interval=interval,
    )
    val_benchmark_equity = benchmark_equity_curve(val_df, benchmark_prices)
    test_benchmark_equity = benchmark_equity_curve(test_df, benchmark_prices)
    val_benchmark_risk = _risk_metrics_from_equity(val_benchmark_equity, prefix="val_benchmark", periods_per_year=bars_per_year)
    test_benchmark_risk = _risk_metrics_from_equity(test_benchmark_equity, prefix="test_benchmark", periods_per_year=bars_per_year)

    env_kwargs: dict = {
        "initial_balance": args.initial_balance,
        "min_trade_notional": args.min_trade_notional,
        "transaction_cost_rate": args.transaction_cost_rate,
        "trade_penalty": args.trade_penalty,
        "execution_mode": args.execution_mode,
        "spread_bps": args.spread_bps,
        "slippage_bps": args.slippage_bps,
        "reward_clip": args.reward_clip,
        "reward_ignore_transaction_cost": args.reward_ignore_transaction_cost,
        "reward_mode": args.reward_mode,
        "rolling_reward_window": 100,
        "reward_epsilon": args.reward_epsilon,
        "reward_pnl_scale": args.reward_pnl_scale,
        "long_only": args.long_only,
        "binary_actions": args.binary_actions,
        "min_hold_bars": args.min_hold_bars,
        "max_episode_steps": args.max_episode_steps,
        "random_start": args.random_start,
        "use_cooldown_obs": args.use_cooldown_obs,
    }

    reward_return_scales = _parse_float_list(args.reward_return_scale)
    reward_pnl_scales = _parse_float_list(args.reward_pnl_scale)
    reward_direction_scales = _parse_float_list(args.reward_direction_scale)
    reward_hold_penalty_scales = _parse_float_list(args.reward_hold_penalty_scale)
    reward_drawdown_penalty_scales = _parse_float_list(args.reward_drawdown_penalty_scale)
    reward_action_bonus_scales = _parse_float_list(args.reward_action_bonus_scale)
    reward_turnover_penalty_scales = _parse_float_list(args.reward_turnover_penalty_scale)
    rolling_reward_windows = _parse_int_list(args.rolling_reward_window)
    max_weight_delta_per_steps = _parse_float_list(args.max_weight_delta_per_step)

    configs = list(itertools.product(
        seeds, timesteps_list, learning_rates, gammas, ent_coefs,
        reward_return_scales, reward_pnl_scales, reward_direction_scales, reward_hold_penalty_scales,
        reward_drawdown_penalty_scales, reward_action_bonus_scales, reward_turnover_penalty_scales,
        rolling_reward_windows, max_weight_delta_per_steps
    ))
    if args.max_runs > 0:
        configs = configs[: args.max_runs]

    rows: list[dict] = []
    canonical_product = resolve_product(args.product)
    print(f"Running {len(configs)} experiment runs on product={canonical_product} interval={interval}...")

    for idx, (seed, timesteps, learning_rate, gamma, ent_coef,
              ret_scale, pnl_scale, dir_scale, hold_scale, dd_scale, bonus_scale, turnover_scale, rolling_window, max_weight_delta) in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] seed={seed} timesteps={timesteps} lr={learning_rate} "
            f"gamma={gamma} ent_coef={ent_coef} dir_scale={dir_scale} mode={args.reward_mode}"
        )
        start_time = time.time()
        lr_arg = linear_schedule(learning_rate) if args.use_lr_schedule else learning_rate

        env_kwargs_run = env_kwargs.copy()
        env_kwargs_run.update({
            "max_weight_delta_per_step": max_weight_delta,
            "reward_return_scale": ret_scale,
            "reward_pnl_scale": pnl_scale,
            "reward_direction_scale": dir_scale,
            "reward_hold_penalty_scale": hold_scale,
            "reward_drawdown_penalty_scale": dd_scale,
            "reward_action_bonus_scale": bonus_scale,
            "reward_turnover_penalty_scale": turnover_scale,
            "rolling_reward_window": rolling_window,
        })

        env_kwargs_train = env_kwargs_run.copy()
        if args.train_zero_friction:
            env_kwargs_train["transaction_cost_rate"] = 0.0
            env_kwargs_train["trade_penalty"] = 0.0

        if args.n_envs > 1:
            env_train = SubprocVecEnv([make_env(train_df, env_kwargs_train) for _ in range(args.n_envs)])
        else:
            env_train = CryptoTradingEnv(train_df, **env_kwargs_train)

        if args.binary_actions:
            ppo_device = "cpu" if DEFAULT_DEVICE == "mps" else DEFAULT_DEVICE

            if args.use_action_masking:
                try:
                    from sb3_contrib import MaskablePPO
                except ImportError:
                    print("ERROR: sb3-contrib is required for action masking but not installed. Run 'pip install sb3-contrib'")
                    raise

                model = MaskablePPO(
                    "MlpPolicy",
                    env_train,
                    verbose=0,
                    seed=seed,
                    learning_rate=lr_arg,
                    gamma=gamma,
                    ent_coef=ent_coef if ent_coef > 0.0 else 0.0,
                    batch_size=args.batch_size,
                    device=ppo_device,
                )
            else:
                model = PPO(
                    "MlpPolicy",
                    env_train,
                    verbose=0,
                    seed=seed,
                    learning_rate=lr_arg,
                    gamma=gamma,
                    ent_coef=ent_coef if ent_coef > 0.0 else 0.0,
                    batch_size=args.batch_size,
                    device=ppo_device,
                )
        else:
            model_ent_coef = ent_coef if ent_coef > 0.0 else "auto"
            model = SAC(
                "MlpPolicy",
                env_train,
                verbose=0,
                seed=seed,
                learning_rate=lr_arg,
                gamma=gamma,
                ent_coef=model_ent_coef,
                batch_size=args.batch_size,
                buffer_size=max(100000, timesteps),
                device=DEFAULT_DEVICE,
            )

        callback = ProgressCallback(total_timesteps=timesteps)
        model.learn(total_timesteps=timesteps, callback=callback)

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        timestamp = _timestamp_slug()
        run_label_slug = _safe_label(args.run_label) if args.run_label else "run"
        model_filename = f"model_{timestamp}_{run_label_slug}_seed{seed}_{interval}.zip"
        model_save_path = Path(args.snapshot_dir) / model_filename
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_save_path)

        if args.n_envs > 1:
            env_train.close()

        val_signals = _simulate_with_model(model, val_df, env_kwargs=env_kwargs_run)
        test_signals = _simulate_with_model(model, test_df, env_kwargs=env_kwargs_run)

        val_enriched = enrich_with_truth_labels(val_signals, threshold=args.threshold, horizon_steps=args.horizon)
        test_enriched = enrich_with_truth_labels(test_signals, threshold=args.threshold, horizon_steps=args.horizon)
        val_metrics = compute_metrics(val_enriched)
        test_metrics = compute_metrics(test_enriched)
        val_reward = _summarize_rewards(val_signals, prefix="val")
        test_reward = _summarize_rewards(test_signals, prefix="test")
        val_strategy_risk = _risk_metrics_from_equity(val_signals["net_worth"], prefix="val", periods_per_year=bars_per_year)
        test_strategy_risk = _risk_metrics_from_equity(test_signals["net_worth"], prefix="test", periods_per_year=bars_per_year)

        row: dict = {
            "leaderboard_version": LEADERBOARD_VERSION,
            "product": canonical_product,
            "interval": interval,
            "run_label": args.run_label.strip(),
            "seed": seed,
            "timesteps": timesteps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "ent_coef": ent_coef,
            "use_stationary_features": int(args.use_stationary_features),
            "threshold": args.threshold,
            "horizon": args.horizon,
            "transaction_cost_rate": args.transaction_cost_rate,
            "trade_penalty": args.trade_penalty,
            "execution_mode": args.execution_mode,
            "spread_bps": args.spread_bps,
            "slippage_bps": args.slippage_bps,
            "max_weight_delta_per_step": max_weight_delta,
            "reward_mode": args.reward_mode,
            "rolling_reward_window": args.rolling_reward_window,
            "reward_epsilon": args.reward_epsilon,
            "reward_pnl_scale": pnl_scale,
            "reward_return_scale": ret_scale,
            "reward_direction_scale": dir_scale,
            "reward_hold_penalty_scale": hold_scale,
            "reward_drawdown_penalty_scale": dd_scale,
            "reward_action_bonus_scale": bonus_scale,
            "reward_turnover_penalty_scale": turnover_scale,
            "reward_clip": args.reward_clip,
            "reward_ignore_transaction_cost": int(args.reward_ignore_transaction_cost),
            "binary_actions": int(args.binary_actions),
            "min_hold_bars": args.min_hold_bars,
            "use_cooldown_obs": int(args.use_cooldown_obs),
            "bars_per_year": bars_per_year,
            "val_overall_accuracy": val_metrics.overall_accuracy,
            "val_actionable_accuracy": val_metrics.actionable_accuracy,
            "val_actionable_support": val_metrics.actionable_support,
            "val_trade_count": val_metrics.trade_count,
            "val_trade_rate": val_metrics.trade_rate,
            "val_trade_win_rate": val_metrics.trade_win_rate,
            "val_cumulative_signal_return": val_metrics.cumulative_signal_return,
            "test_overall_accuracy": test_metrics.overall_accuracy,
            "test_actionable_accuracy": test_metrics.actionable_accuracy,
            "test_actionable_support": test_metrics.actionable_support,
            "test_trade_count": test_metrics.trade_count,
            "test_trade_rate": test_metrics.trade_rate,
            "test_trade_win_rate": test_metrics.trade_win_rate,
            "test_cumulative_signal_return": test_metrics.cumulative_signal_return,
            "val_alpha_vs_btc": float(val_strategy_risk["val_cumulative_return"] - val_benchmark_risk["val_benchmark_cumulative_return"]),
            "test_alpha_vs_btc": float(test_strategy_risk["test_cumulative_return"] - test_benchmark_risk["test_benchmark_cumulative_return"]),
            "ranking_score": _ranking_score(val_metrics),
            "run_duration_seconds": float(time.time() - start_time),
        }
        duration_str = _format_duration(row["run_duration_seconds"])
        print(f"Run {idx} completed in {duration_str}.")
        row.update(val_reward)
        row.update(test_reward)
        row.update(val_strategy_risk)
        row.update(test_strategy_risk)
        row.update(val_benchmark_risk)
        row.update(test_benchmark_risk)
        row["model_path"] = str(model_save_path)
        rows.append(row)

        if hasattr(model, "env") and model.env is not None:
            try:
                model.env.close()
            except Exception:
                pass
        try:
            del model
        except Exception:
            pass
        try:
            del env_train
        except Exception:
            pass
        import gc
        gc.collect()

    leaderboard = pd.DataFrame(rows).sort_values("ranking_score", ascending=False).reset_index(drop=True)
    leaderboard = _attach_config_stability_metrics(leaderboard)
    leaderboard["robustness_score"] = leaderboard.apply(
        lambda row: _robustness_score(
            ranking_score=float(row.get("ranking_score", 0.0)),
            test_alpha_vs_btc=float(row.get("test_alpha_vs_btc", 0.0)),
            val_actionable_accuracy=float(row.get("val_actionable_accuracy", 0.0)),
            test_actionable_accuracy=float(row.get("test_actionable_accuracy", 0.0)),
            test_return_cv_by_config=float(row.get("test_return_cv_by_config", float("inf"))),
        ),
        axis=1,
    )
    leaderboard = leaderboard.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    return leaderboard


# ============================================================================ #
# COMMAND LINE INTERFACE & CONFIGURATION
# ============================================================================ #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fee-aware multi-seed Binary PPO experiment runner (spot crypto).")

    # ------------------------------------------------------------------ #
    # 1. Core Configuration & Data
    # ------------------------------------------------------------------ #
    parser.add_argument("--product", default="btc-usd",
                        help="Coinbase product id, case-insensitive (btc-usd, BTC-USD, eth-usd, ...). "
                             "Resolved via src.products.resolve_product -- not restricted to a fixed choice "
                             "list here since new products only need a data/raw/<PRODUCT>.parquet, not a code "
                             "change. Per CLAUDE.md, ETH/SOL are gated until BTC-USD promotes -- that gate is "
                             "a research-discipline call, not an argparse restriction.")
    parser.add_argument("--interval", default="1h", choices=["1h", "4h"],
                        help="Bar interval. CLAUDE.md fee floor: 1h bars need min-hold>=6, 4h bars need min-hold>=3.")
    parser.add_argument("--start", default=None, help="Training-data start date (YYYY-MM-DD, UTC). Default: full cached history.")
    parser.add_argument("--end", default=None, help="Training-data end date (YYYY-MM-DD, UTC). Default: full cached history.")
    parser.add_argument("--seeds", default="7,13,21", help="Comma-separated seeds.")
    parser.add_argument("--timesteps", default="80000", help="Comma-separated timesteps.")
    parser.add_argument("--learning-rates", default="0.0003", help="Comma-separated learning rates.")
    parser.add_argument("--gammas", default="0.99", help="Comma-separated gammas.")
    parser.add_argument("--ent-coefs", default="0.01,0.02,0.05", help="Comma-separated entropy coefficients.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for VRAM allocation.")
    parser.add_argument("--threshold", type=float, default=0.002, help="Signal threshold for truth-label evaluation.")
    parser.add_argument("--horizon", type=int, default=1, help="Forward horizon steps for truth-label evaluation.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Walk-forward train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Walk-forward validation ratio.")

    # ------------------------------------------------------------------ #
    # 2. Execution Realism & Fees (CLAUDE.md Crypto Delta #1: fees dominate)
    # ------------------------------------------------------------------ #
    parser.add_argument("--transaction-cost-rate", type=float, default=0.006,
                        help="Per-side fee rate. Default 0.006 = 0.60%% retail taker tier. "
                             "Maker-assumption (0.004) is a future sweep variable, not the baseline.")
    parser.add_argument("--trade-penalty", type=float, default=0.0, help="Flat penalty per executed trade, on top of the fee rate.")
    parser.add_argument("--initial-balance", type=float, default=10000.0,
                        help="Starting account balance in USD. Previously not exposed as a flag at all -- every "
                             "sweep silently used the env class default of $1,000. Position sizing is fractional "
                             "(see PositionManager), so this no longer creates a structural entry floor the way it "
                             "did under whole-share sizing, but it still matters for fee-tier realism (CLAUDE.md: "
                             "~0.60%% taker tier assumes < $10K 30-day volume).")
    parser.add_argument("--min-trade-notional", type=float, default=1.0,
                        help="Minimum USD notional for a trade to execute. Mirrors Coinbase's practical minimum "
                             "order size and absorbs the tiny fee-on-fee rebalancing residual that fractional "
                             "weight-targeting produces for a few bars after entry. A full close to flat always "
                             "executes regardless of this floor.")
    parser.add_argument("--execution-mode", default="next_bar", choices=["same_bar", "next_bar"], help="Execution timing model.")
    parser.add_argument("--spread-bps", type=float, default=0.0, help="Half-spread applied around mid for buys/sells (in bps).")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Additional one-way slippage added to execution price (in bps).")
    parser.add_argument("--max-weight-delta-per-step", default="0.10", help="Maximum absolute change in target weight allowed per step (list).")
    parser.add_argument("--reward-return-scale", default="1.0", help="Weight for portfolio-return reward term (list).")
    parser.add_argument("--reward-pnl-scale", default="0.0", help="Additional weight for realized portfolio P&L (list).")
    parser.add_argument("--reward-direction-scale", default="0.35", help="Weight for directional-alignment reward term (list).")
    parser.add_argument("--reward-hold-penalty-scale", default="0.10", help="Penalty scale for hold during movement (list).")
    parser.add_argument("--reward-drawdown-penalty-scale", default="0.10", help="Penalty scale for drawdown term (list).")
    parser.add_argument("--reward-action-bonus-scale", default="0.02", help="Bonus for taking Buy actions (list).")
    parser.add_argument("--reward-turnover-penalty-scale", default="0.0",
                        help="Penalty scale for absolute weight changes (list). MUST default to 0 in this repo: "
                             "the stock bot used this as a proxy for transaction costs it didn't model explicitly. "
                             "Crypto's --transaction-cost-rate now prices trading for real, so stacking a nonzero "
                             "turnover penalty on top double-counts the cost and makes never-trading the reward-optimal "
                             "policy (see CLAUDE.md Failure Patterns: btc-baseline-v1 degenerate always-flat collapse). "
                             "Whipsaw control is --min-hold-bars's job, not this knob's.")

    # ------------------------------------------------------------------ #
    # 3. Reward Shaping & Architecture
    # ------------------------------------------------------------------ #
    parser.add_argument("--reward-mode", default="sharpe", choices=["legacy", "sharpe", "sortino", "sparse"], help="Reward calculation mode.")
    parser.add_argument("--rolling-reward-window", default="100", help="Window size for rolling rewards (list).")
    parser.add_argument("--reward-epsilon", type=float, default=1e-6, help="Epsilon for numerical stability in rewards.")
    parser.add_argument("--max-episode-steps", type=int, default=0, help="If > 0, truncate episodes after this many steps.")
    parser.add_argument("--random-start", action="store_true", help="If set, randomize start step (requires max-episode-steps > 0).")
    parser.add_argument("--reward-clip", type=float, default=1.0, help="Reward clip bound applied symmetrically.")
    parser.add_argument(
        "--reward-ignore-transaction-cost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude transaction costs/penalties from reward shaping while keeping execution unchanged. "
             "CLAUDE.md: never report or compare results produced with this flag set -- fee-free leaderboards are fiction.",
    )
    parser.add_argument(
        "--use-cooldown-obs",
        action="store_true",
        help="Append the active cooldown boolean to the agent's observation state.",
    )
    parser.add_argument(
        "--train-zero-friction",
        action="store_true",
        help="Train in a friction-free environment while keeping realistic friction for out-of-sample evaluations.",
    )
    parser.add_argument("--use-action-masking", action="store_true", help="Use Discrete Action Masking (sb3_contrib.MaskablePPO).")
    parser.add_argument("--use-stationary-features", action="store_true", help="Use log returns and normalized technical indicators + cyclical time features.")
    parser.add_argument("--long-only", action="store_true", help="Clip actions to [0, 1] -- no short positions (spot has no shorting anyway).")
    parser.add_argument("--binary-actions", action="store_true", help="Map actions to binary long/flat (1.0 or 0.0). Locked architecture choice for this project.")
    parser.add_argument("--min-hold-bars", type=int, default=6,
                        help="Minimum bars between executed position flips. CLAUDE.md fee floor: >=6 at 1h bars, >=3 at 4h bars.")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of experiment runs (0 = all).")
    parser.add_argument("--leaderboard-path", default=str(DEFAULT_LEADERBOARD_PATH), help="CSV output path.")
    parser.add_argument("--reward-leaderboard-path", default=str(DEFAULT_REWARD_LEADERBOARD_PATH), help="Reward leaderboard CSV output path.")
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH), help="JSON summary output path.")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR), help="Directory for timestamped leaderboard/reward/summary snapshots.")
    parser.add_argument("--disable-snapshots", action="store_true", help="Disable timestamped snapshot output files.")
    parser.add_argument("--append", action="store_true", help="Append results to existing leaderboard.")
    parser.add_argument("--run-label", default="", help="Optional suffix label appended to snapshot filenames.")
    parser.add_argument("--compact-output", action="store_true", help="Print a compact top-run summary instead of full transposed output.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device (cuda, mps, cpu).")
    parser.add_argument("--use-lr-schedule", action="store_true", help="Use linear learning rate decay.")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Parallel environments for vectorized training. Known gotcha: SubprocVecEnv FD leak "
                             "on long sweeps -- fall back to --n-envs 1 if you hit Errno 24.")

    # ------------------------------------------------------------------ #
    # 4. Promotion Gates (PROVISIONAL -- see CLAUDE.md "Calendar & benchmark")
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--promote-require-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only promote champion model if promotion gates are satisfied.",
    )
    parser.add_argument("--promote-min-test-actionable", type=float, default=0.53,
                        help="PROVISIONAL (carried over from stock-bot thresholds, not yet crypto-calibrated). Gate: minimum test actionable accuracy.")
    parser.add_argument("--promote-min-test-win-rate", type=float, default=0.52,
                        help="PROVISIONAL. Gate: minimum test trade win rate.")
    parser.add_argument("--promote-min-test-alpha", type=float, default=0.00,
                        help="PROVISIONAL. Gate: minimum test alpha vs buy-and-hold BTC.")
    parser.add_argument("--promote-max-val-test-gap", type=float, default=0.05,
                        help="PROVISIONAL. Gate: maximum |val actionable - test actionable|.")
    parser.add_argument("--promote-max-test-cv", type=float, default=1.0,
                        help="PROVISIONAL. Gate: maximum config-level test return CV.")
    parser.add_argument("--promote-min-test-trade-count", type=int, default=0, help="Optional gate: minimum number of test trades (0 disables).")
    parser.add_argument("--promote-min-test-actionable-support", type=int, default=0, help="Optional gate: minimum test actionable support (0 disables).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    global DEFAULT_DEVICE
    DEFAULT_DEVICE = args.device

    leaderboard = run_experiments(args)
    leaderboard_path = Path(args.leaderboard_path)
    reward_leaderboard_path = Path(args.reward_leaderboard_path)
    summary_path = Path(args.summary_path)
    snapshot_dir = None if args.disable_snapshots else Path(args.snapshot_dir)
    run_label = args.run_label.strip() or None
    canonical_product = resolve_product(args.product)
    interval = str(args.interval)
    reward_leaderboard, summary = write_experiment_outputs(
        leaderboard=leaderboard,
        leaderboard_path=leaderboard_path,
        reward_leaderboard_path=reward_leaderboard_path,
        summary_path=summary_path,
        snapshot_dir=snapshot_dir,
        run_label=run_label,
        append_results=args.append,
    )
    top = leaderboard.head(3)

    # --- Champion Promotion ---
    # Naming convention below is a deliberate, documented carry-over from the stock bot
    # (CLAUDE.md "Known gotchas: legacy sac_trading_bot_*.zip naming, not a bug, do not rename").
    if not leaderboard.empty:
        candidate_rows = leaderboard
        if args.promote_require_gates:
            mask = leaderboard.apply(lambda row: _passes_promotion_gates(row, args), axis=1)
            candidate_rows = leaderboard[mask].sort_values("ranking_score", ascending=False).reset_index(drop=True)
            print(
                "Promotion gates active (PROVISIONAL thresholds): "
                f"min_test_actionable={args.promote_min_test_actionable:.3f}, "
                f"min_test_win_rate={args.promote_min_test_win_rate:.3f}, "
                f"min_test_alpha={args.promote_min_test_alpha:.3f}"
            )

        candidate_products = candidate_rows.get("product", pd.Series(["" for _ in range(len(candidate_rows))]))
        product_matches = candidate_rows[candidate_products.astype(str).str.upper() == canonical_product]
        if not product_matches.empty:
            best_run = product_matches.iloc[0]
            best_product = str(best_run.get("product", "unknown"))
            best_model_path = Path(best_run["model_path"])
            if best_model_path.exists():
                import shutil

                default_model_path = ROOT_DIR / "models" / f"sac_trading_bot_{best_product}_{interval}.zip"
                default_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(best_model_path, default_model_path)
                print(
                    f"Champion promoted to: {default_model_path} "
                    f"(Ranking Score: {best_run['ranking_score']:.4f}, "
                    f"test_actionable={float(best_run.get('test_actionable_accuracy', 0.0)):.4f}, "
                    f"test_win_rate={float(best_run.get('test_trade_win_rate', 0.0)):.4f}, "
                    f"test_alpha_vs_btc={float(best_run.get('test_alpha_vs_btc', 0.0)):.4f}, "
                    f"test_cv={float(best_run.get('test_return_cv_by_config', float('inf'))):.4f}, "
                    f"product={best_product})"
                )
            else:
                print(f"No champion promoted: selected model path missing: {best_model_path}")
        else:
            print(f"No champion promoted: no candidates match product '{canonical_product}'")

    print(f"Saved leaderboard: {leaderboard_path}")
    print(f"Saved reward leaderboard: {reward_leaderboard_path}")
    print(f"Saved summary: {summary_path}")
    if "snapshot_paths" in summary:
        snapshot_paths = summary["snapshot_paths"]
        if isinstance(snapshot_paths, dict):
            print(f"Saved snapshots: {snapshot_paths.get('leaderboard')}")
    if args.compact_output:
        compact_cols = [
            "run_label", "product", "interval", "seed", "min_hold_bars",
            "test_trade_count", "test_trade_win_rate", "test_actionable_accuracy",
            "test_alpha_vs_btc", "ranking_score",
        ]
        available = [c for c in compact_cols if c in top.columns]
        print("Top runs (compact):")
        print(top[available].to_string(index=False))
    else:
        print("Top run (Transposed for readability):")
        print(top.head(1).T.to_string(header=False))


if __name__ == "__main__":
    main()

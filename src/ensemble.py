"""
Multi-seed ensemble inference. Ported unchanged (logic-wise) from the sibling stock bot's
src/ensemble.py -- SparseEnsemble is fully generic over the leaderboard schema and doesn't
know or care whether "seed" rows came from a stock or crypto sweep.

Dropped relative to the source: the optional ExitManager integration (predict_with_exit).
Exit-rule position management is a stock-bot Phase 3 feature not in this project's scope
(see CLAUDE.md build order) -- there is no src/exit_manager.py here, and reintroducing the
hook without the module behind it would be dead code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

_ROOT_DIR = Path(__file__).resolve().parents[1]
_CROSS_OS_ANCHORS = [
    "data/experiment_snapshots/",
    "staging/models/",
    "models/",
    "data/",
]


def _resolve_model_path_cross_os(raw_path: str) -> Optional[Path]:
    """Resolve a model path that may have been written on a different OS or machine."""
    raw_str = raw_path.replace("\\", "/")
    candidate = Path(raw_str)

    if candidate.exists():
        return candidate

    if not candidate.is_absolute():
        rooted = _ROOT_DIR / candidate
        if rooted.exists():
            return rooted

    for anchor in _CROSS_OS_ANCHORS:
        idx = raw_str.find(anchor)
        if idx != -1:
            rooted = _ROOT_DIR / raw_str[idx:]
            if rooted.exists():
                return rooted

    filename = Path(raw_str).name
    if filename:
        for search_dir in [
            _ROOT_DIR / "data" / "experiment_snapshots",
            _ROOT_DIR / "staging" / "models",
            _ROOT_DIR / "models",
        ]:
            if search_dir.exists():
                for match in search_dir.rglob(filename):
                    return match

    return None


class SparseEnsemble:
    """
    Multi-seed ensemble. Loads models based on a leaderboard CSV to automatically filter
    by trades and rank by the configured metric (default: test_sharpe_ratio).
    """

    def __init__(self, leaderboard_csv_path: str, ranking_metric: str = "test_sharpe_ratio"):
        self.leaderboard_path = Path(leaderboard_csv_path)
        self.ranking_metric = ranking_metric
        self.leaderboard = pd.read_csv(self.leaderboard_path)

        required_cols = ["model_path", "test_trade_count", ranking_metric]
        for col in required_cols:
            if col not in self.leaderboard.columns:
                raise ValueError(f"Leaderboard CSV missing required column: {col}")

        self.active_seeds_df = self.leaderboard.copy()
        self.models: Dict[int, object] = {}
        self.top_models_info: list = []

    def filter_active_seeds(self, min_test_trades: int = 20) -> int:
        """Remove collapsed seeds (test_trades < min_test_trades) from ensemble considerations."""
        initial_count = len(self.active_seeds_df)
        self.active_seeds_df = self.active_seeds_df[self.active_seeds_df["test_trade_count"] >= min_test_trades]
        return initial_count - len(self.active_seeds_df)

    def rank_by_metric(self, metric: Optional[str] = None) -> List[Tuple[int, float]]:
        """Return seeds ranked by metric in descending order."""
        rank_col = metric if metric else self.ranking_metric
        ranked = self.active_seeds_df.sort_values(rank_col, ascending=False)
        return list(zip(ranked["seed"], ranked[rank_col]))

    def load_top_n_models(self, n: int = 3, seed_filter=None, run_label_filter=None) -> int:
        """Loads the top N models into memory based on the ranking metric."""
        df = self.active_seeds_df
        if seed_filter is not None:
            df = df[df["seed"].isin(seed_filter)]
        if run_label_filter is not None:
            df = df[df["run_label"] == run_label_filter]
        ranked = df.sort_values(self.ranking_metric, ascending=False)
        ranked = ranked.drop_duplicates(subset=["seed"], keep="first")
        top_n = ranked.head(n)

        self.models = {}
        self.top_models_info = []

        for _, row in top_n.iterrows():
            seed = int(row["seed"])
            model_path = _resolve_model_path_cross_os(str(row["model_path"]))
            if model_path is None:
                raise FileNotFoundError(
                    f"Model file not found: {row['model_path']}\n"
                    f"Searched local project root ({_ROOT_DIR}) and snapshot directories. "
                    f"Ensure the .zip is present under data/experiment_snapshots/ or models/."
                )

            # Robust auto-detection: try SAC first, then MaskablePPO, then PPO.
            try:
                model = SAC.load(model_path)
            except Exception:
                try:
                    try:
                        from sb3_contrib import MaskablePPO
                        model = MaskablePPO.load(model_path)
                    except ImportError:
                        model = PPO.load(model_path)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model at {model_path} as either SAC, MaskablePPO, or PPO: {e2}")

            self.models[seed] = model
            self.top_models_info.append(row)

        return len(self.models)

    def ensemble_predict(self, observation: np.ndarray, method: str = "mean") -> Tuple[int, float]:
        """
        Args:
            observation: Current market state (padded/trimmed per model).
            method: "mean" (continuous avg, default) | "voting" (majority) | "weighted" (by metric)
        Returns:
            action: 0 (Hold/flat), 1 (Buy/long)
            confidence: fraction of ensemble agreeing / weighted probability
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_top_n_models() first.")

        try:
            import random as _random
            _random.seed(42)
            np.random.seed(42)
            import torch as _torch
            _torch.manual_seed(42)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(42)
        except Exception:
            pass

        votes = []
        weights = []

        for info in self.top_models_info:
            seed = int(info["seed"])
            model = self.models[seed]

            model_obs_shape = model.observation_space.shape[0]
            if observation.shape[0] < model_obs_shape:
                padded_obs = np.concatenate([observation, np.zeros(model_obs_shape - observation.shape[0], dtype=np.float32)])
            elif observation.shape[0] > model_obs_shape:
                padded_obs = observation[:model_obs_shape]
            else:
                padded_obs = observation

            action, _ = model.predict(padded_obs, deterministic=True)

            if isinstance(model, PPO):
                action_val = int(action.item() if isinstance(action, np.ndarray) else action)
            else:
                raw = action.item() if isinstance(action, np.ndarray) else float(action)
                action_val = 1 if raw > 0.0 else 0

            votes.append(action_val)
            weights.append(float(info[self.ranking_metric]))

        if method == "voting":
            vote_counts = {0: 0, 1: 0}
            for v in votes:
                vote_counts[v] += 1
            winning_action = 1 if vote_counts[1] > vote_counts[0] else 0
            confidence = vote_counts[winning_action] / len(votes)
            return winning_action, confidence

        elif method == "mean":
            raws = []
            model_types = []
            for info in self.top_models_info:
                seed = int(info["seed"])
                model = self.models[seed]

                model_obs_shape = model.observation_space.shape[0]
                if observation.shape[0] < model_obs_shape:
                    padded_obs = np.concatenate([observation, np.zeros(model_obs_shape - observation.shape[0], dtype=np.float32)])
                elif observation.shape[0] > model_obs_shape:
                    padded_obs = observation[:model_obs_shape]
                else:
                    padded_obs = observation

                action, _ = model.predict(padded_obs, deterministic=True)

                if isinstance(model, PPO):
                    raw = float(action.item() if isinstance(action, np.ndarray) else action)
                    model_types.append("discrete")
                else:
                    raw = action.item() if isinstance(action, np.ndarray) else float(action)
                    model_types.append("continuous")

                raws.append(raw)

            mean_raw = float(np.mean(raws))
            threshold = 0.0 if any(t == "continuous" for t in model_types) else 0.5
            winning_action = 1 if mean_raw > threshold else 0
            confidence = float(np.mean([1.0 if r > threshold else 0.0 for r in raws]))
            return winning_action, confidence

        elif method == "weighted":
            min_weight = min(weights)
            shifted_weights = [w - min_weight + 0.1 for w in weights] if min_weight < 0 else weights
            total_weight = sum(shifted_weights)
            norm_weights = [w / total_weight for w in shifted_weights]

            score_1 = sum(w for v, w in zip(votes, norm_weights) if v == 1)
            score_0 = sum(w for v, w in zip(votes, norm_weights) if v == 0)

            winning_action = 1 if score_1 > score_0 else 0
            confidence = max(score_1, score_0)
            return winning_action, confidence

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def aggregate_metrics(self) -> Dict[str, float]:
        """Return ensemble-level metrics averaged over the loaded top-N seeds."""
        if not self.top_models_info:
            return {}

        df = pd.DataFrame(self.top_models_info)

        metrics = {
            "ensemble_mean_test_sharpe": float(df["test_sharpe_ratio"].mean()) if "test_sharpe_ratio" in df else 0.0,
            "ensemble_mean_test_return": float(df["test_cumulative_signal_return"].mean()) if "test_cumulative_signal_return" in df else 0.0,
            "ensemble_mean_test_accuracy": float(df["test_actionable_accuracy"].mean()) if "test_actionable_accuracy" in df else 0.0,
        }

        if "val_actionable_accuracy" in df and "test_actionable_accuracy" in df:
            metrics["ensemble_mean_val_test_gap"] = float((df["val_actionable_accuracy"] - df["test_actionable_accuracy"]).abs().mean())

        return metrics

"""
Signal quality metrics: truth-label enrichment and accuracy/win-rate/edge computation.

Trimmed port of the sibling stock bot's src/signal_analytics.py -- this repo drops
simulate_agent_signals / simulate_ensemble_signals (dashboard-only helpers requiring
src/trading_agent.py, which is out of scope; see CLAUDE.md, no dashboard work is
planned here) and keeps only what src/experiments.py and scripts/evaluate_sweep.py
actually consume: truth-label enrichment and metrics aggregation. Both are fully
market-agnostic (they operate on a generic {action, price} signal frame), so nothing
crypto-specific changed here beyond dropping the news-feature constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACTION_LABELS = {0: "Hold", 1: "Buy", 2: "Sell"}


@dataclass(frozen=True)
class MetricsSummary:
    overall_accuracy: float
    actionable_accuracy: float
    actionable_support: int
    buy_precision: float
    buy_recall: float
    sell_precision: float
    sell_recall: float
    trade_count: int
    trade_rate: float
    trade_win_rate: float
    mean_trade_edge: float
    cumulative_signal_return: float


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def enrich_with_truth_labels(
    signal_df: pd.DataFrame,
    threshold: float = 0.002,
    horizon_steps: int = 1,
) -> pd.DataFrame:
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    enriched = signal_df.copy()
    future_price = enriched["price"].shift(-horizon_steps)
    horizon_return = (future_price / enriched["price"]) - 1.0
    horizon_return = horizon_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    enriched["horizon_return"] = horizon_return

    true_signal = np.where(
        enriched["horizon_return"] > threshold,
        1,
        np.where(enriched["horizon_return"] < -threshold, 2, 0),
    )
    enriched["true_signal"] = true_signal.astype(int)
    enriched["true_label"] = enriched["true_signal"].map(ACTION_LABELS)
    enriched["is_correct"] = enriched["action"] == enriched["true_signal"]

    trade_mask = enriched["action"].isin([1, 2])
    enriched["trade_edge"] = np.where(
        enriched["action"] == 1,
        enriched["horizon_return"],
        np.where(enriched["action"] == 2, -enriched["horizon_return"], 0.0),
    )
    enriched["trade_win"] = np.where(trade_mask, enriched["trade_edge"] > 0, False)
    return enriched


def compute_metrics(signal_df: pd.DataFrame) -> MetricsSummary:
    overall_accuracy = _safe_div(float(signal_df["is_correct"].sum()), float(len(signal_df)))

    actionable_true = signal_df["true_signal"].isin([1, 2])
    actionable_pred = signal_df["action"].isin([1, 2])
    actionable_rows = signal_df[actionable_true & actionable_pred]
    actionable_support = int(len(actionable_rows))
    actionable_accuracy = _safe_div(
        float((actionable_rows["action"] == actionable_rows["true_signal"]).sum()),
        float(actionable_support),
    )

    predicted_buy = signal_df["action"] == 1
    predicted_sell = signal_df["action"] == 2
    true_buy = signal_df["true_signal"] == 1
    true_sell = signal_df["true_signal"] == 2

    buy_tp = float((predicted_buy & true_buy).sum())
    buy_precision = _safe_div(buy_tp, float(predicted_buy.sum()))
    buy_recall = _safe_div(buy_tp, float(true_buy.sum()))

    sell_tp = float((predicted_sell & true_sell).sum())
    sell_precision = _safe_div(sell_tp, float(predicted_sell.sum()))
    sell_recall = _safe_div(sell_tp, float(true_sell.sum()))

    trades = signal_df[signal_df["action"].isin([1, 2])]
    trade_count = int(len(trades))
    trade_rate = _safe_div(float(trade_count), float(len(signal_df)))
    trade_win_rate = _safe_div(float(trades["trade_win"].sum()), float(trade_count))
    mean_trade_edge = float(trades["trade_edge"].mean()) if trade_count else 0.0

    cumulative_signal_return = float((1.0 + signal_df["trade_edge"]).cumprod().iloc[-1] - 1.0)

    return MetricsSummary(
        overall_accuracy=overall_accuracy,
        actionable_accuracy=actionable_accuracy,
        actionable_support=actionable_support,
        buy_precision=buy_precision,
        buy_recall=buy_recall,
        sell_precision=sell_precision,
        sell_recall=sell_recall,
        trade_count=trade_count,
        trade_rate=trade_rate,
        trade_win_rate=trade_win_rate,
        mean_trade_edge=mean_trade_edge,
        cumulative_signal_return=cumulative_signal_return,
    )


def confusion_matrix(signal_df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(
        signal_df["true_label"],
        signal_df["action_label"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    )
    order = [ACTION_LABELS[0], ACTION_LABELS[1], ACTION_LABELS[2]]
    return matrix.reindex(index=order, columns=order, fill_value=0)

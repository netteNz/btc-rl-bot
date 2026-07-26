"""
scripts/sanity_scan.py

Audits experiment_leaderboard*.csv (and snapshots) for contamination and sanitation
risk: schema drift, out-of-range metrics, duplicate/mixed configs, missing model
artifacts, and orphaned model files. Produces a JSON report, a Markdown summary, and a
quarantine file of rows recommended for deletion -- it never mutates data itself
(--apply execute is a placeholder for a future sanitize_apply.py, matching the sibling
stock bot's contract).

Ported near-verbatim from the sibling stock bot's scripts/sanity_scan.py -- this is
pure pandas/schema validation logic with no market-specific assumptions, so per
CLAUDE.md ("carries over unchanged") the only real change is the "ticker" -> "product"
column rename (a crypto pair like BTC-USD isn't a ticker) and lightly crypto-flavored
run_label family keywords.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCANNER_VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CORE_COLUMNS = {
    "leaderboard_version",
    "product",
    "run_label",
    "seed",
    "timesteps",
    "learning_rate",
    "gamma",
    "ent_coef",
    "threshold",
    "horizon",
    "transaction_cost_rate",
    "trade_penalty",
    "execution_mode",
    "reward_mode",
    "ranking_score",
}

RANGE_CHECKS = {
    "val_overall_accuracy": (0.0, 1.0),
    "val_actionable_accuracy": (0.0, 1.0),
    "val_trade_rate": (0.0, 1.0),
    "val_trade_win_rate": (0.0, 1.0),
    "test_overall_accuracy": (0.0, 1.0),
    "test_actionable_accuracy": (0.0, 1.0),
    "test_trade_rate": (0.0, 1.0),
    "test_trade_win_rate": (0.0, 1.0),
}

NONNEGATIVE_COLUMNS = {
    "seed",
    "timesteps",
    "horizon",
    "val_actionable_support",
    "val_trade_count",
    "test_actionable_support",
    "test_trade_count",
    "run_duration_seconds",
}

CONFIG_ID_COLUMNS = [
    "product",
    "interval",
    "timesteps",
    "seed",
    "learning_rate",
    "gamma",
    "ent_coef",
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
]


@dataclass
class Issue:
    code: str
    severity: str
    action_class: str
    message: str
    artifact: str | None = None
    row_index: int | None = None
    run_label: str | None = None
    extra: dict[str, Any] | None = None
    recommendation: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    try:
        hasher = hashlib.new(algorithm)
        with open(path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return ""


def scan_downstream_dependencies(root_dir: Path) -> list[dict[str, Any]]:
    """Find Python/shell scripts that import or read leaderboard CSVs."""
    dependencies: list[dict[str, Any]] = []

    leaderboard_patterns = [
        "experiment_leaderboard",
        "experiment_reward_leaderboard",
        "leaderboard.csv",
        "data/experiment_",
    ]

    script_extensions = {".py", ".sh", ".ps1"}

    for script_path in root_dir.rglob("*"):
        if script_path.suffix not in script_extensions or script_path.name.startswith("."):
            continue

        try:
            content = script_path.read_text(encoding="utf-8", errors="ignore")

            for pattern in leaderboard_patterns:
                if pattern.lower() in content.lower():
                    dependencies.append({
                        "script_path": str(script_path),
                        "pattern": pattern,
                        "type": "read_leaderboard",
                        "risk": "Script depends on clean leaderboard data; apply may affect results.",
                    })
                    break
        except Exception:
            pass

    return dependencies


def validate_json_artifact(path: Path) -> tuple[bool, str | None]:
    """Validate JSON file well-formedness."""
    if not path.exists():
        return False, "File does not exist"

    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)[:100]}"
    except Exception as e:
        return False, str(e)[:100]


def scan_json_artifacts(root_dir: Path) -> list[dict[str, Any]]:
    """Scan reports/ and summary_*.json for well-formedness."""
    issues: list[dict[str, Any]] = []

    json_paths = list((root_dir / "reports").glob("*.json")) if (root_dir / "reports").exists() else []
    json_paths.extend(root_dir.glob("summary_*.json"))

    for json_path in json_paths:
        valid, error = validate_json_artifact(json_path)
        if not valid:
            issues.append({
                "path": str(json_path),
                "issue": "Corrupted JSON",
                "error": error,
                "action": "Review and fix manually",
            })

    return issues


def is_missing_or_invalid(value: Any) -> bool:
    """Check if a value is missing, NaN, or infinite."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            return True
    except Exception:
        return True
    return False


def scan_snapshot_directory(root_dir: Path) -> tuple[list[Path], list[Issue]]:
    """Scan experiment_snapshots/ for CSV files and validate them."""
    snapshot_dir = root_dir / "data" / "experiment_snapshots"
    issues: list[Issue] = []
    csv_files: list[Path] = []

    if not snapshot_dir.exists():
        issues.append(Issue(
            code="missing_snapshot_dir",
            severity="low",
            action_class="none",
            message="experiment_snapshots directory does not exist.",
            artifact=str(snapshot_dir),
        ))
        return csv_files, issues

    csv_files = list(snapshot_dir.glob("*.csv"))
    return csv_files, issues


def safe_read_csv(path: Path) -> tuple[pd.DataFrame | None, list[Issue]]:
    """Safely read a CSV file with error handling."""
    issues: list[Issue] = []
    if not path.exists():
        issues.append(Issue(
            code="missing_file",
            severity="medium",
            action_class="needs_manual_review",
            message=f"File does not exist: {path}",
            artifact=str(path),
        ))
        return None, issues

    try:
        df = pd.read_csv(path)
        return df, issues
    except Exception as exc:
        issues.append(Issue(
            code="unreadable_csv",
            severity="high",
            action_class="needs_manual_review",
            message=f"Could not read CSV: {exc}",
            artifact=str(path),
        ))
        return None, issues


def classify_run_family(run_label: str) -> str:
    label = (run_label or "").lower()
    if "minhold" in label or "min-hold" in label or "min_hold" in label:
        return "min_hold"
    if "horizon" in label or "4h" in label:
        return "horizon"
    if "feefree" in label or "fee-free" in label or "zerofriction" in label:
        return "fee_free_diagnostic"
    if "ent" in label or "entropy" in label:
        return "entropy"
    if "reward" in label:
        return "reward"
    if not label.strip():
        return "missing"
    return "other"


def build_config_identity(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in CONFIG_ID_COLUMNS if c in df.columns]
    if not cols:
        return pd.Series(["<missing-config-columns>"] * len(df), index=df.index)

    normalized = df[cols].copy()
    for col in cols:
        normalized[col] = normalized[col].astype(str).fillna("")
    return normalized.apply(lambda row: "|".join(f"{k}={row[k]}" for k in cols), axis=1)


def scan_dataframe(df: pd.DataFrame, artifact: Path, root_dir: Path) -> tuple[list[Issue], dict[str, Any]]:
    issues: list[Issue] = []
    stats: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
    }

    missing_core = sorted(EXPECTED_CORE_COLUMNS - set(df.columns))
    if missing_core:
        issues.append(Issue(
            code="missing_core_columns",
            severity="high",
            action_class="needs_manual_review",
            message=f"Missing expected core columns: {missing_core}",
            artifact=str(artifact),
            extra={"missing_columns": missing_core},
        ))

    if df.empty:
        issues.append(Issue(
            code="empty_csv",
            severity="medium",
            action_class="archive_only",
            message="CSV is empty.",
            artifact=str(artifact),
        ))
        return issues, stats

    if "run_label" in df.columns:
        run_families = df["run_label"].fillna("").astype(str).map(classify_run_family)
        family_counts = run_families.value_counts(dropna=False).to_dict()
        stats["run_family_counts"] = family_counts
        non_missing_families = {k for k in family_counts.keys() if k != "missing"}
        if len(non_missing_families) > 1:
            issues.append(Issue(
                code="mixed_run_families",
                severity="high",
                action_class="needs_code_patch",
                message="Artifact mixes multiple experiment families, likely due to shared append behavior.",
                artifact=str(artifact),
                extra={"run_family_counts": family_counts},
            ))

    if "product" in df.columns:
        products = sorted({str(x).strip().upper() for x in df["product"].dropna().tolist() if str(x).strip()})
        stats["products"] = products
        if len(products) > 1:
            issues.append(Issue(
                code="mixed_products",
                severity="high",
                action_class="sanitize_row",
                message=f"Artifact contains multiple products: {products}",
                artifact=str(artifact),
                extra={"products": products},
            ))

    if "leaderboard_version" in df.columns:
        versions = sorted({str(x) for x in df["leaderboard_version"].dropna().tolist()})
        stats["leaderboard_versions"] = versions
        if len(versions) > 1:
            issues.append(Issue(
                code="mixed_leaderboard_versions",
                severity="medium",
                action_class="sanitize_row",
                message=f"Artifact contains multiple leaderboard versions: {versions}",
                artifact=str(artifact),
                extra={"versions": versions},
            ))

    duplicate_rows = int(df.duplicated().sum())
    stats["duplicate_rows"] = duplicate_rows
    if duplicate_rows > 0:
        issues.append(Issue(
            code="duplicate_rows",
            severity="medium",
            action_class="sanitize_row",
            message=f"Found {duplicate_rows} exact duplicate rows.",
            artifact=str(artifact),
            extra={"duplicate_rows": duplicate_rows},
        ))

    config_identity = build_config_identity(df)
    config_dupes = int(config_identity.duplicated().sum())
    stats["duplicate_config_identities"] = config_dupes
    if config_dupes > 0:
        issues.append(Issue(
            code="duplicate_logical_configs",
            severity="medium",
            action_class="sanitize_row",
            message=f"Found {config_dupes} duplicate logical config identities.",
            artifact=str(artifact),
            extra={"duplicate_config_identities": config_dupes},
        ))

    if "run_label" in df.columns:
        run_label_to_configs = (
            pd.DataFrame({"run_label": df["run_label"].fillna("").astype(str), "config_id": config_identity})
            .groupby("run_label")["config_id"]
            .nunique()
        )
        suspicious_labels = run_label_to_configs[run_label_to_configs > 1]
        if not suspicious_labels.empty:
            issues.append(Issue(
                code="run_label_reused_across_configs",
                severity="medium",
                action_class="needs_manual_review",
                message="Some run labels map to multiple distinct config identities.",
                artifact=str(artifact),
                extra={"suspicious_run_labels": suspicious_labels.to_dict()},
            ))

    for col, (low, high) in RANGE_CHECKS.items():
        if col in df.columns:
            bad_mask = pd.to_numeric(df[col], errors="coerce")
            invalid = bad_mask[(bad_mask < low) | (bad_mask > high) | bad_mask.isna()]
            if not invalid.empty:
                for idx in invalid.index[:25]:
                    rec = f"Delete row {int(idx)} from {artifact.name}"
                    if "run_label" in df.columns:
                        rec += f" (run={df.at[idx, 'run_label']})"
                    issues.append(Issue(
                        code="out_of_range_metric",
                        severity="high",
                        action_class="sanitize_row",
                        message=f"{col} is missing or outside [{low}, {high}].",
                        artifact=str(artifact),
                        row_index=int(idx),
                        run_label=str(df.at[idx, "run_label"]) if "run_label" in df.columns else None,
                        extra={"column": col, "value": None if pd.isna(df.at[idx, col]) else str(df.at[idx, col])},
                        recommendation=rec,
                    ))

    for col in NONNEGATIVE_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = values[(values < 0) | values.isna()]
            if not invalid.empty:
                for idx in invalid.index[:25]:
                    rec = f"Delete row {int(idx)} from {artifact.name}"
                    if "run_label" in df.columns:
                        rec += f" (run={df.at[idx, 'run_label']})"
                    issues.append(Issue(
                        code="negative_or_invalid_count",
                        severity="high",
                        action_class="sanitize_row",
                        message=f"{col} is negative or invalid.",
                        artifact=str(artifact),
                        row_index=int(idx),
                        run_label=str(df.at[idx, "run_label"]) if "run_label" in df.columns else None,
                        extra={"column": col, "value": None if pd.isna(df.at[idx, col]) else str(df.at[idx, col])},
                        recommendation=rec,
                    ))

    if "model_path" in df.columns:
        missing_model_rows = 0
        for idx, model_path in df["model_path"].items():
            if pd.isna(model_path) or not str(model_path).strip():
                missing_model_rows += 1
                rec = f"Delete row {int(idx)} from {artifact.name} (missing model_path)"
                issues.append(Issue(
                    code="missing_model_path",
                    severity="high",
                    action_class="sanitize_row",
                    message="Row has empty model_path.",
                    artifact=str(artifact),
                    row_index=int(idx),
                    run_label=str(df.at[idx, "run_label"]) if "run_label" in df.columns else None,
                    recommendation=rec,
                ))
                continue

            resolved = (root_dir / str(model_path)).resolve() if not Path(str(model_path)).is_absolute() else Path(str(model_path))
            if not resolved.exists():
                missing_model_rows += 1
                rec = f"Delete row {int(idx)} from {artifact.name} (model not found at {str(model_path)})"
                issues.append(Issue(
                    code="missing_model_file",
                    severity="high",
                    action_class="sanitize_row",
                    message="model_path points to a missing file.",
                    artifact=str(artifact),
                    row_index=int(idx),
                    run_label=str(df.at[idx, "run_label"]) if "run_label" in df.columns else None,
                    extra={"model_path": str(model_path), "resolved_path": str(resolved)},
                    recommendation=rec,
                ))
        stats["rows_with_missing_model_reference"] = missing_model_rows

    return issues, stats


def scan_orphaned_models(root_dir: Path, referenced_paths: set[Path]) -> list[dict[str, Any]]:
    orphaned: list[dict[str, Any]] = []
    models_dir = root_dir / "models"
    snapshot_dir = root_dir / "data" / "experiment_snapshots"

    candidates: list[Path] = []
    if models_dir.exists():
        candidates.extend(models_dir.rglob("*.zip"))
    if snapshot_dir.exists():
        candidates.extend(snapshot_dir.rglob("*.zip"))

    for path in sorted(set(candidates)):
        if path.resolve() not in referenced_paths:
            orphaned.append({
                "path": str(path),
                "action_class": "safe_delete",
                "message": "Model artifact is not referenced by scanned CSV rows.",
            })

    return orphaned


def collect_referenced_model_paths(dataframes: list[pd.DataFrame], root_dir: Path) -> set[Path]:
    referenced: set[Path] = set()
    for df in dataframes:
        if "model_path" not in df.columns:
            continue
        for value in df["model_path"].dropna().astype(str):
            p = Path(value)
            resolved = (root_dir / p).resolve() if not p.is_absolute() else p.resolve()
            referenced.add(resolved)
    return referenced


def write_markdown_summary(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Sanity Scan Summary")
    lines.append("")
    lines.append(f"- Generated at: {report['scan_metadata']['generated_at_utc']}")
    lines.append(f"- Scanner version: {report['scan_metadata']['scanner_version']}")
    lines.append("")

    summary = report["summary"]
    lines.append("## Counts")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## High-severity issues")
    lines.append("")
    for issue in report["all_issues"]:
        if issue["severity"] == "high":
            lines.append(f"- `{issue['code']}` in `{issue.get('artifact')}`: {issue['message']}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit experiment artifacts for contamination and sanitation risk.")
    parser.add_argument("--root-dir", default=".", help="Repo root.")
    parser.add_argument("--report-json", default="reports/sanity_scan_report.json", help="JSON report output path.")
    parser.add_argument("--report-md", default="reports/sanity_scan_summary.md", help="Markdown summary output path.")
    parser.add_argument("--quarantine-json", default="reports/sanity_quarantine.json", help="Quarantine file output path.")
    parser.add_argument("--apply", default="dry-run", choices=["dry-run", "execute"], help="Dry-run (default) or execute mutations.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    artifact_paths = [
        root_dir / "data" / "experiment_leaderboard.csv",
        root_dir / "data" / "experiment_reward_leaderboard.csv",
        root_dir / "data" / "experiment_leaderboard_history.csv",
        root_dir / "data" / "experiment_reward_leaderboard_history.csv",
    ]

    all_issues: list[Issue] = []
    artifacts_report: list[dict[str, Any]] = []
    loaded_frames: list[pd.DataFrame] = []
    quarantine_rows: list[dict[str, Any]] = []

    for artifact in artifact_paths:
        df, read_issues = safe_read_csv(artifact)
        all_issues.extend(read_issues)

        artifact_entry: dict[str, Any] = {
            "path": str(artifact),
            "kind": "csv_artifact",
            "exists": artifact.exists(),
            "readable": df is not None,
        }

        if df is not None:
            loaded_frames.append(df)
            df_issues, stats = scan_dataframe(df, artifact, root_dir)
            all_issues.extend(df_issues)
            artifact_entry.update(stats)
            artifact_entry["issues"] = [asdict(issue) for issue in df_issues]

            for issue in df_issues:
                if issue.action_class == "sanitize_row" and issue.row_index is not None:
                    quarantine_rows.append({
                        "artifact": str(artifact),
                        "row_index": issue.row_index,
                        "issue_code": issue.code,
                        "reason": issue.message,
                        "run_label": issue.run_label,
                        "recommendation": issue.recommendation,
                        "data_hash": compute_file_hash(artifact),
                    })
        else:
            artifact_entry["issues"] = [asdict(issue) for issue in read_issues]

        artifacts_report.append(artifact_entry)

    referenced_paths = collect_referenced_model_paths(loaded_frames, root_dir)
    orphaned_models = scan_orphaned_models(root_dir, referenced_paths)

    snapshot_csvs, snapshot_issues = scan_snapshot_directory(root_dir)
    all_issues.extend(snapshot_issues)

    snapshot_report: list[dict[str, Any]] = []
    for snapshot_csv in snapshot_csvs:
        df, read_issues = safe_read_csv(snapshot_csv)
        all_issues.extend(read_issues)
        snapshot_entry: dict[str, Any] = {
            "path": str(snapshot_csv),
            "kind": "snapshot_csv",
            "exists": snapshot_csv.exists(),
            "readable": df is not None,
        }
        if df is not None:
            df_issues, stats = scan_dataframe(df, snapshot_csv, root_dir)
            all_issues.extend(df_issues)
            snapshot_entry.update(stats)
            snapshot_entry["issues"] = [asdict(issue) for issue in df_issues]
        else:
            snapshot_entry["issues"] = [asdict(issue) for issue in read_issues]
        snapshot_report.append(snapshot_entry)

    json_validation_issues = scan_json_artifacts(root_dir)
    downstream_deps = scan_downstream_dependencies(root_dir)

    summary = {
        "high_severity_issue_count": sum(1 for i in all_issues if i.severity == "high"),
        "sanitize_row_count": sum(1 for i in all_issues if i.action_class == "sanitize_row"),
        "archive_only_count": sum(1 for i in all_issues if i.action_class == "archive_only"),
        "safe_delete_count": len(orphaned_models) + sum(1 for i in all_issues if i.action_class == "safe_delete"),
        "needs_code_patch_count": sum(1 for i in all_issues if i.action_class == "needs_code_patch"),
        "needs_manual_review_count": sum(1 for i in all_issues if i.action_class == "needs_manual_review"),
        "downstream_dependencies": len(downstream_deps),
        "json_validation_issues": len(json_validation_issues),
        "snapshot_csvs_scanned": len(snapshot_csvs),
        "quarantine_rows": len(quarantine_rows),
    }

    report = {
        "scan_metadata": {
            "generated_at_utc": utc_now(),
            "root_dir": str(root_dir),
            "scanner_version": SCANNER_VERSION,
            "apply_mode": args.apply,
        },
        "artifacts": artifacts_report,
        "snapshots": snapshot_report,
        "orphaned_models": orphaned_models,
        "downstream_dependencies": downstream_deps,
        "json_validation": json_validation_issues,
        "quarantine_rows": quarantine_rows,
        "all_issues": [asdict(issue) for issue in all_issues],
        "summary": summary,
    }

    report_json = Path(args.report_json)
    if not report_json.is_absolute():
        report_json = root_dir / report_json
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report_md = Path(args.report_md)
    if not report_md.is_absolute():
        report_md = root_dir / report_md
    write_markdown_summary(report, report_md)

    quarantine_json = Path(args.quarantine_json)
    if not quarantine_json.is_absolute():
        quarantine_json = root_dir / quarantine_json
    quarantine_json.parent.mkdir(parents=True, exist_ok=True)
    quarantine_json.write_text(json.dumps({
        "metadata": {
            "generated_at_utc": utc_now(),
            "scanner_version": SCANNER_VERSION,
            "apply_mode": args.apply,
        },
        "rows_to_quarantine": quarantine_rows,
    }, indent=2), encoding="utf-8")

    print(f"Saved JSON report: {report_json}")
    print(f"Saved Markdown summary: {report_md}")
    print(f"Saved quarantine file: {quarantine_json}")
    print(f"\nDownstream dependencies: {summary['downstream_dependencies']}")
    print(f"JSON validation issues: {summary['json_validation_issues']}")
    print(f"Snapshot CSVs scanned: {summary['snapshot_csvs_scanned']}")
    print(f"Quarantine rows: {summary['quarantine_rows']}")

    if args.apply == "dry-run":
        print("\n[DRY-RUN MODE] No mutations applied. Run with --apply=execute to sanitize.")
    else:
        print("\n[EXECUTE MODE] Sanitization would be applied by a future sanitize_apply.py.")


if __name__ == "__main__":
    main()

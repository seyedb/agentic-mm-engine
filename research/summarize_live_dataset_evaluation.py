#!/usr/bin/env python3
"""Summarize collected public quote datasets and policy-gate results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATTERN = "data/quotes/*.meta.json"
DEFAULT_DATASET_SUMMARY = Path("target/research/policy_gate_dataset_summary.csv")
DEFAULT_POLICY_SUMMARY = Path("target/research/policy_gate_policy_summary.csv")
DEFAULT_MANIFEST_OUTPUT = Path("target/research/live_dataset_manifest.csv")
DEFAULT_REPORT_OUTPUT = Path("target/research/live_dataset_evaluation.md")


@dataclass(frozen=True)
class Dataset:
    run_id: str
    pair: str
    samples: int
    interval_seconds: float
    started_at: str
    ended_at: str
    csv_path: Path
    evaluated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize live/public quote datasets and policy-gate evaluation outputs."
    )
    parser.add_argument(
        "metadata_paths",
        nargs="*",
        type=Path,
        help="Quote dataset metadata files. Defaults to data/quotes/*.meta.json.",
    )
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--policy-summary", type=Path, default=DEFAULT_POLICY_SUMMARY)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def discover_metadata(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(project_path(path) for path in paths)
    return sorted(Path(path) for path in glob.glob(str(PROJECT_ROOT / DEFAULT_METADATA_PATTERN)))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_dataset(path: Path) -> Dataset:
    metadata = load_json(path)
    csv_value = metadata.get("csv_path")
    if not csv_value:
        raise ValueError(f"{path}: metadata missing csv_path")
    csv_path = project_path(Path(str(csv_value)))
    return Dataset(
        run_id=str(metadata.get("run_id") or path.name.replace(".meta.json", "")),
        pair=str(metadata.get("pair") or ""),
        samples=parse_int(metadata.get("samples", 0), "samples"),
        interval_seconds=parse_float(
            metadata.get("interval_seconds", 0.0),
            "interval_seconds",
        ),
        started_at=str(metadata.get("started_at") or ""),
        ended_at=str(metadata.get("ended_at") or ""),
        csv_path=csv_path,
        evaluated=isinstance(metadata.get("evaluation"), dict),
    )


def parse_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def parse_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    with project_path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        return list(reader)


def row_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in {column}: {row[column]!r}") from exc


def configured_rows(rows: list[dict[str, str]], run_id: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["assumption"] == "configured" and row["run_id"] == run_id
    ]


def best_row(rows: list[dict[str, str]], utility_column: str) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row_float(row, utility_column))


def group_by(rows: list[dict[str, str]], column: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[column], []).append(row)
    return groups


def write_manifest(
    path: Path,
    datasets: list[Dataset],
    dataset_rows: list[dict[str, str]],
) -> None:
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "pair",
                "samples",
                "interval_seconds",
                "started_at",
                "ended_at",
                "csv_path",
                "evaluated",
                "configured_policies",
                "configured_windows",
                "configured_winner",
                "configured_utility",
            ]
        )
        for dataset in datasets:
            rows = configured_rows(dataset_rows, dataset.run_id)
            winner = best_row(rows, "utility")
            writer.writerow(
                [
                    dataset.run_id,
                    dataset.pair,
                    dataset.samples,
                    format_float(dataset.interval_seconds),
                    dataset.started_at,
                    dataset.ended_at,
                    dataset.csv_path,
                    str(dataset.evaluated).lower(),
                    ",".join(sorted({row["policy"] for row in rows})),
                    rows[0]["windows"] if rows else "0",
                    winner["policy"] if winner else "",
                    format_float(row_float(winner, "utility")) if winner else "",
                ]
            )


def render_report(
    datasets: list[Dataset],
    dataset_rows: list[dict[str, str]],
    policy_rows: list[dict[str, str]],
) -> str:
    if not datasets:
        raise ValueError("no datasets to summarize")

    configured = [row for row in policy_rows if row["assumption"] == "configured"]
    configured_winner = best_row(configured, "mean_utility")
    assumption_winners = {
        assumption: best_row(rows, "mean_utility")
        for assumption, rows in group_by(policy_rows, "assumption").items()
    }
    configured_dataset_winners = dataset_winners(dataset_rows)
    configured_by_dataset = group_by(
        [row for row in dataset_rows if row["assumption"] == "configured"],
        "run_id",
    )
    total_windows = sum(int(rows[0]["windows"]) for rows in configured_by_dataset.values())

    lines = [
        "# Live Dataset Evaluation",
        "",
        "## Scope",
        "",
        "This report summarizes public quote datasets that were collected from live top-of-book snapshots and then replayed through the Rust paper engine.",
        "",
        "## Dataset Manifest",
        "",
        f"- Datasets: `{len(datasets)}`.",
        f"- Pairs: `{', '.join(sorted({dataset.pair for dataset in datasets}))}`.",
        f"- Total quote samples: `{sum(dataset.samples for dataset in datasets)}`.",
        f"- Evaluation windows: `{total_windows}`.",
        f"- Snapshot interval range: `{format_float(min(dataset.interval_seconds for dataset in datasets))}` "
        f"to `{format_float(max(dataset.interval_seconds for dataset in datasets))}` seconds.",
        "",
        "| run_id | samples | interval_s | evaluated | configured_winner | utility |",
        "|---|---:|---:|---|---|---:|",
    ]
    for dataset in datasets:
        rows = configured_rows(dataset_rows, dataset.run_id)
        winner = best_row(rows, "utility")
        lines.append(
            "| "
            f"{dataset.run_id} | {dataset.samples} | "
            f"{format_float(dataset.interval_seconds)} | "
            f"{str(dataset.evaluated).lower()} | "
            f"{winner['policy'] if winner else ''} | "
            f"{format_float(row_float(winner, 'utility')) if winner else ''} |"
        )

    lines.extend(
        [
            "",
            "## Policy Result",
            "",
        ]
    )
    if configured_winner:
        lines.append(
            f"- Best configured policy: `{configured_winner['policy']}` "
            f"with mean utility `{format_float(row_float(configured_winner, 'mean_utility'))}`."
        )
    lines.append(
        f"- Configured dataset wins: {', '.join(format_count(policy, count) for policy, count in sorted(configured_dataset_winners.items()))}."
    )
    lines.extend(
        [
            "",
            "| assumption | winner | utility | pnl | drawdown | fills |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for assumption, winner in sorted(assumption_winners.items()):
        if winner is None:
            continue
        lines.append(
            "| "
            f"{assumption} | {winner['policy']} | "
            f"{format_float(row_float(winner, 'mean_utility'))} | "
            f"{format_float(row_float(winner, 'mean_pnl'))} | "
            f"{format_float(row_float(winner, 'mean_drawdown'))} | "
            f"{format_float(row_float(winner, 'mean_fills'))} |"
        )

    lines.extend(
        [
            "",
            "## Configured Policy Table",
            "",
            "| policy | utility | pnl | drawdown | fills | adaptive_step_pct | dataset_wins | window_wins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(configured, key=lambda item: -row_float(item, "mean_utility")):
        lines.append(
            "| "
            f"{row['policy']} | {format_float(row_float(row, 'mean_utility'))} | "
            f"{format_float(row_float(row, 'mean_pnl'))} | "
            f"{format_float(row_float(row, 'mean_drawdown'))} | "
            f"{format_float(row_float(row, 'mean_fills'))} | "
            f"{format_float(row_float(row, 'adaptive_step_pct'))} | "
            f"{row['dataset_utility_wins']} | {row['window_utility_wins']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is closer to live evaluation than the synthetic simulator because the input data comes from collected public quote snapshots.",
            "- It is still a paper replay result: fills are modeled, not exchange-confirmed.",
            "- A policy is interesting only if it remains competitive across fresh datasets and fill assumptions.",
            "- The current result supports keeping the logistic learned selector as the main agentic proof of concept, while treating the bandit as research-only.",
            "",
        ]
    )
    return "\n".join(lines)


def dataset_winners(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    configured_by_dataset = group_by(
        [row for row in rows if row["assumption"] == "configured"],
        "run_id",
    )
    for dataset_rows in configured_by_dataset.values():
        winner = best_row(dataset_rows, "utility")
        if winner:
            counts[winner["policy"]] = counts.get(winner["policy"], 0) + 1
    return counts


def format_count(policy: str, count: int) -> str:
    return f"`{policy}` {count}"


def format_float(value: float) -> str:
    return f"{value:.6f}"


def main() -> int:
    args = parse_args()

    try:
        metadata_paths = discover_metadata(args.metadata_paths)
        if not metadata_paths:
            raise ValueError("no quote-dataset metadata files found")
        datasets = [load_dataset(path) for path in metadata_paths]
        dataset_rows = read_csv(args.dataset_summary)
        policy_rows = read_csv(args.policy_summary)

        write_manifest(args.manifest_output, datasets, dataset_rows)
        report_path = project_path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(datasets, dataset_rows, policy_rows) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {project_path(args.manifest_output)}")
    print(f"wrote {project_path(args.report_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

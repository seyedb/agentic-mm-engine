#!/usr/bin/env python3
"""Build a ledger from collected quote-dataset policy evaluations."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_METADATA_PATTERN = "data/quotes/*.meta.json"
DEFAULT_LEDGER_OUTPUT = Path("target/research/quote_dataset_policy_ledger.csv")
DEFAULT_PAIRS_OUTPUT = Path("target/research/quote_dataset_policy_pairs.csv")
DEFAULT_SUMMARY_OUTPUT = Path("target/research/quote_dataset_policy_summary.md")


@dataclass(frozen=True)
class PolicyRow:
    run_id: str
    pair: str
    started_at: str
    samples: int
    interval_seconds: float
    config: str
    policy: str
    windows: int
    mean_fills: float
    mean_pnl: float
    std_pnl: float
    mean_fees: float
    mean_drawdown: float
    mean_spread: float
    mean_quote_distance: float
    aggregate_output: Path


@dataclass(frozen=True)
class PairRow:
    run_id: str
    pair: str
    started_at: str
    samples: int
    interval_seconds: float
    static_config: str
    adaptive_config: str
    windows: int
    static_pnl: float
    adaptive_pnl: float
    pnl_delta: float
    static_drawdown: float
    adaptive_drawdown: float
    drawdown_delta: float
    static_fills: float
    adaptive_fills: float
    fills_delta: float
    static_fees: float
    adaptive_fees: float
    fees_delta: float
    static_pnl_std: float
    adaptive_pnl_std: float
    pnl_std_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare policy evaluation results across collected quote datasets."
    )
    parser.add_argument(
        "metadata_paths",
        nargs="*",
        type=Path,
        help="Quote-dataset metadata files. Defaults to data/quotes/*.meta.json.",
    )
    parser.add_argument(
        "--ledger-output",
        type=Path,
        default=DEFAULT_LEDGER_OUTPUT,
        help="Per-policy ledger CSV output.",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=DEFAULT_PAIRS_OUTPUT,
        help="Paired static-vs-adaptive CSV output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Markdown summary output.",
    )
    return parser.parse_args()


def discover_metadata(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(Path(path) for path in glob.glob(DEFAULT_METADATA_PATTERN))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    return value


def metadata_str(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "" if value is None else str(value)


def metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata field {key!r} must be an integer: {value!r}") from exc


def metadata_float(metadata: dict[str, Any], key: str) -> float:
    value = metadata.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata field {key!r} must be a number: {value!r}") from exc


def evaluation_output(metadata: dict[str, Any], metadata_path: Path) -> Path | None:
    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    output = evaluation.get("aggregate_output")
    if not output:
        return None
    path = Path(str(output))
    if not path.exists():
        raise ValueError(f"{metadata_path}: evaluation output not found: {path}")
    return path


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in column {column!r}: {row[column]!r}") from exc


def parse_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid integer in column {column!r}: {row[column]!r}") from exc


def read_policy_rows(metadata_path: Path) -> list[PolicyRow]:
    metadata = load_json(metadata_path)
    aggregate_output = evaluation_output(metadata, metadata_path)
    if aggregate_output is None:
        return []

    rows = []
    with aggregate_output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{aggregate_output}: CSV file is empty")

        for row in reader:
            rows.append(
                PolicyRow(
                    run_id=metadata_str(metadata, "run_id"),
                    pair=metadata_str(metadata, "pair"),
                    started_at=metadata_str(metadata, "started_at"),
                    samples=metadata_int(metadata, "samples"),
                    interval_seconds=metadata_float(metadata, "interval_seconds"),
                    config=row["config"],
                    policy=row["policy"],
                    windows=parse_int(row, "windows"),
                    mean_fills=parse_float(row, "mean_fills"),
                    mean_pnl=parse_float(row, "mean_pnl"),
                    std_pnl=parse_float(row, "std_pnl"),
                    mean_fees=parse_float(row, "mean_fees"),
                    mean_drawdown=parse_float(row, "mean_drawdown"),
                    mean_spread=parse_float(row, "mean_spread"),
                    mean_quote_distance=parse_float(row, "mean_quote_distance"),
                    aggregate_output=aggregate_output,
                )
            )
    return rows


def paired_rows(rows: list[PolicyRow]) -> list[PairRow]:
    by_run: dict[str, list[PolicyRow]] = {}
    for row in rows:
        by_run.setdefault(row.run_id, []).append(row)

    pairs = []
    for run_id, group in sorted(by_run.items()):
        static = first_policy(group, "static")
        adaptive = first_policy(group, "adaptive")
        if static is None or adaptive is None:
            continue
        pairs.append(
            PairRow(
                run_id=run_id,
                pair=adaptive.pair,
                started_at=adaptive.started_at,
                samples=adaptive.samples,
                interval_seconds=adaptive.interval_seconds,
                static_config=static.config,
                adaptive_config=adaptive.config,
                windows=min(static.windows, adaptive.windows),
                static_pnl=static.mean_pnl,
                adaptive_pnl=adaptive.mean_pnl,
                pnl_delta=adaptive.mean_pnl - static.mean_pnl,
                static_drawdown=static.mean_drawdown,
                adaptive_drawdown=adaptive.mean_drawdown,
                drawdown_delta=adaptive.mean_drawdown - static.mean_drawdown,
                static_fills=static.mean_fills,
                adaptive_fills=adaptive.mean_fills,
                fills_delta=adaptive.mean_fills - static.mean_fills,
                static_fees=static.mean_fees,
                adaptive_fees=adaptive.mean_fees,
                fees_delta=adaptive.mean_fees - static.mean_fees,
                static_pnl_std=static.std_pnl,
                adaptive_pnl_std=adaptive.std_pnl,
                pnl_std_delta=adaptive.std_pnl - static.std_pnl,
            )
        )
    return pairs


def first_policy(rows: list[PolicyRow], policy: str) -> PolicyRow | None:
    for row in rows:
        if row.policy == policy:
            return row
    return None


def write_policy_rows(path: Path, rows: list[PolicyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "pair",
                "started_at",
                "samples",
                "interval_seconds",
                "config",
                "policy",
                "windows",
                "mean_fills",
                "mean_pnl",
                "std_pnl",
                "mean_fees",
                "mean_drawdown",
                "mean_spread",
                "mean_quote_distance",
                "aggregate_output",
            ]
        )
        for row in rows:
            writer.writerow(policy_row_to_csv(row))


def policy_row_to_csv(row: PolicyRow) -> list[str]:
    return [
        row.run_id,
        row.pair,
        row.started_at,
        str(row.samples),
        format_float(row.interval_seconds),
        row.config,
        row.policy,
        str(row.windows),
        format_float(row.mean_fills),
        format_float(row.mean_pnl),
        format_float(row.std_pnl),
        format_float(row.mean_fees),
        format_float(row.mean_drawdown),
        format_float(row.mean_spread),
        format_float(row.mean_quote_distance),
        str(row.aggregate_output),
    ]


def write_pair_rows(path: Path, rows: list[PairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "pair",
                "started_at",
                "samples",
                "interval_seconds",
                "windows",
                "static_config",
                "adaptive_config",
                "static_pnl",
                "adaptive_pnl",
                "pnl_delta",
                "static_drawdown",
                "adaptive_drawdown",
                "drawdown_delta",
                "static_fills",
                "adaptive_fills",
                "fills_delta",
                "static_fees",
                "adaptive_fees",
                "fees_delta",
                "static_pnl_std",
                "adaptive_pnl_std",
                "pnl_std_delta",
            ]
        )
        for row in rows:
            writer.writerow(pair_row_to_csv(row))


def pair_row_to_csv(row: PairRow) -> list[str]:
    return [
        row.run_id,
        row.pair,
        row.started_at,
        str(row.samples),
        format_float(row.interval_seconds),
        str(row.windows),
        row.static_config,
        row.adaptive_config,
        format_float(row.static_pnl),
        format_float(row.adaptive_pnl),
        format_float(row.pnl_delta),
        format_float(row.static_drawdown),
        format_float(row.adaptive_drawdown),
        format_float(row.drawdown_delta),
        format_float(row.static_fills),
        format_float(row.adaptive_fills),
        format_float(row.fills_delta),
        format_float(row.static_fees),
        format_float(row.adaptive_fees),
        format_float(row.fees_delta),
        format_float(row.static_pnl_std),
        format_float(row.adaptive_pnl_std),
        format_float(row.pnl_std_delta),
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_float(value: float) -> str:
    return f"{value:.6f}"


def render_summary(policy_rows: list[PolicyRow], pair_rows: list[PairRow]) -> str:
    lines = [
        "# Quote Dataset Policy Ledger",
        "",
        "## Overview",
        "",
        f"- Datasets with policy rows: {len({row.run_id for row in policy_rows})}",
        f"- Static/adaptive paired datasets: {len(pair_rows)}",
    ]
    if pair_rows:
        pnl_wins = sum(1 for row in pair_rows if row.pnl_delta > 0.0)
        drawdown_wins = sum(1 for row in pair_rows if row.drawdown_delta < 0.0)
        variance_wins = sum(1 for row in pair_rows if row.pnl_std_delta < 0.0)
        lines.extend(
            [
                f"- Adaptive PnL wins: {pnl_wins}/{len(pair_rows)}",
                f"- Adaptive drawdown reductions: {drawdown_wins}/{len(pair_rows)}",
                f"- Adaptive PnL-std reductions: {variance_wins}/{len(pair_rows)}",
                f"- Mean adaptive-static PnL delta: {format_float(mean([row.pnl_delta for row in pair_rows]))}",
                f"- Mean adaptive-static drawdown delta: {format_float(mean([row.drawdown_delta for row in pair_rows]))}",
                f"- Mean adaptive-static fill delta: {format_float(mean([row.fills_delta for row in pair_rows]))}",
                "",
                "## Paired Results",
                "",
                "| run_id | pair | windows | pnl_delta | drawdown_delta | fills_delta | fees_delta | pnl_std_delta |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in pair_rows:
            lines.append(
                "| "
                f"{row.run_id} | {row.pair} | {row.windows} | "
                f"{format_float(row.pnl_delta)} | {format_float(row.drawdown_delta)} | "
                f"{format_float(row.fills_delta)} | {format_float(row.fees_delta)} | "
                f"{format_float(row.pnl_std_delta)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive `pnl_delta` means adaptive outperformed static on average PnL.",
            "- Negative `drawdown_delta`, `fees_delta`, or `pnl_std_delta` means adaptive reduced that risk/cost measure.",
            "- Treat this as experiment tracking, not proof of strategy quality.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        metadata_paths = discover_metadata(args.metadata_paths)
        if not metadata_paths:
            raise ValueError("no quote-dataset metadata files found")

        policy_rows = []
        for metadata_path in metadata_paths:
            policy_rows.extend(read_policy_rows(metadata_path))
        if not policy_rows:
            raise ValueError("no metadata files contained policy evaluation outputs")

        pair_rows = paired_rows(policy_rows)
        write_policy_rows(args.ledger_output, policy_rows)
        write_pair_rows(args.pairs_output, pair_rows)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(render_summary(policy_rows, pair_rows))
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {args.ledger_output}")
    print(f"wrote {args.pairs_output}")
    print(f"wrote {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

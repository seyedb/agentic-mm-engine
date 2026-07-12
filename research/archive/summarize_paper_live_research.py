#!/usr/bin/env python3
"""Write a compact Markdown summary of live paper research artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNS = Path("target/research/paper_live_runs.csv")
DEFAULT_CALIBRATIONS = Path("target/research/paper_fill_calibrations.csv")
DEFAULT_OUTPUT = Path("target/research/paper_live_summary.md")


@dataclass(frozen=True)
class RunRow:
    run_id: str
    pair: str
    steps: int
    fills: int
    fill_rate: float
    final_pnl: float
    max_drawdown: float
    total_fees: float
    traded_notional: float
    min_inventory: float
    max_inventory: float
    avg_spread: float
    avg_volatility: float
    csv_path: str


@dataclass(frozen=True)
class CalibrationRow:
    run_id: str
    observed_fills: int
    predicted_fills: float
    best_base_intensity: float
    best_distance_decay: float
    best_volatility_boost: float
    mean_nll: float
    fill_count_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize live paper research outputs.")
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUNS,
        help="paper_live_runs.csv path.",
    )
    parser.add_argument(
        "--calibrations",
        type=Path,
        default=DEFAULT_CALIBRATIONS,
        help="paper_fill_calibrations.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown summary output path.",
    )
    return parser.parse_args()


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in column '{column}': {row[column]!r}") from exc


def parse_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid integer in column '{column}': {row[column]!r}") from exc


def read_runs(path: Path) -> list[RunRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV is empty")
        return [
            RunRow(
                run_id=row["run_id"],
                pair=row["pair"],
                steps=parse_int(row, "steps"),
                fills=parse_int(row, "fills"),
                fill_rate=parse_float(row, "fill_rate"),
                final_pnl=parse_float(row, "final_pnl"),
                max_drawdown=parse_float(row, "max_drawdown"),
                total_fees=parse_float(row, "total_fees"),
                traded_notional=parse_float(row, "traded_notional"),
                min_inventory=parse_float(row, "min_inventory"),
                max_inventory=parse_float(row, "max_inventory"),
                avg_spread=parse_float(row, "avg_spread"),
                avg_volatility=parse_float(row, "avg_volatility"),
                csv_path=row["csv_path"],
            )
            for row in reader
        ]


def read_calibrations(path: Path) -> dict[str, CalibrationRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV is empty")
        return {
            row["run_id"]: CalibrationRow(
                run_id=row["run_id"],
                observed_fills=parse_int(row, "observed_fills"),
                predicted_fills=parse_float(row, "predicted_fills"),
                best_base_intensity=parse_float(row, "best_base_intensity"),
                best_distance_decay=parse_float(row, "best_distance_decay"),
                best_volatility_boost=parse_float(row, "best_volatility_boost"),
                mean_nll=parse_float(row, "mean_nll"),
                fill_count_rmse=parse_float(row, "fill_count_rmse"),
            )
            for row in reader
        }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def value_range(values: list[float]) -> tuple[float, float]:
    return (min(values), max(values)) if values else (0.0, 0.0)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_parameter(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def stability_note(calibrations: list[CalibrationRow]) -> str:
    if len(calibrations) < 2:
        return "Only one run is available, so parameter stability cannot be assessed yet."

    bases = {row.best_base_intensity for row in calibrations}
    decays = {row.best_distance_decay for row in calibrations}
    boosts = {row.best_volatility_boost for row in calibrations}
    if len(bases) == len(decays) == len(boosts) == 1:
        return "Best calibration parameters are identical across preserved runs."

    return "Best calibration parameters vary across runs; collect more data before updating defaults."


def render_summary(runs: list[RunRow], calibrations_by_run: dict[str, CalibrationRow]) -> str:
    if not runs:
        raise ValueError("no run rows to summarize")

    calibrations = [
        calibrations_by_run[run.run_id]
        for run in runs
        if run.run_id in calibrations_by_run
    ]
    pnl_low, pnl_high = value_range([run.final_pnl for run in runs])
    drawdown_low, drawdown_high = value_range([run.max_drawdown for run in runs])

    lines = [
        "# Live Paper Research Summary",
        "",
        "## Overview",
        "",
        f"- Runs: {len(runs)}",
        f"- Total steps: {sum(run.steps for run in runs)}",
        f"- Total fills: {sum(run.fills for run in runs)}",
        f"- Average fill-rate: {format_float(mean([run.fill_rate for run in runs]))}",
        f"- Final PnL range: {format_float(pnl_low)} to {format_float(pnl_high)}",
        f"- Max drawdown range: {format_float(drawdown_low)} to {format_float(drawdown_high)}",
        f"- Total fees: {format_float(sum(run.total_fees for run in runs))}",
        "",
        "## Run Results",
        "",
        "| run_id | steps | fills | final_pnl | max_drawdown | fees | inventory_range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for run in runs:
        lines.append(
            "| "
            f"{run.run_id} | {run.steps} | {run.fills} | "
            f"{format_float(run.final_pnl)} | {format_float(run.max_drawdown)} | "
            f"{format_float(run.total_fees)} | "
            f"{format_float(run.min_inventory)} to {format_float(run.max_inventory)} |"
        )

    lines.extend(
        [
            "",
            "## Fill Calibration",
            "",
            "| run_id | observed_fills | predicted_fills | base | decay | vol_boost | mean_nll | rmse |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for run in runs:
        calibration = calibrations_by_run.get(run.run_id)
        if calibration is None:
            lines.append(f"| {run.run_id} |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| "
            f"{run.run_id} | {calibration.observed_fills} | "
            f"{format_float(calibration.predicted_fills)} | "
            f"{format_parameter(calibration.best_base_intensity)} | "
            f"{format_parameter(calibration.best_distance_decay)} | "
            f"{format_parameter(calibration.best_volatility_boost)} | "
            f"{format_float(calibration.mean_nll)} | "
            f"{format_float(calibration.fill_count_rmse)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {stability_note(calibrations)}",
            "- These results are from paper fills, not exchange-confirmed executions.",
            "- Short runs are useful for workflow checks but weak evidence for parameter changes.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        runs = read_runs(args.runs)
        calibrations = read_calibrations(args.calibrations)
        summary = render_summary(runs, calibrations)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

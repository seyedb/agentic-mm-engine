#!/usr/bin/env python3
"""Compare replay sweep results across datasets.

This script reads one or more *_replay_sweep.csv files and writes two compact
research artifacts:

    target/research/replay_sweep_best.csv
    target/research/replay_sweep_parameters.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


OUTPUT_DIR = Path("target/research")
BEST_OUTPUT = OUTPUT_DIR / "replay_sweep_best.csv"
PARAMETER_OUTPUT = OUTPUT_DIR / "replay_sweep_parameters.csv"

REQUIRED_COLUMNS = {
    "rank",
    "spread",
    "skew",
    "quantity",
    "fee_rate",
    "runs",
    "score",
    "score_std",
    "stable_score",
    "avg_final_pnl",
    "avg_max_drawdown",
    "avg_max_abs_inventory",
    "avg_total_fills",
    "avg_total_fees",
    "inactivity_penalty",
}


@dataclass
class SweepRow:
    dataset: str
    rank: int
    spread: float
    skew: float
    quantity: float
    fee_rate: float
    runs: int
    score: float
    score_std: float
    stable_score: float
    avg_final_pnl: float
    avg_max_drawdown: float
    avg_max_abs_inventory: float
    avg_total_fills: float
    avg_total_fees: float
    inactivity_penalty: float

    def parameter_key(self) -> tuple[float, float, float, float]:
        return (self.spread, self.skew, self.quantity, self.fee_rate)


@dataclass
class ParameterAccumulator:
    datasets: set[str] = field(default_factory=set)
    seed_runs: int = 0
    stable_scores: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    final_pnls: list[float] = field(default_factory=list)
    drawdowns: list[float] = field(default_factory=list)
    fills: list[float] = field(default_factory=list)
    fees: list[float] = field(default_factory=list)
    max_inventories: list[float] = field(default_factory=list)

    def update(self, row: SweepRow) -> None:
        self.datasets.add(row.dataset)
        self.seed_runs += row.runs
        self.stable_scores.append(row.stable_score)
        self.scores.append(row.score)
        self.final_pnls.append(row.avg_final_pnl)
        self.drawdowns.append(row.avg_max_drawdown)
        self.fills.append(row.avg_total_fills)
        self.fees.append(row.avg_total_fees)
        self.max_inventories.append(row.avg_max_abs_inventory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare replay sweep CSVs across datasets."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="One or more *_replay_sweep.csv files.",
    )
    return parser.parse_args()


def dataset_name(path: Path) -> str:
    name = path.stem
    suffix = "_replay_sweep"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def parse_float(value: str, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float in column '{column}': {value!r}") from exc


def parse_int(value: str, column: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer in column '{column}': {value!r}") from exc


def parse_row(dataset: str, row: dict[str, str]) -> SweepRow:
    return SweepRow(
        dataset=dataset,
        rank=parse_int(row["rank"], "rank"),
        spread=parse_float(row["spread"], "spread"),
        skew=parse_float(row["skew"], "skew"),
        quantity=parse_float(row["quantity"], "quantity"),
        fee_rate=parse_float(row["fee_rate"], "fee_rate"),
        runs=parse_int(row["runs"], "runs"),
        score=parse_float(row["score"], "score"),
        score_std=parse_float(row["score_std"], "score_std"),
        stable_score=parse_float(row["stable_score"], "stable_score"),
        avg_final_pnl=parse_float(row["avg_final_pnl"], "avg_final_pnl"),
        avg_max_drawdown=parse_float(row["avg_max_drawdown"], "avg_max_drawdown"),
        avg_max_abs_inventory=parse_float(
            row["avg_max_abs_inventory"], "avg_max_abs_inventory"
        ),
        avg_total_fills=parse_float(row["avg_total_fills"], "avg_total_fills"),
        avg_total_fees=parse_float(row["avg_total_fees"], "avg_total_fees"),
        inactivity_penalty=parse_float(row["inactivity_penalty"], "inactivity_penalty"),
    )


def read_sweep(path: Path) -> list[SweepRow]:
    dataset = dataset_name(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: CSV missing required columns: {', '.join(missing)}")

        rows = [parse_row(dataset, row) for row in reader]

    if not rows:
        raise ValueError(f"{path}: CSV contains no replay sweep rows")

    rows.sort(key=lambda row: row.rank)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std_dev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def format_number(value: float) -> str:
    return f"{value:.6f}"


def write_best(rows_by_dataset: dict[str, list[SweepRow]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "spread",
                "skew",
                "quantity",
                "fee_rate",
                "runs",
                "stable_score",
                "score",
                "score_std",
                "avg_final_pnl",
                "avg_max_drawdown",
                "avg_total_fills",
                "avg_total_fees",
                "avg_max_abs_inventory",
            ]
        )

        for dataset in sorted(rows_by_dataset):
            best = rows_by_dataset[dataset][0]
            writer.writerow(
                [
                    best.dataset,
                    format_number(best.spread),
                    format_number(best.skew),
                    format_number(best.quantity),
                    format_number(best.fee_rate),
                    best.runs,
                    format_number(best.stable_score),
                    format_number(best.score),
                    format_number(best.score_std),
                    format_number(best.avg_final_pnl),
                    format_number(best.avg_max_drawdown),
                    format_number(best.avg_total_fills),
                    format_number(best.avg_total_fees),
                    format_number(best.avg_max_abs_inventory),
                ]
            )


def grouped_parameters(rows: list[SweepRow]) -> dict:
    groups: dict[tuple[float, float, float, float], ParameterAccumulator] = defaultdict(
        ParameterAccumulator
    )
    for row in rows:
        groups[row.parameter_key()].update(row)
    return groups


def write_parameters(rows: list[SweepRow], path: Path) -> None:
    groups = grouped_parameters(rows)
    ranked = sorted(
        groups.items(),
        key=lambda item: (mean(item[1].stable_scores), -std_dev(item[1].stable_scores)),
        reverse=True,
    )

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "spread",
                "skew",
                "quantity",
                "fee_rate",
                "datasets",
                "seed_runs",
                "avg_stable_score",
                "stable_score_std",
                "avg_score",
                "avg_final_pnl",
                "avg_max_drawdown",
                "avg_total_fills",
                "avg_total_fees",
                "avg_max_abs_inventory",
            ]
        )

        for (spread, skew, quantity, fee_rate), acc in ranked:
            writer.writerow(
                [
                    format_number(spread),
                    format_number(skew),
                    format_number(quantity),
                    format_number(fee_rate),
                    len(acc.datasets),
                    acc.seed_runs,
                    format_number(mean(acc.stable_scores)),
                    format_number(std_dev(acc.stable_scores)),
                    format_number(mean(acc.scores)),
                    format_number(mean(acc.final_pnls)),
                    format_number(mean(acc.drawdowns)),
                    format_number(mean(acc.fills)),
                    format_number(mean(acc.fees)),
                    format_number(mean(acc.max_inventories)),
                ]
            )


def main() -> int:
    args = parse_args()

    try:
        rows_by_dataset = {dataset_name(path): read_sweep(path) for path in args.csv_paths}
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows = [row for dataset_rows in rows_by_dataset.values() for row in dataset_rows]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_best(rows_by_dataset, BEST_OUTPUT)
    write_parameters(rows, PARAMETER_OUTPUT)

    print(f"read {len(rows_by_dataset)} replay sweeps")
    print(f"wrote {BEST_OUTPUT}")
    print(f"wrote {PARAMETER_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

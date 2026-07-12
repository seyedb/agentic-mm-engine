#!/usr/bin/env python3
"""Analyze replay sweep result CSVs.

The Rust engine produces ranked replay sweep results. This script summarizes
the same CSV by parameter so simple replay experiments are easier to compare.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


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
    rank: int
    spread: float
    skew: float
    quantity: float
    fee_rate: float
    runs: int
    score: float
    score_std: float
    stable_score: float
    quote_distance_penalty: float
    avg_final_pnl: float
    avg_max_drawdown: float
    avg_max_abs_inventory: float
    avg_total_fills: float
    avg_total_fees: float
    avg_observed_quote_steps: float
    avg_quote_distance: float
    inactivity_penalty: float


@dataclass
class Accumulator:
    parameter_sets: int = 0
    seed_runs: int = 0
    score: float = 0.0
    score_std: float = 0.0
    stable_score: float = 0.0
    quote_distance_penalty: float = 0.0
    final_pnl: float = 0.0
    fills: float = 0.0
    fees: float = 0.0
    max_drawdown: float = 0.0
    max_abs_inventory: float = 0.0
    observed_quote_steps: float = 0.0
    quote_distance: float = 0.0
    inactivity_penalty: float = 0.0

    def update(self, row: SweepRow) -> None:
        self.parameter_sets += 1
        self.seed_runs += row.runs
        self.score += row.score
        self.score_std += row.score_std
        self.stable_score += row.stable_score
        self.quote_distance_penalty += row.quote_distance_penalty
        self.final_pnl += row.avg_final_pnl
        self.fills += row.avg_total_fills
        self.fees += row.avg_total_fees
        self.max_drawdown += row.avg_max_drawdown
        self.max_abs_inventory += row.avg_max_abs_inventory
        self.observed_quote_steps += row.avg_observed_quote_steps
        self.quote_distance += row.avg_quote_distance
        self.inactivity_penalty += row.inactivity_penalty

    def avg(self, value: float) -> float:
        return value / self.parameter_sets if self.parameter_sets else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a replay sweep CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to *_replay_sweep.csv")
    return parser.parse_args()


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


def parse_optional_float(row: dict[str, str], column: str) -> float:
    value = row.get(column)
    if value is None or value == "":
        return 0.0
    return parse_float(value, column)


def parse_row(row: dict[str, str]) -> SweepRow:
    return SweepRow(
        rank=parse_int(row["rank"], "rank"),
        spread=parse_float(row["spread"], "spread"),
        skew=parse_float(row["skew"], "skew"),
        quantity=parse_float(row["quantity"], "quantity"),
        fee_rate=parse_float(row["fee_rate"], "fee_rate"),
        runs=parse_int(row["runs"], "runs"),
        score=parse_float(row["score"], "score"),
        score_std=parse_float(row["score_std"], "score_std"),
        stable_score=parse_float(row["stable_score"], "stable_score"),
        quote_distance_penalty=parse_optional_float(row, "quote_distance_penalty"),
        avg_final_pnl=parse_float(row["avg_final_pnl"], "avg_final_pnl"),
        avg_max_drawdown=parse_float(row["avg_max_drawdown"], "avg_max_drawdown"),
        avg_max_abs_inventory=parse_float(
            row["avg_max_abs_inventory"], "avg_max_abs_inventory"
        ),
        avg_total_fills=parse_float(row["avg_total_fills"], "avg_total_fills"),
        avg_total_fees=parse_float(row["avg_total_fees"], "avg_total_fees"),
        avg_observed_quote_steps=parse_optional_float(row, "avg_observed_quote_steps"),
        avg_quote_distance=parse_optional_float(row, "avg_quote_distance"),
        inactivity_penalty=parse_float(row["inactivity_penalty"], "inactivity_penalty"),
    )


def read_rows(path: Path) -> list[SweepRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        rows = [parse_row(row) for row in reader]

    if not rows:
        raise ValueError("CSV contains no replay sweep rows")

    rows.sort(key=lambda row: row.rank)
    return rows


def grouped(rows: list[SweepRow], field: str) -> dict[float, Accumulator]:
    groups: dict[float, Accumulator] = defaultdict(Accumulator)
    for row in rows:
        groups[getattr(row, field)].update(row)
    return groups


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def render_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = [title]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def accumulator_row(label: float, acc: Accumulator) -> list[str]:
    return [
        format_number(label, 4),
        str(acc.parameter_sets),
        str(acc.seed_runs),
        format_number(acc.avg(acc.stable_score), 4),
        format_number(acc.avg(acc.score), 4),
        format_number(acc.avg(acc.score_std), 4),
        format_number(acc.avg(acc.quote_distance_penalty), 4),
        format_number(acc.avg(acc.final_pnl), 4),
        format_number(acc.avg(acc.fills), 2),
        format_number(acc.avg(acc.fees), 4),
        format_number(acc.avg(acc.max_drawdown), 4),
        format_number(acc.avg(acc.max_abs_inventory), 4),
        format_number(acc.avg(acc.observed_quote_steps), 2),
        format_number(acc.avg(acc.quote_distance), 4),
        format_number(acc.avg(acc.inactivity_penalty), 4),
    ]


def grouped_rows(groups: dict[float, Accumulator]) -> list[list[str]]:
    return [
        accumulator_row(label, groups[label])
        for label in sorted(
            groups,
            key=lambda value: groups[value].avg(groups[value].stable_score),
            reverse=True,
        )
    ]


def print_best(row: SweepRow) -> None:
    print("Best result")
    print(
        "spread={spread:.2f} skew={skew:.2f} quantity={quantity:.2f} "
        "runs={runs} stable_score={stable:.4f} avg_score={score:.4f} "
        "score_std={score_std:.4f} quote_distance={quote_distance:.4f} "
        "avg_pnl={pnl:.4f} avg_fills={fills:.2f}".format(
            spread=row.spread,
            skew=row.skew,
            quantity=row.quantity,
            runs=row.runs,
            stable=row.stable_score,
            score=row.score,
            score_std=row.score_std,
            quote_distance=row.avg_quote_distance,
            pnl=row.avg_final_pnl,
            fills=row.avg_total_fills,
        )
    )


def main() -> int:
    args = parse_args()

    try:
        rows = read_rows(args.csv_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: {args.csv_path}")
    print(f"Rows: {len(rows)}")
    print()
    print_best(rows[0])
    print()

    headers = [
        "value",
        "sets",
        "seed_runs",
        "avg_stable",
        "avg_score",
        "avg_score_sd",
        "avg_q_pen",
        "avg_pnl",
        "avg_fills",
        "avg_fees",
        "avg_drawdown",
        "avg_max_inv",
        "avg_q_steps",
        "avg_q_dist",
        "avg_idle",
    ]
    print(render_table("By spread", headers, grouped_rows(grouped(rows, "spread"))))
    print()
    print(render_table("By quantity", headers, grouped_rows(grouped(rows, "quantity"))))
    print()
    print(render_table("By skew", headers, grouped_rows(grouped(rows, "skew"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

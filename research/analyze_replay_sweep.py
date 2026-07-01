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
    "score",
    "final_pnl",
    "max_drawdown",
    "max_abs_inventory",
    "total_fills",
    "total_fees",
    "inactivity_penalty",
}


@dataclass
class SweepRow:
    rank: int
    spread: float
    skew: float
    quantity: float
    fee_rate: float
    score: float
    final_pnl: float
    max_drawdown: float
    max_abs_inventory: float
    total_fills: int
    total_fees: float
    inactivity_penalty: float


@dataclass
class Accumulator:
    runs: int = 0
    score: float = 0.0
    final_pnl: float = 0.0
    fills: int = 0
    fees: float = 0.0
    max_drawdown: float = 0.0
    max_abs_inventory: float = 0.0
    inactivity_penalty: float = 0.0

    def update(self, row: SweepRow) -> None:
        self.runs += 1
        self.score += row.score
        self.final_pnl += row.final_pnl
        self.fills += row.total_fills
        self.fees += row.total_fees
        self.max_drawdown += row.max_drawdown
        self.max_abs_inventory += row.max_abs_inventory
        self.inactivity_penalty += row.inactivity_penalty

    def avg(self, value: float) -> float:
        return value / self.runs if self.runs else 0.0


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


def parse_row(row: dict[str, str]) -> SweepRow:
    return SweepRow(
        rank=parse_int(row["rank"], "rank"),
        spread=parse_float(row["spread"], "spread"),
        skew=parse_float(row["skew"], "skew"),
        quantity=parse_float(row["quantity"], "quantity"),
        fee_rate=parse_float(row["fee_rate"], "fee_rate"),
        score=parse_float(row["score"], "score"),
        final_pnl=parse_float(row["final_pnl"], "final_pnl"),
        max_drawdown=parse_float(row["max_drawdown"], "max_drawdown"),
        max_abs_inventory=parse_float(row["max_abs_inventory"], "max_abs_inventory"),
        total_fills=parse_int(row["total_fills"], "total_fills"),
        total_fees=parse_float(row["total_fees"], "total_fees"),
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
        str(acc.runs),
        format_number(acc.avg(acc.score), 4),
        format_number(acc.avg(acc.final_pnl), 4),
        format_number(acc.avg(acc.fills), 2),
        format_number(acc.avg(acc.fees), 4),
        format_number(acc.avg(acc.max_drawdown), 4),
        format_number(acc.avg(acc.max_abs_inventory), 4),
        format_number(acc.avg(acc.inactivity_penalty), 4),
    ]


def grouped_rows(groups: dict[float, Accumulator]) -> list[list[str]]:
    return [
        accumulator_row(label, groups[label])
        for label in sorted(
            groups,
            key=lambda value: groups[value].avg(groups[value].score),
            reverse=True,
        )
    ]


def print_best(row: SweepRow) -> None:
    print("Best result")
    print(
        "spread={spread:.2f} skew={skew:.2f} quantity={quantity:.2f} "
        "score={score:.4f} pnl={pnl:.4f} fills={fills} drawdown={drawdown:.4f}".format(
            spread=row.spread,
            skew=row.skew,
            quantity=row.quantity,
            score=row.score,
            pnl=row.final_pnl,
            fills=row.total_fills,
            drawdown=row.max_drawdown,
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
        "runs",
        "avg_score",
        "avg_pnl",
        "avg_fills",
        "avg_fees",
        "avg_drawdown",
        "avg_max_inv",
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

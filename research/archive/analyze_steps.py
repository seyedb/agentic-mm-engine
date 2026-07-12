#!/usr/bin/env python3
"""Analyze exported best-strategy step datasets.

The Rust engine remains the source of truth for simulation and strategy logic.
This script is an offline research consumer for the CSV artifacts it produces.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = {
    "seed",
    "regime",
    "spread",
    "inventory",
    "total_fills",
    "fees",
    "adverse_selection",
}

REGIME_ORDER = ["LowVol", "NormalVol", "HighVol"]


@dataclass
class Accumulator:
    steps: int = 0
    fills: int = 0
    fees: float = 0.0
    adverse_selection_cost: float = 0.0
    signed_adverse_move: float = 0.0
    abs_inventory: float = 0.0
    spread: float = 0.0
    observed_quote_steps: int = 0
    bid_distance: float = 0.0
    ask_distance: float = 0.0

    def update(self, row: dict[str, str]) -> None:
        self.steps += 1
        self.fills += parse_int(row["total_fills"], "total_fills")
        self.fees += parse_float(row["fees"], "fees")
        adverse_move = parse_float(row["adverse_selection"], "adverse_selection")
        self.adverse_selection_cost += abs(adverse_move)
        self.signed_adverse_move += adverse_move
        self.abs_inventory += abs(parse_float(row["inventory"], "inventory"))
        self.spread += parse_float(row["spread"], "spread")
        if has_observed_quote_distances(row):
            self.observed_quote_steps += 1
            self.bid_distance += parse_float(
                row["bid_distance_to_observed_ask"], "bid_distance_to_observed_ask"
            )
            self.ask_distance += parse_float(
                row["ask_distance_to_observed_bid"], "ask_distance_to_observed_bid"
            )

    @property
    def fill_rate(self) -> float:
        return self.fills / self.steps if self.steps else 0.0

    @property
    def avg_abs_inventory(self) -> float:
        return self.abs_inventory / self.steps if self.steps else 0.0

    @property
    def avg_spread(self) -> float:
        return self.spread / self.steps if self.steps else 0.0

    @property
    def avg_bid_distance(self) -> float:
        return (
            self.bid_distance / self.observed_quote_steps
            if self.observed_quote_steps
            else 0.0
        )

    @property
    def avg_ask_distance(self) -> float:
        return (
            self.ask_distance / self.observed_quote_steps
            if self.observed_quote_steps
            else 0.0
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Rust mm_engine best-strategy step dataset."
    )
    parser.add_argument("csv_path", type=Path, help="Path to *_best_steps.csv")
    parser.add_argument(
        "--spread-bucket-size",
        type=float,
        default=0.25,
        help="Spread bucket width used in the spread diagnostics.",
    )
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


def has_observed_quote_distances(row: dict[str, str]) -> bool:
    return bool(
        row.get("bid_distance_to_observed_ask")
        and row.get("ask_distance_to_observed_bid")
    )


def spread_bucket(spread: float, bucket_size: float) -> tuple[float, str]:
    if bucket_size <= 0.0 or not math.isfinite(bucket_size):
        raise ValueError("--spread-bucket-size must be a positive finite number")

    lower = math.floor(spread / bucket_size) * bucket_size
    upper = lower + bucket_size
    label = f"{lower:.2f}-{upper:.2f}"
    return lower, label


def read_dataset(path: Path, bucket_size: float) -> tuple[Accumulator, dict, dict, set[str]]:
    by_regime: dict[str, Accumulator] = defaultdict(Accumulator)
    by_spread: dict[tuple[float, str], Accumulator] = defaultdict(Accumulator)
    overall = Accumulator()
    seeds: set[str] = set()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        for row in reader:
            seeds.add(row["seed"])

            overall.update(row)
            by_regime[row["regime"]].update(row)

            spread = parse_float(row["spread"], "spread")
            by_spread[spread_bucket(spread, bucket_size)].update(row)

    return overall, by_regime, by_spread, seeds


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


def accumulator_row(name: str, acc: Accumulator) -> list[str]:
    return [
        name,
        str(acc.steps),
        str(acc.fills),
        format_number(acc.fill_rate, 4),
        format_number(acc.fees),
        format_number(acc.adverse_selection_cost),
        format_number(acc.signed_adverse_move),
        format_number(acc.avg_abs_inventory),
        format_number(acc.avg_spread),
    ]


def quote_distance_row(name: str, acc: Accumulator) -> list[str]:
    return [
        name,
        str(acc.observed_quote_steps),
        str(acc.fills),
        format_number(acc.fill_rate, 4),
        format_number(acc.avg_bid_distance, 4),
        format_number(acc.avg_ask_distance, 4),
    ]


def regime_rows(by_regime: dict[str, Accumulator]) -> list[list[str]]:
    ordered = [regime for regime in REGIME_ORDER if regime in by_regime]
    ordered.extend(sorted(regime for regime in by_regime if regime not in REGIME_ORDER))
    return [accumulator_row(regime, by_regime[regime]) for regime in ordered]


def spread_rows(by_spread: dict[tuple[float, str], Accumulator]) -> list[list[str]]:
    return [
        accumulator_row(label, acc)
        for (_lower, label), acc in sorted(by_spread.items(), key=lambda item: item[0][0])
    ]


def main() -> int:
    args = parse_args()

    try:
        overall, by_regime, by_spread, seeds = read_dataset(
            args.csv_path, args.spread_bucket_size
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: {args.csv_path}")
    print(f"Rows: {overall.steps}")
    print(f"Seeds: {len(seeds)}")
    print(f"Total fills: {overall.fills}")
    print(f"Total fees: {format_number(overall.fees)}")
    print(f"Adverse selection cost: {format_number(overall.adverse_selection_cost)}")
    print(f"Signed adverse move: {format_number(overall.signed_adverse_move)}")
    print()

    headers = [
        "bucket",
        "steps",
        "fills",
        "fill_rate",
        "fees",
        "adv_cost",
        "signed_adv",
        "avg_abs_inv",
        "avg_spread",
    ]
    print(render_table("By regime", headers, regime_rows(by_regime)))
    print()
    print(render_table("By spread bucket", headers, spread_rows(by_spread)))
    if overall.observed_quote_steps:
        quote_headers = [
            "bucket",
            "quote_steps",
            "fills",
            "fill_rate",
            "avg_bid_dist",
            "avg_ask_dist",
        ]
        print()
        print(render_table("By quote distance", quote_headers, [quote_distance_row("overall", overall)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

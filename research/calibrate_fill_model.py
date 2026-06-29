#!/usr/bin/env python3
"""Estimate empirical fill intensity from exported step datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "seed",
    "regime",
    "spread",
    "estimated_volatility",
    "total_fills",
    "fees",
    "adverse_selection",
}

REGIME_ORDER = ["LowVol", "NormalVol", "HighVol"]


@dataclass
class CalibrationStats:
    steps: int = 0
    fill_steps: int = 0
    fills: int = 0
    fees: float = 0.0
    adverse_selection_cost: float = 0.0
    signed_adverse_move: float = 0.0
    spread: float = 0.0
    estimated_volatility: float = 0.0

    def update(self, row: dict[str, str]) -> None:
        total_fills = parse_int(row["total_fills"], "total_fills")
        adverse_move = parse_float(row["adverse_selection"], "adverse_selection")

        self.steps += 1
        self.fill_steps += int(total_fills > 0)
        self.fills += total_fills
        self.fees += parse_float(row["fees"], "fees")
        self.adverse_selection_cost += abs(adverse_move)
        self.signed_adverse_move += adverse_move
        self.spread += parse_float(row["spread"], "spread")
        self.estimated_volatility += parse_float(
            row["estimated_volatility"], "estimated_volatility"
        )

    @property
    def fill_probability(self) -> float:
        return self.fill_steps / self.steps if self.steps else 0.0

    @property
    def fill_intensity(self) -> float:
        return self.fills / self.steps if self.steps else 0.0

    @property
    def avg_fills_when_filled(self) -> float:
        return self.fills / self.fill_steps if self.fill_steps else 0.0

    @property
    def fees_per_fill(self) -> float:
        return self.fees / self.fills if self.fills else 0.0

    @property
    def adverse_selection_cost_per_fill(self) -> float:
        return self.adverse_selection_cost / self.fills if self.fills else 0.0

    @property
    def avg_spread(self) -> float:
        return self.spread / self.steps if self.steps else 0.0

    @property
    def avg_estimated_volatility(self) -> float:
        return self.estimated_volatility / self.steps if self.steps else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "fill_steps": self.fill_steps,
            "fills": self.fills,
            "fill_probability": self.fill_probability,
            "fill_intensity": self.fill_intensity,
            "avg_fills_when_filled": self.avg_fills_when_filled,
            "fees": self.fees,
            "fees_per_fill": self.fees_per_fill,
            "adverse_selection_cost": self.adverse_selection_cost,
            "adverse_selection_cost_per_fill": self.adverse_selection_cost_per_fill,
            "signed_adverse_move": self.signed_adverse_move,
            "avg_spread": self.avg_spread,
            "avg_estimated_volatility": self.avg_estimated_volatility,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate empirical fill intensity from a step dataset."
    )
    parser.add_argument("csv_path", type=Path, help="Path to *_best_steps.csv")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the calibration JSON report.",
    )
    parser.add_argument(
        "--spread-bucket-size",
        type=float,
        default=0.25,
        help="Spread bucket width used for conditional estimates.",
    )
    parser.add_argument(
        "--volatility-bucket-size",
        type=float,
        default=0.05,
        help="Estimated-volatility bucket width used for conditional estimates.",
    )
    return parser.parse_args()


def parse_float(value: str, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float in column '{column}': {value!r}") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"non-finite float in column '{column}': {value!r}")
    return parsed


def parse_int(value: str, column: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer in column '{column}': {value!r}") from exc


def numeric_bucket(value: float, bucket_size: float, flag_name: str) -> tuple[float, float, str]:
    if bucket_size <= 0.0 or not math.isfinite(bucket_size):
        raise ValueError(f"{flag_name} must be a positive finite number")

    lower = math.floor(value / bucket_size) * bucket_size
    upper = lower + bucket_size
    return lower, upper, f"{lower:.2f}-{upper:.2f}"


def default_output_path(csv_path: Path) -> Path:
    return Path("target/research") / f"{csv_path.stem}_fill_calibration.json"


def read_dataset(
    path: Path, spread_bucket_size: float, volatility_bucket_size: float
) -> tuple[CalibrationStats, dict, dict, dict, set[str]]:
    overall = CalibrationStats()
    by_regime: dict[str, CalibrationStats] = defaultdict(CalibrationStats)
    by_spread: dict[tuple[float, float, str], CalibrationStats] = defaultdict(
        CalibrationStats
    )
    by_volatility: dict[tuple[float, float, str], CalibrationStats] = defaultdict(
        CalibrationStats
    )
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

            spread = parse_float(row["spread"], "spread")
            volatility = parse_float(row["estimated_volatility"], "estimated_volatility")

            overall.update(row)
            by_regime[row["regime"]].update(row)
            by_spread[
                numeric_bucket(spread, spread_bucket_size, "--spread-bucket-size")
            ].update(row)
            by_volatility[
                numeric_bucket(
                    volatility, volatility_bucket_size, "--volatility-bucket-size"
                )
            ].update(row)

    return overall, by_regime, by_spread, by_volatility, seeds


def ordered_regimes(by_regime: dict[str, CalibrationStats]) -> list[tuple[str, CalibrationStats]]:
    ordered = [regime for regime in REGIME_ORDER if regime in by_regime]
    ordered.extend(sorted(regime for regime in by_regime if regime not in REGIME_ORDER))
    return [(regime, by_regime[regime]) for regime in ordered]


def ordered_buckets(
    buckets: dict[tuple[float, float, str], CalibrationStats],
) -> list[tuple[float, float, str, CalibrationStats]]:
    return [
        (lower, upper, label, stats)
        for (lower, upper, label), stats in sorted(
            buckets.items(), key=lambda item: item[0][0]
        )
    ]


def build_report(
    source: Path,
    spread_bucket_size: float,
    volatility_bucket_size: float,
    overall: CalibrationStats,
    by_regime: dict[str, CalibrationStats],
    by_spread: dict[tuple[float, float, str], CalibrationStats],
    by_volatility: dict[tuple[float, float, str], CalibrationStats],
    seeds: set[str],
) -> dict[str, Any]:
    return {
        "source": str(source),
        "rows": overall.steps,
        "seeds": sorted(seeds),
        "bucket_config": {
            "spread_bucket_size": spread_bucket_size,
            "volatility_bucket_size": volatility_bucket_size,
        },
        "overall": overall.as_dict(),
        "by_regime": {
            regime: stats.as_dict() for regime, stats in ordered_regimes(by_regime)
        },
        "by_spread_bucket": [
            {
                "bucket": label,
                "lower": lower,
                "upper": upper,
                **stats.as_dict(),
            }
            for lower, upper, label, stats in ordered_buckets(by_spread)
        ],
        "by_volatility_bucket": [
            {
                "bucket": label,
                "lower": lower,
                "upper": upper,
                **stats.as_dict(),
            }
            for lower, upper, label, stats in ordered_buckets(by_volatility)
        ],
    }


def format_number(value: float, decimals: int = 4) -> str:
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


def stats_row(name: str, stats: CalibrationStats) -> list[str]:
    return [
        name,
        str(stats.steps),
        str(stats.fill_steps),
        str(stats.fills),
        format_number(stats.fill_probability),
        format_number(stats.fill_intensity),
        format_number(stats.avg_spread),
        format_number(stats.avg_estimated_volatility),
        format_number(stats.adverse_selection_cost_per_fill),
    ]


def main() -> int:
    args = parse_args()
    output_path = args.output or default_output_path(args.csv_path)

    try:
        overall, by_regime, by_spread, by_volatility, seeds = read_dataset(
            args.csv_path, args.spread_bucket_size, args.volatility_bucket_size
        )
        report = build_report(
            args.csv_path,
            args.spread_bucket_size,
            args.volatility_bucket_size,
            overall,
            by_regime,
            by_spread,
            by_volatility,
            seeds,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    headers = [
        "bucket",
        "steps",
        "fill_steps",
        "fills",
        "fill_prob",
        "fill_int",
        "avg_spread",
        "avg_vol",
        "adv_per_fill",
    ]

    print(f"Dataset: {args.csv_path}")
    print(f"Calibration JSON: {output_path}")
    print(f"Rows: {overall.steps}")
    print(f"Seeds: {len(seeds)}")
    print()
    print(render_table("Overall", headers, [stats_row("all", overall)]))
    print()
    print(
        render_table(
            "By regime",
            headers,
            [stats_row(regime, stats) for regime, stats in ordered_regimes(by_regime)],
        )
    )
    print()
    print(
        render_table(
            "By spread bucket",
            headers,
            [
                stats_row(label, stats)
                for _lower, _upper, label, stats in ordered_buckets(by_spread)
            ],
        )
    )
    print()
    print(
        render_table(
            "By volatility bucket",
            headers,
            [
                stats_row(label, stats)
                for _lower, _upper, label, stats in ordered_buckets(by_volatility)
            ],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit touch-intensity paper fill parameters to paper-session logs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("target/research")
REQUIRED_COLUMNS = {
    "timestamp_ms",
    "observed_bid",
    "observed_ask",
    "estimated_volatility",
    "bid",
    "ask",
    "fills",
    "buy_fills",
    "sell_fills",
}


@dataclass(frozen=True)
class PaperFillRow:
    observed_bid: float
    observed_ask: float
    estimated_volatility: float
    bid: float
    ask: float
    fills: int
    buy_fills: int
    sell_fills: int


@dataclass(frozen=True)
class Candidate:
    base_intensity: float
    distance_decay: float
    volatility_boost: float


@dataclass(frozen=True)
class CalibrationResult:
    candidate: Candidate
    nll: float
    fill_count_rmse: float
    observed_fills: int
    predicted_fills: float
    observed_buy_fills: int
    predicted_buy_fills: float
    observed_sell_fills: int
    predicted_sell_fills: float
    steps: int

    @property
    def mean_nll(self) -> float:
        return self.nll / (2 * self.steps) if self.steps else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate touch-intensity paper fill parameters from a paper CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to a paper-session CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: target/research/<csv_stem>_paper_fill_calibration.csv",
    )
    parser.add_argument(
        "--base-intensities",
        default="0.05,0.1,0.2,0.35,0.5,0.75,1.0",
        help="Comma-separated base intensity grid.",
    )
    parser.add_argument(
        "--distance-decays",
        default="20,40,80,120,160,240,320",
        help="Comma-separated distance decay grid.",
    )
    parser.add_argument(
        "--volatility-boosts",
        default="0,1,2,5",
        help="Comma-separated volatility boost grid.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of ranked rows to print.",
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


def parse_grid(value: str, name: str) -> list[float]:
    values = [parse_float(part.strip(), name) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} grid must not be empty")
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} grid values must be non-negative")
    return values


def read_rows(path: Path) -> list[PaperFillRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        rows = []
        for row in reader:
            if row["observed_bid"] == "" or row["observed_ask"] == "":
                continue
            rows.append(
                PaperFillRow(
                    observed_bid=parse_float(row["observed_bid"], "observed_bid"),
                    observed_ask=parse_float(row["observed_ask"], "observed_ask"),
                    estimated_volatility=parse_float(
                        row["estimated_volatility"], "estimated_volatility"
                    ),
                    bid=parse_float(row["bid"], "bid"),
                    ask=parse_float(row["ask"], "ask"),
                    fills=parse_int(row["fills"], "fills"),
                    buy_fills=parse_int(row["buy_fills"], "buy_fills"),
                    sell_fills=parse_int(row["sell_fills"], "sell_fills"),
                )
            )

    if not rows:
        raise ValueError("CSV contains no rows with observed bid/ask quotes")
    return rows


def default_output_path(csv_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{csv_path.stem}_paper_fill_calibration.csv"


def probability(distance: float, volatility: float, candidate: Candidate) -> float:
    volatility_multiplier = max(1.0 + candidate.volatility_boost * volatility, 0.0)
    value = (
        candidate.base_intensity
        * math.exp(-candidate.distance_decay * distance)
        * volatility_multiplier
    )
    return min(max(value, 0.0), 1.0)


def side_probabilities(row: PaperFillRow, candidate: Candidate) -> tuple[float, float]:
    if row.bid >= row.observed_ask:
        buy_probability = 1.0
    else:
        bid_distance = max(row.observed_bid - row.bid, 0.0)
        buy_probability = probability(
            bid_distance, row.estimated_volatility, candidate
        )

    if row.ask <= row.observed_bid:
        sell_probability = 1.0
    else:
        ask_distance = max(row.ask - row.observed_ask, 0.0)
        sell_probability = probability(
            ask_distance, row.estimated_volatility, candidate
        )

    return buy_probability, sell_probability


def bernoulli_nll(observed: bool, predicted: float) -> float:
    epsilon = 1e-12
    p = min(max(predicted, epsilon), 1.0 - epsilon)
    return -math.log(p if observed else 1.0 - p)


def evaluate(rows: list[PaperFillRow], candidate: Candidate) -> CalibrationResult:
    nll = 0.0
    squared_fill_count_error = 0.0
    predicted_fills = 0.0
    predicted_buy_fills = 0.0
    predicted_sell_fills = 0.0
    observed_fills = 0
    observed_buy_fills = 0
    observed_sell_fills = 0

    for row in rows:
        buy_probability, sell_probability = side_probabilities(row, candidate)
        expected_fills = buy_probability + sell_probability

        nll += bernoulli_nll(row.buy_fills > 0, buy_probability)
        nll += bernoulli_nll(row.sell_fills > 0, sell_probability)
        squared_fill_count_error += (row.fills - expected_fills) ** 2

        predicted_fills += expected_fills
        predicted_buy_fills += buy_probability
        predicted_sell_fills += sell_probability
        observed_fills += row.fills
        observed_buy_fills += row.buy_fills
        observed_sell_fills += row.sell_fills

    return CalibrationResult(
        candidate=candidate,
        nll=nll,
        fill_count_rmse=math.sqrt(squared_fill_count_error / len(rows)),
        observed_fills=observed_fills,
        predicted_fills=predicted_fills,
        observed_buy_fills=observed_buy_fills,
        predicted_buy_fills=predicted_buy_fills,
        observed_sell_fills=observed_sell_fills,
        predicted_sell_fills=predicted_sell_fills,
        steps=len(rows),
    )


def calibrate(
    rows: list[PaperFillRow],
    base_intensities: list[float],
    distance_decays: list[float],
    volatility_boosts: list[float],
) -> list[CalibrationResult]:
    results = []
    for base_intensity in base_intensities:
        for distance_decay in distance_decays:
            for volatility_boost in volatility_boosts:
                results.append(
                    evaluate(
                        rows,
                        Candidate(
                            base_intensity=base_intensity,
                            distance_decay=distance_decay,
                            volatility_boost=volatility_boost,
                        ),
                    )
                )

    return sorted(results, key=lambda result: (result.nll, result.fill_count_rmse))


def write_results(path: Path, results: list[CalibrationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "base_intensity",
                "distance_decay",
                "volatility_boost",
                "nll",
                "mean_nll",
                "fill_count_rmse",
                "steps",
                "observed_fills",
                "predicted_fills",
                "observed_buy_fills",
                "predicted_buy_fills",
                "observed_sell_fills",
                "predicted_sell_fills",
            ]
        )
        for rank, result in enumerate(results, start=1):
            writer.writerow(csv_row(rank, result))


def csv_row(rank: int, result: CalibrationResult) -> list[str]:
    candidate = result.candidate
    return [
        str(rank),
        format_float(candidate.base_intensity),
        format_float(candidate.distance_decay),
        format_float(candidate.volatility_boost),
        format_float(result.nll),
        format_float(result.mean_nll),
        format_float(result.fill_count_rmse),
        str(result.steps),
        str(result.observed_fills),
        format_float(result.predicted_fills),
        str(result.observed_buy_fills),
        format_float(result.predicted_buy_fills),
        str(result.observed_sell_fills),
        format_float(result.predicted_sell_fills),
    ]


def format_float(value: float) -> str:
    return f"{value:.6f}"


def render_table(results: list[CalibrationResult], top: int) -> str:
    headers = [
        "rank",
        "base",
        "decay",
        "vol_boost",
        "mean_nll",
        "rmse",
        "obs",
        "pred",
    ]
    rows = []
    for rank, result in enumerate(results[:top], start=1):
        candidate = result.candidate
        rows.append(
            [
                str(rank),
                f"{candidate.base_intensity:.2f}",
                f"{candidate.distance_decay:.1f}",
                f"{candidate.volatility_boost:.1f}",
                f"{result.mean_nll:.4f}",
                f"{result.fill_count_rmse:.4f}",
                str(result.observed_fills),
                f"{result.predicted_fills:.2f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = ["Top paper fill calibration results"]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        rows = read_rows(args.csv_path)
        base_intensities = parse_grid(args.base_intensities, "--base-intensities")
        distance_decays = parse_grid(args.distance_decays, "--distance-decays")
        volatility_boosts = parse_grid(args.volatility_boosts, "--volatility-boosts")
        output = args.output or default_output_path(args.csv_path)

        results = calibrate(rows, base_intensities, distance_decays, volatility_boosts)
        write_results(output, results)

        print(render_table(results, args.top))
        print(f"\nwrote {output}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

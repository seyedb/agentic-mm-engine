#!/usr/bin/env python3
"""Compare touch-intensity calibration results across live paper runs."""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import calibrate_paper_fill_model as calibration


DEFAULT_INPUT_PATTERN = "target/reports/paper_live/*.csv"
DEFAULT_OUTPUT = Path("target/research/paper_fill_calibrations.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate preserved live paper runs and compare best parameters."
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help="Live paper CSVs. Defaults to target/reports/paper_live/*.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output comparison CSV.",
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
    return parser.parse_args()


def discover_inputs(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(Path(path) for path in glob.glob(DEFAULT_INPUT_PATTERN))


def run_id(path: Path) -> str:
    return path.stem


def calibrate_run(
    path: Path,
    base_intensities: list[float],
    distance_decays: list[float],
    volatility_boosts: list[float],
) -> calibration.CalibrationResult:
    rows = calibration.read_rows(path)
    return calibration.calibrate(
        rows, base_intensities, distance_decays, volatility_boosts
    )[0]


def write_comparison(
    path: Path,
    rows: list[tuple[str, Path, calibration.CalibrationResult]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "steps",
                "observed_fills",
                "predicted_fills",
                "observed_buy_fills",
                "predicted_buy_fills",
                "observed_sell_fills",
                "predicted_sell_fills",
                "best_base_intensity",
                "best_distance_decay",
                "best_volatility_boost",
                "mean_nll",
                "fill_count_rmse",
                "csv_path",
            ]
        )
        for run_id_value, csv_path, result in rows:
            writer.writerow(csv_row(run_id_value, csv_path, result))


def csv_row(
    run_id_value: str,
    csv_path: Path,
    result: calibration.CalibrationResult,
) -> list[str]:
    candidate = result.candidate
    return [
        run_id_value,
        str(result.steps),
        str(result.observed_fills),
        calibration.format_float(result.predicted_fills),
        str(result.observed_buy_fills),
        calibration.format_float(result.predicted_buy_fills),
        str(result.observed_sell_fills),
        calibration.format_float(result.predicted_sell_fills),
        calibration.format_float(candidate.base_intensity),
        calibration.format_float(candidate.distance_decay),
        calibration.format_float(candidate.volatility_boost),
        calibration.format_float(result.mean_nll),
        calibration.format_float(result.fill_count_rmse),
        str(csv_path),
    ]


def render_table(rows: list[tuple[str, Path, calibration.CalibrationResult]]) -> str:
    headers = [
        "run_id",
        "steps",
        "obs",
        "pred",
        "base",
        "decay",
        "vol_boost",
        "mean_nll",
        "rmse",
    ]
    table_rows = []
    for run_id_value, _, result in rows:
        candidate = result.candidate
        table_rows.append(
            [
                run_id_value,
                str(result.steps),
                str(result.observed_fills),
                f"{result.predicted_fills:.2f}",
                f"{candidate.base_intensity:.2f}",
                f"{candidate.distance_decay:.1f}",
                f"{candidate.volatility_boost:.1f}",
                f"{result.mean_nll:.4f}",
                f"{result.fill_count_rmse:.4f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = ["Paper fill calibration comparison"]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in table_rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        inputs = discover_inputs(args.csv_paths)
        if not inputs:
            raise ValueError("no live paper CSVs found")

        base_intensities = calibration.parse_grid(
            args.base_intensities, "--base-intensities"
        )
        distance_decays = calibration.parse_grid(
            args.distance_decays, "--distance-decays"
        )
        volatility_boosts = calibration.parse_grid(
            args.volatility_boosts, "--volatility-boosts"
        )
        rows = [
            (
                run_id(path),
                path,
                calibrate_run(path, base_intensities, distance_decays, volatility_boosts),
            )
            for path in inputs
        ]
        rows.sort(key=lambda row: row[0])
        write_comparison(args.output, rows)

        print(render_table(rows))
        print(f"\nwrote {args.output}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

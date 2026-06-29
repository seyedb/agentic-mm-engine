#!/usr/bin/env python3
"""Compare fill-calibration reports across experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PATTERN = "target/research/*_fill_calibration.json"
DEFAULT_OUTPUT = Path("target/research/fill_calibration_comparison.csv")
REGIME_ORDER = ["LowVol", "NormalVol", "HighVol"]


@dataclass
class ComparisonRow:
    experiment: str
    rows: int
    seeds: int
    fill_probability: float
    fill_intensity: float
    avg_spread: float
    avg_volatility: float
    adverse_selection_cost_per_fill: float
    low_vol_fill_intensity: float | None
    normal_vol_fill_intensity: float | None
    high_vol_fill_intensity: float | None
    spread_response: float | None
    volatility_response: float | None

    def as_csv_row(self) -> dict[str, str]:
        return {
            "experiment": self.experiment,
            "rows": str(self.rows),
            "seeds": str(self.seeds),
            "fill_probability": format_float(self.fill_probability),
            "fill_intensity": format_float(self.fill_intensity),
            "avg_spread": format_float(self.avg_spread),
            "avg_volatility": format_float(self.avg_volatility),
            "adverse_selection_cost_per_fill": format_float(
                self.adverse_selection_cost_per_fill
            ),
            "low_vol_fill_intensity": format_optional(self.low_vol_fill_intensity),
            "normal_vol_fill_intensity": format_optional(
                self.normal_vol_fill_intensity
            ),
            "high_vol_fill_intensity": format_optional(self.high_vol_fill_intensity),
            "spread_response": format_optional(self.spread_response),
            "volatility_response": format_optional(self.volatility_response),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare empirical fill calibration reports."
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="Calibration JSON reports. Defaults to target/research/*_fill_calibration.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path for the comparison summary.",
    )
    parser.add_argument(
        "--min-bucket-steps",
        type=int,
        default=100,
        help="Minimum steps required for a bucket to be used in response estimates.",
    )
    return parser.parse_args()


def discover_reports(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(Path(path) for path in glob.glob(DEFAULT_PATTERN))


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    for field in ["source", "rows", "seeds", "overall"]:
        if field not in report:
            raise ValueError(f"{path}: missing required field '{field}'")

    return report


def experiment_name(report: dict[str, Any], fallback_path: Path) -> str:
    source = str(report.get("source", ""))
    if source:
        name = Path(source).stem
    else:
        name = fallback_path.stem.replace("_fill_calibration", "")

    if name.endswith("_best_steps"):
        return name.removesuffix("_best_steps")
    return name


def number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"field '{field}' must be numeric")
    return float(value)


def optional_regime_fill_intensity(report: dict[str, Any], regime: str) -> float | None:
    by_regime = report.get("by_regime", {})
    if not isinstance(by_regime, dict) or regime not in by_regime:
        return None
    regime_stats = by_regime[regime]
    if not isinstance(regime_stats, dict):
        return None
    value = regime_stats.get("fill_intensity")
    return number(value, f"by_regime.{regime}.fill_intensity")


def bucket_response(
    report: dict[str, Any], field: str, min_bucket_steps: int
) -> float | None:
    buckets = report.get(field, [])
    if not isinstance(buckets, list) or len(buckets) < 2:
        return None

    eligible = [
        bucket
        for bucket in buckets
        if isinstance(bucket, dict)
        and isinstance(bucket.get("steps"), int)
        and bucket["steps"] >= min_bucket_steps
    ]
    if len(eligible) < 2:
        return None

    first = eligible[0]
    last = eligible[-1]
    if not isinstance(first, dict) or not isinstance(last, dict):
        return None

    return number(last.get("fill_intensity"), f"{field}[-1].fill_intensity") - number(
        first.get("fill_intensity"), f"{field}[0].fill_intensity"
    )


def spread_response(report: dict[str, Any], min_bucket_steps: int) -> float | None:
    response = bucket_response(report, "by_spread_bucket", min_bucket_steps)
    return -response if response is not None else None


def comparison_row(path: Path, min_bucket_steps: int) -> ComparisonRow:
    report = load_report(path)
    overall = report["overall"]
    if not isinstance(overall, dict):
        raise ValueError(f"{path}: field 'overall' must be an object")

    return ComparisonRow(
        experiment=experiment_name(report, path),
        rows=int(report["rows"]),
        seeds=len(report["seeds"]),
        fill_probability=number(overall.get("fill_probability"), "overall.fill_probability"),
        fill_intensity=number(overall.get("fill_intensity"), "overall.fill_intensity"),
        avg_spread=number(overall.get("avg_spread"), "overall.avg_spread"),
        avg_volatility=number(
            overall.get("avg_estimated_volatility"), "overall.avg_estimated_volatility"
        ),
        adverse_selection_cost_per_fill=number(
            overall.get("adverse_selection_cost_per_fill"),
            "overall.adverse_selection_cost_per_fill",
        ),
        low_vol_fill_intensity=optional_regime_fill_intensity(report, "LowVol"),
        normal_vol_fill_intensity=optional_regime_fill_intensity(report, "NormalVol"),
        high_vol_fill_intensity=optional_regime_fill_intensity(report, "HighVol"),
        spread_response=spread_response(report, min_bucket_steps),
        volatility_response=bucket_response(
            report, "by_volatility_bucket", min_bucket_steps
        ),
    )


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_optional(value: float | None) -> str:
    return "" if value is None else format_float(value)


def render_table(rows: list[ComparisonRow]) -> str:
    headers = [
        "experiment",
        "rows",
        "seeds",
        "fill_prob",
        "fill_int",
        "avg_spread",
        "avg_vol",
        "adv_fill",
        "low_int",
        "normal_int",
        "high_int",
        "spread_resp",
        "vol_resp",
    ]
    table_rows = [
        [
            row.experiment,
            str(row.rows),
            str(row.seeds),
            format_float(row.fill_probability),
            format_float(row.fill_intensity),
            format_float(row.avg_spread),
            format_float(row.avg_volatility),
            format_float(row.adverse_selection_cost_per_fill),
            format_optional(row.low_vol_fill_intensity),
            format_optional(row.normal_vol_fill_intensity),
            format_optional(row.high_vol_fill_intensity),
            format_optional(row.spread_response),
            format_optional(row.volatility_response),
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = ["Calibration comparison"]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in table_rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def write_csv(path: Path, rows: list[ComparisonRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].as_csv_row().keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def main() -> int:
    args = parse_args()
    report_paths = discover_reports(args.reports)

    if not report_paths:
        print(
            f"error: no calibration reports found for {DEFAULT_PATTERN}",
            file=sys.stderr,
        )
        return 1

    try:
        if args.min_bucket_steps < 1:
            raise ValueError("--min-bucket-steps must be at least 1")

        rows = [comparison_row(path, args.min_bucket_steps) for path in report_paths]
        rows.sort(key=lambda row: row.experiment)
        write_csv(args.output, rows)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_table(rows))
    print()
    print(f"Comparison CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

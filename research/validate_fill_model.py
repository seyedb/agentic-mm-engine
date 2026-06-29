#!/usr/bin/env python3
"""Validate empirical fill-model behavior from calibration comparisons."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("target/research/fill_calibration_comparison.csv")
DEFAULT_OUTPUT = Path("target/research/fill_model_validation.txt")


@dataclass
class ExperimentRow:
    experiment: str
    rows: int
    seeds: int
    fill_probability: float
    fill_intensity: float
    avg_spread: float
    avg_volatility: float
    adverse_selection_cost_per_fill: float
    low_vol_steps: int | None
    normal_vol_steps: int | None
    high_vol_steps: int | None
    low_vol_fill_intensity: float | None
    normal_vol_fill_intensity: float | None
    high_vol_fill_intensity: float | None
    spread_response: float | None
    volatility_response: float | None


@dataclass
class ValidationCheck:
    status: str
    scope: str
    check: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fill-model calibration behavior."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Calibration comparison CSV produced by compare_calibrations.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the validation text report.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1_000,
        help="Minimum rows expected per experiment.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help="Minimum random seeds expected per experiment.",
    )
    parser.add_argument(
        "--min-regime-steps",
        type=int,
        default=100,
        help="Minimum low/high regime steps required for regime-intensity checks.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Numerical tolerance for directional checks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when warnings are present.",
    )
    return parser.parse_args()


def optional_float(value: str) -> float | None:
    value = value.strip()
    return None if value == "" else float(value)


def optional_int(value: str) -> int | None:
    value = value.strip()
    return None if value == "" else int(value)


def parse_row(row: dict[str, str]) -> ExperimentRow:
    return ExperimentRow(
        experiment=row["experiment"],
        rows=int(row["rows"]),
        seeds=int(row["seeds"]),
        fill_probability=float(row["fill_probability"]),
        fill_intensity=float(row["fill_intensity"]),
        avg_spread=float(row["avg_spread"]),
        avg_volatility=float(row["avg_volatility"]),
        adverse_selection_cost_per_fill=float(row["adverse_selection_cost_per_fill"]),
        low_vol_steps=optional_int(row.get("low_vol_steps", "")),
        normal_vol_steps=optional_int(row.get("normal_vol_steps", "")),
        high_vol_steps=optional_int(row.get("high_vol_steps", "")),
        low_vol_fill_intensity=optional_float(row["low_vol_fill_intensity"]),
        normal_vol_fill_intensity=optional_float(row["normal_vol_fill_intensity"]),
        high_vol_fill_intensity=optional_float(row["high_vol_fill_intensity"]),
        spread_response=optional_float(row["spread_response"]),
        volatility_response=optional_float(row["volatility_response"]),
    )


def read_rows(path: Path) -> list[ExperimentRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("comparison CSV is empty")
        return [parse_row(row) for row in reader]


def pass_check(scope: str, check: str, detail: str) -> ValidationCheck:
    return ValidationCheck("PASS", scope, check, detail)


def warn_check(scope: str, check: str, detail: str) -> ValidationCheck:
    return ValidationCheck("WARN", scope, check, detail)


def validate_experiment(
    row: ExperimentRow,
    min_rows: int,
    min_seeds: int,
    min_regime_steps: int,
    tolerance: float,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    scope = row.experiment

    if row.rows >= min_rows:
        checks.append(pass_check(scope, "sample_size", f"{row.rows} rows"))
    else:
        checks.append(warn_check(scope, "sample_size", f"only {row.rows} rows"))

    if row.seeds >= min_seeds:
        checks.append(pass_check(scope, "seed_count", f"{row.seeds} seeds"))
    else:
        checks.append(warn_check(scope, "seed_count", f"only {row.seeds} seeds"))

    if row.fill_probability > 0.0 and row.fill_intensity > 0.0:
        checks.append(
            pass_check(
                scope,
                "fill_activity",
                f"fill_prob={row.fill_probability:.4f}, fill_int={row.fill_intensity:.4f}",
            )
        )
    else:
        checks.append(warn_check(scope, "fill_activity", "no observed fill activity"))

    if row.spread_response is None:
        checks.append(warn_check(scope, "spread_response", "insufficient spread buckets"))
    elif row.spread_response >= -tolerance:
        checks.append(
            pass_check(
                scope,
                "spread_response",
                f"response={row.spread_response:.4f}",
            )
        )
    else:
        checks.append(
            warn_check(
                scope,
                "spread_response",
                f"response={row.spread_response:.4f}; fills rose as spreads widened",
            )
        )

    if row.volatility_response is None:
        checks.append(
            warn_check(scope, "volatility_response", "insufficient volatility buckets")
        )
    elif row.volatility_response >= -tolerance:
        checks.append(
            pass_check(
                scope,
                "volatility_response",
                f"response={row.volatility_response:.4f}",
            )
        )
    else:
        checks.append(
            warn_check(
                scope,
                "volatility_response",
                f"response={row.volatility_response:.4f}; fills fell in higher-vol buckets",
            )
        )

    if row.low_vol_fill_intensity is not None and row.high_vol_fill_intensity is not None:
        if (
            row.low_vol_steps is None
            or row.high_vol_steps is None
            or row.low_vol_steps < min_regime_steps
            or row.high_vol_steps < min_regime_steps
        ):
            checks.append(
                warn_check(
                    scope,
                    "regime_intensity",
                    (
                        "insufficient regime sample: "
                        f"low_steps={row.low_vol_steps}, "
                        f"high_steps={row.high_vol_steps}"
                    ),
                )
            )
        elif row.high_vol_fill_intensity + tolerance >= row.low_vol_fill_intensity:
            checks.append(
                pass_check(
                    scope,
                    "regime_intensity",
                    (
                        f"low={row.low_vol_fill_intensity:.4f}, "
                        f"high={row.high_vol_fill_intensity:.4f}"
                    ),
                )
            )
        else:
            checks.append(
                warn_check(
                    scope,
                    "regime_intensity",
                    (
                        f"low={row.low_vol_fill_intensity:.4f}, "
                        f"high={row.high_vol_fill_intensity:.4f}; "
                        "high-vol regime did not fill more often"
                    ),
                )
            )

    return checks


def validate_cross_experiment(
    rows: list[ExperimentRow], tolerance: float
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    sorted_rows = sorted(rows, key=lambda row: row.avg_volatility)

    adverse_violations = []
    spread_violations = []
    for previous, current in zip(sorted_rows, sorted_rows[1:]):
        if (
            current.adverse_selection_cost_per_fill + tolerance
            < previous.adverse_selection_cost_per_fill
        ):
            adverse_violations.append((previous, current))
        if current.avg_spread + tolerance < previous.avg_spread:
            spread_violations.append((previous, current))

    if adverse_violations:
        details = "; ".join(
            (
                f"{previous.experiment}->{current.experiment}: "
                f"{previous.adverse_selection_cost_per_fill:.4f}->"
                f"{current.adverse_selection_cost_per_fill:.4f}"
            )
            for previous, current in adverse_violations
        )
        checks.append(
            warn_check(
                "cross_experiment",
                "adverse_vs_volatility",
                f"adverse cost per fill decreased with volatility: {details}",
            )
        )
    else:
        checks.append(
            pass_check(
                "cross_experiment",
                "adverse_vs_volatility",
                "adverse cost per fill is nondecreasing with average volatility",
            )
        )

    if spread_violations:
        details = "; ".join(
            (
                f"{previous.experiment}->{current.experiment}: "
                f"{previous.avg_spread:.4f}->{current.avg_spread:.4f}"
            )
            for previous, current in spread_violations
        )
        checks.append(
            warn_check(
                "cross_experiment",
                "spread_vs_volatility",
                f"average spread decreased with volatility: {details}",
            )
        )
    else:
        checks.append(
            pass_check(
                "cross_experiment",
                "spread_vs_volatility",
                "average spread is nondecreasing with average volatility",
            )
        )

    return checks


def render_report(input_path: Path, rows: list[ExperimentRow], checks: list[ValidationCheck]) -> str:
    passed = sum(1 for check in checks if check.status == "PASS")
    warned = sum(1 for check in checks if check.status == "WARN")

    headers = ["status", "scope", "check", "detail"]
    table_rows = [
        [check.status, check.scope, check.check, check.detail] for check in checks
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        widths = [
            max(width, len(value)) for width, value in zip(widths, table_row)
        ]

    lines = [
        "Fill-model validation",
        f"Input: {input_path}",
        f"Experiments: {len(rows)}",
        f"Checks: {passed} pass, {warned} warn",
        "",
        "  ".join(header.rjust(width) for header, width in zip(headers, widths)),
    ]
    for table_row in table_rows:
        lines.append(
            "  ".join(value.rjust(width) for value, width in zip(table_row, widths))
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        if args.min_rows < 1:
            raise ValueError("--min-rows must be at least 1")
        if args.min_seeds < 1:
            raise ValueError("--min-seeds must be at least 1")
        if args.min_regime_steps < 1:
            raise ValueError("--min-regime-steps must be at least 1")
        if args.tolerance < 0.0:
            raise ValueError("--tolerance must be nonnegative")

        rows = read_rows(args.input)
        if not rows:
            raise ValueError("comparison CSV has no experiment rows")

        checks: list[ValidationCheck] = []
        for row in rows:
            checks.extend(
                validate_experiment(
                    row,
                    args.min_rows,
                    args.min_seeds,
                    args.min_regime_steps,
                    args.tolerance,
                )
            )
        checks.extend(validate_cross_experiment(rows, args.tolerance))

        report = render_report(args.input, rows, checks)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report)
    print()
    print(f"Validation report: {args.output}")

    has_warnings = any(check.status == "WARN" for check in checks)
    return 1 if args.strict and has_warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())

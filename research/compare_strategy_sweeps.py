#!/usr/bin/env python3
"""Compare best strategy-sweep results under a shared simulation setup."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIGS = [
    Path("configs/mixed_regime_fixed_spread_sweep.json"),
    Path("configs/mixed_regime_volatility_aware_sweep.json"),
    Path("configs/mixed_regime_inventory_risk_sweep.json"),
    Path("configs/mixed_regime_rule_based_controller_sweep.json"),
    Path("configs/mixed_regime_adaptive_sweep.json"),
    Path("configs/mixed_regime_avellaneda_stoikov_sweep.json"),
]
REPORT_DIR = Path("target/reports")
OUTPUT_DIR = Path("target/research")
OUTPUT_PATH = OUTPUT_DIR / "strategy_comparison.csv"

REQUIRED_COLUMNS = {
    "rank",
    "experiment",
    "strategy_type",
    "spread",
    "runs",
    "stable_score",
    "score",
    "score_std",
    "avg_final_pnl",
    "avg_max_drawdown",
    "avg_max_abs_inventory",
    "avg_total_fills",
    "avg_total_fees",
}


@dataclass
class BestStrategy:
    experiment: str
    strategy_type: str
    spread: float
    volatility_coeff: str
    risk_aversion: str
    liquidity_depth: str
    horizon: str
    inventory_limit: str
    skew: str
    runs: int
    stable_score: float
    score: float
    score_std: float
    avg_final_pnl: float
    avg_max_drawdown: float
    avg_max_abs_inventory: float
    avg_total_fills: float
    avg_total_fees: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and compare mixed-regime strategy sweep configs."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="Sweep config JSON files. Defaults to the mixed-regime strategy set.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Read existing target/reports CSV files without running cargo first.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Comparison CSV path. Default: {OUTPUT_PATH}",
    )
    return parser.parse_args()


def selected_configs(args: argparse.Namespace) -> list[Path]:
    return args.configs if args.configs else DEFAULT_CONFIGS


def validate_configs(configs: list[Path]) -> None:
    missing = [path for path in configs if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise ValueError(f"missing config files: {paths}")


def config_name(path: Path) -> str:
    with path.open() as handle:
        config = json.load(handle)

    name = config.get("name")
    if isinstance(name, str) and name:
        return name

    return path.stem


def run_configs(configs: list[Path]) -> None:
    command = ["cargo", "run", "--", *[str(path) for path in configs]]
    subprocess.run(command, check=True)


def report_path(config: Path) -> Path:
    return REPORT_DIR / f"{config_name(config)}.csv"


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


def optional_value(row: dict[str, str], column: str) -> str:
    return row.get(column, "")


def read_best(path: Path) -> BestStrategy:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: CSV missing required columns: {', '.join(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: CSV contains no sweep rows")

    rows.sort(key=lambda row: parse_int(row["rank"], "rank"))
    row = rows[0]

    return BestStrategy(
        experiment=row["experiment"],
        strategy_type=row["strategy_type"],
        spread=parse_float(row["spread"], "spread"),
        volatility_coeff=optional_value(row, "volatility_coeff"),
        risk_aversion=optional_value(row, "risk_aversion"),
        liquidity_depth=optional_value(row, "liquidity_depth"),
        horizon=optional_value(row, "horizon"),
        inventory_limit=optional_value(row, "inventory_limit"),
        skew=optional_value(row, "skew"),
        runs=parse_int(row["runs"], "runs"),
        stable_score=parse_float(row["stable_score"], "stable_score"),
        score=parse_float(row["score"], "score"),
        score_std=parse_float(row["score_std"], "score_std"),
        avg_final_pnl=parse_float(row["avg_final_pnl"], "avg_final_pnl"),
        avg_max_drawdown=parse_float(row["avg_max_drawdown"], "avg_max_drawdown"),
        avg_max_abs_inventory=parse_float(
            row["avg_max_abs_inventory"], "avg_max_abs_inventory"
        ),
        avg_total_fills=parse_float(row["avg_total_fills"], "avg_total_fills"),
        avg_total_fees=parse_float(row["avg_total_fees"], "avg_total_fees"),
    )


def format_number(value: float) -> str:
    return f"{value:.6f}"


def write_comparison(rows: list[BestStrategy], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "experiment",
                "strategy_type",
                "spread",
                "volatility_coeff",
                "risk_aversion",
                "liquidity_depth",
                "horizon",
                "inventory_limit",
                "skew",
                "runs",
                "stable_score",
                "score",
                "score_std",
                "avg_final_pnl",
                "avg_max_drawdown",
                "avg_max_abs_inventory",
                "avg_total_fills",
                "avg_total_fees",
            ]
        )

        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    row.experiment,
                    row.strategy_type,
                    format_number(row.spread),
                    row.volatility_coeff,
                    row.risk_aversion,
                    row.liquidity_depth,
                    row.horizon,
                    row.inventory_limit,
                    row.skew,
                    row.runs,
                    format_number(row.stable_score),
                    format_number(row.score),
                    format_number(row.score_std),
                    format_number(row.avg_final_pnl),
                    format_number(row.avg_max_drawdown),
                    format_number(row.avg_max_abs_inventory),
                    format_number(row.avg_total_fills),
                    format_number(row.avg_total_fees),
                ]
            )


def print_summary(rows: list[BestStrategy]) -> None:
    print("Strategy comparison")
    print(
        f"{'rank':<4} {'strategy':<34} {'stable':>10} {'pnl':>10} "
        f"{'fills':>10} {'drawdown':>10}"
    )
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:<4} {row.strategy_type:<34} {row.stable_score:>10.2f} "
            f"{row.avg_final_pnl:>10.2f} {row.avg_total_fills:>10.1f} "
            f"{row.avg_max_drawdown:>10.2f}"
        )


def main() -> int:
    args = parse_args()
    configs = selected_configs(args)

    try:
        validate_configs(configs)
        if not args.skip_run:
            run_configs(configs)

        paths = [report_path(config) for config in configs]
        missing_reports = [path for path in paths if not path.exists()]
        if missing_reports:
            missing = ", ".join(str(path) for path in missing_reports)
            raise ValueError(f"missing report CSV files: {missing}")

        rows = [read_best(path) for path in paths]
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: row.stable_score, reverse=True)
    write_comparison(rows, args.out)
    print_summary(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run and compare paper-session configs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("target/research/paper_config_comparison.csv")
REQUIRED_COLUMNS = {
    "observed_bid",
    "observed_ask",
    "estimated_volatility",
    "spread",
    "inventory",
    "pnl",
    "drawdown",
    "fills",
    "fill_notional",
    "fees",
}


@dataclass(frozen=True)
class ConfigSummary:
    config: Path
    output: Path
    policy: str
    fee_rate: float
    fee_spread_multiplier: float
    base_intensity: str
    distance_decay: str
    volatility_boost: str
    steps: int
    fills: int
    final_pnl: float
    total_fees: float
    max_drawdown: float
    min_inventory: float
    max_inventory: float
    avg_spread: float
    avg_quote_distance: float
    observed_quote_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paper-session configs.")
    parser.add_argument("config_paths", nargs="+", type=Path, help="paper_session configs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Comparison CSV output path.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Compare existing output CSVs without running configs.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = json.load(handle)

    if config.get("type") != "paper_session":
        raise ValueError(f"{path}: config type must be 'paper_session'")
    if "output" not in config:
        raise ValueError(f"{path}: paper_session config must include output")
    return config


def run_config(path: Path) -> None:
    subprocess.run(["cargo", "run", "--", "run", str(path)], cwd=PROJECT_ROOT, check=True)


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fill_model_value(config: dict[str, Any], key: str) -> str:
    fill_model = config.get("fill_model", {})
    if not isinstance(fill_model, dict):
        return ""
    value = fill_model.get(key)
    return "" if value is None else str(value)


def policy_name(config: dict[str, Any]) -> str:
    policy = config.get("policy", {})
    if not isinstance(policy, dict):
        return "static"
    value = policy.get("type", "static")
    return str(value)


def summarize(config_path: Path, config: dict[str, Any]) -> ConfigSummary:
    output = project_path(Path(config["output"]))
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{output}: CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{output}: CSV missing required columns: {', '.join(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{output}: CSV contains no rows")

    fills = [parse_int(row["fills"], "fills") for row in rows]
    inventories = [parse_float(row["inventory"], "inventory") for row in rows]
    spreads = [parse_float(row["spread"], "spread") for row in rows]
    drawdowns = [parse_float(row["drawdown"], "drawdown") for row in rows]
    fees = [parse_float(row["fees"], "fees") for row in rows]
    bid_distances = [
        parse_float(row["observed_ask"], "observed_ask") - parse_float(row["bid"], "bid")
        for row in rows
        if row["observed_ask"] != ""
    ]
    ask_distances = [
        parse_float(row["ask"], "ask") - parse_float(row["observed_bid"], "observed_bid")
        for row in rows
        if row["observed_bid"] != ""
    ]

    return ConfigSummary(
        config=config_path,
        output=output,
        policy=policy_name(config),
        fee_rate=float(config.get("fee_rate", 0.0)),
        fee_spread_multiplier=float(config.get("fee_spread_multiplier", 0.0)),
        base_intensity=fill_model_value(config, "base_intensity"),
        distance_decay=fill_model_value(config, "distance_decay"),
        volatility_boost=fill_model_value(config, "volatility_boost"),
        steps=len(rows),
        fills=sum(fills),
        final_pnl=parse_float(rows[-1]["pnl"], "pnl"),
        total_fees=sum(fees),
        max_drawdown=max(drawdowns),
        min_inventory=min(inventories),
        max_inventory=max(inventories),
        avg_spread=mean(spreads),
        avg_quote_distance=mean([mean(bid_distances), mean(ask_distances)]),
        observed_quote_steps=min(len(bid_distances), len(ask_distances)),
    )


def write_summaries(path: Path, summaries: list[ConfigSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "config",
                "policy",
                "fee_rate",
                "fee_spread_multiplier",
                "base_intensity",
                "distance_decay",
                "volatility_boost",
                "steps",
                "fills",
                "final_pnl",
                "total_fees",
                "max_drawdown",
                "min_inventory",
                "max_inventory",
                "avg_spread",
                "avg_quote_distance",
                "observed_quote_steps",
                "output",
            ]
        )
        for summary in summaries:
            writer.writerow(summary_to_csv_row(summary))


def summary_to_csv_row(summary: ConfigSummary) -> list[str]:
    return [
        str(summary.config),
        summary.policy,
        format_float(summary.fee_rate),
        format_float(summary.fee_spread_multiplier),
        summary.base_intensity,
        summary.distance_decay,
        summary.volatility_boost,
        str(summary.steps),
        str(summary.fills),
        format_float(summary.final_pnl),
        format_float(summary.total_fees),
        format_float(summary.max_drawdown),
        format_float(summary.min_inventory),
        format_float(summary.max_inventory),
        format_float(summary.avg_spread),
        format_float(summary.avg_quote_distance),
        str(summary.observed_quote_steps),
        str(summary.output),
    ]


def format_float(value: float) -> str:
    return f"{value:.6f}"


def render_table(summaries: list[ConfigSummary]) -> str:
    headers = ["config", "policy", "fee", "floor", "fills", "pnl", "fees", "spread", "qdist"]
    rows = [
        [
            summary.config.stem,
            summary.policy,
            f"{summary.fee_rate:.5f}",
            f"{summary.fee_spread_multiplier:.2f}",
            str(summary.fills),
            f"{summary.final_pnl:.4f}",
            f"{summary.total_fees:.4f}",
            f"{summary.avg_spread:.4f}",
            f"{summary.avg_quote_distance:.4f}",
        ]
        for summary in summaries
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = ["Paper config comparison"]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        configs = [(project_path(path), load_config(project_path(path))) for path in args.config_paths]

        if not args.skip_run:
            for path, _ in configs:
                print(f"Running {path}", flush=True)
                run_config(path)

        summaries = [summarize(path, config) for path, config in configs]
        write_summaries(args.output, summaries)

        print(render_table(summaries))
        print(f"\nwrote {args.output}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

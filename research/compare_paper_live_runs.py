#!/usr/bin/env python3
"""Compare preserved live paper run CSVs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATTERN = "target/reports/paper_live/*.csv"
DEFAULT_OUTPUT = Path("target/research/paper_live_runs.csv")
REQUIRED_COLUMNS = {
    "mid_price",
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
class RunSummary:
    run_id: str
    csv_path: Path
    started_at: str
    ended_at: str
    pair: str
    samples_config: str
    interval_seconds: str
    base_intensity: str
    distance_decay: str
    volatility_boost: str
    steps: int
    fills: int
    fill_rate: float
    final_pnl: float
    max_drawdown: float
    total_fees: float
    traded_notional: float
    min_inventory: float
    max_inventory: float
    avg_spread: float
    avg_volatility: float
    observed_quote_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare live paper run CSVs.")
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
    return parser.parse_args()


def discover_inputs(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(Path(path) for path in glob.glob(DEFAULT_INPUT_PATTERN))


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


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def load_metadata(csv_path: Path) -> dict[str, Any]:
    path = metadata_path(csv_path)
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "" if value is None else str(value)


def config_value(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    return "" if value is None else str(value)


def fill_model_value(config: dict[str, Any], key: str) -> str:
    fill_model = config.get("fill_model", {})
    if not isinstance(fill_model, dict):
        return ""
    value = fill_model.get(key)
    return "" if value is None else str(value)


def run_id(csv_path: Path, metadata: dict[str, Any]) -> str:
    value = metadata.get("run_id")
    if value:
        return str(value)
    return csv_path.stem


def summarize_run(csv_path: Path) -> RunSummary:
    metadata = load_metadata(csv_path)
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        config = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{csv_path}: CSV missing required columns: {', '.join(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path}: CSV contains no rows")

    fills = [parse_int(row["fills"], "fills") for row in rows]
    inventories = [parse_float(row["inventory"], "inventory") for row in rows]
    spreads = [parse_float(row["spread"], "spread") for row in rows]
    volatilities = [
        parse_float(row["estimated_volatility"], "estimated_volatility") for row in rows
    ]
    drawdowns = [parse_float(row["drawdown"], "drawdown") for row in rows]
    fees = [parse_float(row["fees"], "fees") for row in rows]
    notionals = [parse_float(row["fill_notional"], "fill_notional") for row in rows]
    observed_quote_steps = sum(
        1 for row in rows if row["observed_bid"] != "" and row["observed_ask"] != ""
    )

    return RunSummary(
        run_id=run_id(csv_path, metadata),
        csv_path=csv_path,
        started_at=metadata_value(metadata, "started_at"),
        ended_at=metadata_value(metadata, "ended_at"),
        pair=config_value(config, "pair"),
        samples_config=config_value(config, "samples"),
        interval_seconds=config_value(config, "interval_seconds"),
        base_intensity=fill_model_value(config, "base_intensity"),
        distance_decay=fill_model_value(config, "distance_decay"),
        volatility_boost=fill_model_value(config, "volatility_boost"),
        steps=len(rows),
        fills=sum(fills),
        fill_rate=sum(1 for value in fills if value > 0) / len(rows),
        final_pnl=parse_float(rows[-1]["pnl"], "pnl"),
        max_drawdown=max(drawdowns),
        total_fees=sum(fees),
        traded_notional=sum(notionals),
        min_inventory=min(inventories),
        max_inventory=max(inventories),
        avg_spread=mean(spreads),
        avg_volatility=mean(volatilities),
        observed_quote_steps=observed_quote_steps,
    )


def write_summaries(path: Path, summaries: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "pair",
                "started_at",
                "ended_at",
                "steps",
                "samples_config",
                "interval_seconds",
                "fills",
                "fill_rate",
                "final_pnl",
                "max_drawdown",
                "total_fees",
                "traded_notional",
                "min_inventory",
                "max_inventory",
                "avg_spread",
                "avg_volatility",
                "observed_quote_steps",
                "base_intensity",
                "distance_decay",
                "volatility_boost",
                "csv_path",
            ]
        )
        for summary in summaries:
            writer.writerow(summary_to_csv_row(summary))


def summary_to_csv_row(summary: RunSummary) -> list[str]:
    return [
        summary.run_id,
        summary.pair,
        summary.started_at,
        summary.ended_at,
        str(summary.steps),
        summary.samples_config,
        summary.interval_seconds,
        str(summary.fills),
        format_float(summary.fill_rate),
        format_float(summary.final_pnl),
        format_float(summary.max_drawdown),
        format_float(summary.total_fees),
        format_float(summary.traded_notional),
        format_float(summary.min_inventory),
        format_float(summary.max_inventory),
        format_float(summary.avg_spread),
        format_float(summary.avg_volatility),
        str(summary.observed_quote_steps),
        summary.base_intensity,
        summary.distance_decay,
        summary.volatility_boost,
        str(summary.csv_path),
    ]


def format_float(value: float) -> str:
    return f"{value:.6f}"


def render_table(summaries: list[RunSummary]) -> str:
    headers = [
        "run_id",
        "steps",
        "fills",
        "pnl",
        "drawdown",
        "fees",
        "inv_min",
        "inv_max",
    ]
    rows = [
        [
            summary.run_id,
            str(summary.steps),
            str(summary.fills),
            f"{summary.final_pnl:.4f}",
            f"{summary.max_drawdown:.4f}",
            f"{summary.total_fees:.4f}",
            f"{summary.min_inventory:.4f}",
            f"{summary.max_inventory:.4f}",
        ]
        for summary in summaries
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = ["Paper live run comparison"]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        inputs = discover_inputs(args.csv_paths)
        if not inputs:
            raise ValueError("no live paper CSVs found")

        summaries = [summarize_run(path) for path in inputs]
        summaries.sort(key=lambda summary: summary.run_id)
        write_summaries(args.output, summaries)

        print(render_table(summaries))
        print(f"\nwrote {args.output}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

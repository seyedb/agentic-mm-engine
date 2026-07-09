#!/usr/bin/env python3
"""Evaluate paper-session policies across quote windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_OUTPUT = Path("target/research/paper_policy_window_runs.csv")
DEFAULT_AGG_OUTPUT = Path("target/research/paper_policy_evaluation.csv")
DEFAULT_WORK_DIR = Path("target/research/paper_policy_windows")
REQUIRED_PAPER_COLUMNS = {
    "observed_bid",
    "observed_ask",
    "bid",
    "ask",
    "spread",
    "inventory",
    "pnl",
    "drawdown",
    "fills",
    "fees",
}


@dataclass(frozen=True)
class Window:
    name: str
    path: Path
    start_row: int
    end_row: int


@dataclass(frozen=True)
class RunSummary:
    window: str
    config: str
    policy: str
    steps: int
    fills: int
    final_pnl: float
    total_fees: float
    max_drawdown: float
    min_inventory: float
    max_inventory: float
    avg_spread: float
    avg_quote_distance: float
    output: Path


@dataclass(frozen=True)
class AggregateSummary:
    config: str
    policy: str
    windows: int
    mean_fills: float
    std_fills: float
    mean_pnl: float
    std_pnl: float
    mean_fees: float
    mean_drawdown: float
    mean_spread: float
    mean_quote_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paper-session policies across quote-data windows."
    )
    parser.add_argument("config_paths", nargs="+", type=Path, help="paper_session configs.")
    parser.add_argument(
        "--data",
        type=Path,
        help="Quote CSV to window. Defaults to the data path in the first config.",
    )
    parser.add_argument("--window-size", type=int, default=10, help="Rows per window.")
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Rows between window starts.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum windows to evaluate. Use 0 for all windows.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory for generated window CSVs, runtime configs, and run CSVs.",
    )
    parser.add_argument(
        "--runs-output",
        type=Path,
        default=DEFAULT_RUN_OUTPUT,
        help="Per-window run summary CSV.",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=DEFAULT_AGG_OUTPUT,
        help="Aggregate summary CSV.",
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
    return config


def policy_name(config: dict[str, Any]) -> str:
    policy = config.get("policy", {})
    if not isinstance(policy, dict):
        return "static"
    return str(policy.get("type", "static"))


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: CSV contains no rows")
    return reader.fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_windows(
    data_path: Path,
    work_dir: Path,
    window_size: int,
    step_size: int,
    max_windows: int,
) -> list[Window]:
    if window_size <= 0:
        raise ValueError("--window-size must be positive")
    if step_size <= 0:
        raise ValueError("--step-size must be positive")
    if max_windows < 0:
        raise ValueError("--max-windows must be non-negative")

    fieldnames, rows = read_csv_rows(data_path)
    if len(rows) < window_size:
        raise ValueError(f"{data_path}: only {len(rows)} rows, fewer than --window-size")

    windows = []
    for start in range(0, len(rows) - window_size + 1, step_size):
        end = start + window_size
        name = f"window_{len(windows) + 1:03d}_rows_{start + 1}_{end}"
        path = project_path(work_dir) / "data" / f"{name}.csv"
        write_csv_rows(path, fieldnames, rows[start:end])
        windows.append(Window(name=name, path=path, start_row=start + 1, end_row=end))
        if max_windows and len(windows) >= max_windows:
            break

    return windows


def runtime_config_path(work_dir: Path, window: Window, config_path: Path) -> Path:
    return project_path(work_dir) / "configs" / f"{window.name}_{config_path.stem}.json"


def runtime_output_path(work_dir: Path, window: Window, config_path: Path) -> Path:
    return project_path(work_dir) / "runs" / f"{window.name}_{config_path.stem}.csv"


def write_runtime_config(
    path: Path,
    config: dict[str, Any],
    window: Window,
    output_path: Path,
) -> None:
    runtime_config = dict(config)
    runtime_config["data"] = str(window.path)
    runtime_config["output"] = str(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_config, indent=2, sort_keys=True) + "\n")


def run_config(path: Path) -> None:
    subprocess.run(["cargo", "run", "--quiet", "--", "run", str(path)], cwd=PROJECT_ROOT, check=True)


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in column '{column}': {row[column]!r}") from exc


def parse_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid integer in column '{column}': {row[column]!r}") from exc


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def summarize_output(
    window: Window,
    config_path: Path,
    config: dict[str, Any],
    output_path: Path,
) -> RunSummary:
    fieldnames, rows = read_csv_rows(output_path)
    missing = sorted(REQUIRED_PAPER_COLUMNS.difference(fieldnames))
    if missing:
        raise ValueError(f"{output_path}: CSV missing required columns: {', '.join(missing)}")

    bid_distances = [
        parse_float(row, "observed_ask") - parse_float(row, "bid")
        for row in rows
        if row["observed_ask"] != ""
    ]
    ask_distances = [
        parse_float(row, "ask") - parse_float(row, "observed_bid")
        for row in rows
        if row["observed_bid"] != ""
    ]
    inventories = [parse_float(row, "inventory") for row in rows]
    fees = [parse_float(row, "fees") for row in rows]
    drawdowns = [parse_float(row, "drawdown") for row in rows]
    spreads = [parse_float(row, "spread") for row in rows]

    return RunSummary(
        window=window.name,
        config=config_path.stem,
        policy=policy_name(config),
        steps=len(rows),
        fills=sum(parse_int(row, "fills") for row in rows),
        final_pnl=parse_float(rows[-1], "pnl"),
        total_fees=sum(fees),
        max_drawdown=max(drawdowns),
        min_inventory=min(inventories),
        max_inventory=max(inventories),
        avg_spread=mean(spreads),
        avg_quote_distance=mean([mean(bid_distances), mean(ask_distances)]),
        output=output_path,
    )


def aggregate_runs(runs: list[RunSummary]) -> list[AggregateSummary]:
    groups: dict[tuple[str, str], list[RunSummary]] = {}
    for run in runs:
        groups.setdefault((run.config, run.policy), []).append(run)

    summaries = []
    for (config, policy), group in sorted(groups.items()):
        fills = [float(run.fills) for run in group]
        pnls = [run.final_pnl for run in group]
        summaries.append(
            AggregateSummary(
                config=config,
                policy=policy,
                windows=len(group),
                mean_fills=mean(fills),
                std_fills=sample_std(fills),
                mean_pnl=mean(pnls),
                std_pnl=sample_std(pnls),
                mean_fees=mean([run.total_fees for run in group]),
                mean_drawdown=mean([run.max_drawdown for run in group]),
                mean_spread=mean([run.avg_spread for run in group]),
                mean_quote_distance=mean([run.avg_quote_distance for run in group]),
            )
        )
    return summaries


def write_run_summaries(path: Path, runs: list[RunSummary]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "config",
                "policy",
                "steps",
                "fills",
                "final_pnl",
                "total_fees",
                "max_drawdown",
                "min_inventory",
                "max_inventory",
                "avg_spread",
                "avg_quote_distance",
                "output",
            ]
        )
        for run in runs:
            writer.writerow(run_to_csv_row(run))


def run_to_csv_row(run: RunSummary) -> list[str]:
    return [
        run.window,
        run.config,
        run.policy,
        str(run.steps),
        str(run.fills),
        format_float(run.final_pnl),
        format_float(run.total_fees),
        format_float(run.max_drawdown),
        format_float(run.min_inventory),
        format_float(run.max_inventory),
        format_float(run.avg_spread),
        format_float(run.avg_quote_distance),
        str(run.output),
    ]


def write_aggregate_summaries(path: Path, summaries: list[AggregateSummary]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "config",
                "policy",
                "windows",
                "mean_fills",
                "std_fills",
                "mean_pnl",
                "std_pnl",
                "mean_fees",
                "mean_drawdown",
                "mean_spread",
                "mean_quote_distance",
            ]
        )
        for summary in summaries:
            writer.writerow(aggregate_to_csv_row(summary))


def aggregate_to_csv_row(summary: AggregateSummary) -> list[str]:
    return [
        summary.config,
        summary.policy,
        str(summary.windows),
        format_float(summary.mean_fills),
        format_float(summary.std_fills),
        format_float(summary.mean_pnl),
        format_float(summary.std_pnl),
        format_float(summary.mean_fees),
        format_float(summary.mean_drawdown),
        format_float(summary.mean_spread),
        format_float(summary.mean_quote_distance),
    ]


def format_float(value: float) -> str:
    return f"{value:.6f}"


def render_aggregate_table(summaries: list[AggregateSummary]) -> str:
    headers = ["config", "policy", "windows", "fills", "pnl", "pnl_std", "fees", "drawdown", "spread", "qdist"]
    rows = [
        [
            summary.config,
            summary.policy,
            str(summary.windows),
            f"{summary.mean_fills:.2f}",
            f"{summary.mean_pnl:.4f}",
            f"{summary.std_pnl:.4f}",
            f"{summary.mean_fees:.4f}",
            f"{summary.mean_drawdown:.4f}",
            f"{summary.mean_spread:.4f}",
            f"{summary.mean_quote_distance:.4f}",
        ]
        for summary in summaries
    ]
    return render_table("Paper policy evaluation", headers, rows)


def render_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = [title]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        config_pairs = [(project_path(path), load_config(project_path(path))) for path in args.config_paths]
        data_path = project_path(args.data) if args.data else project_path(Path(config_pairs[0][1]["data"]))
        windows = build_windows(
            data_path,
            args.work_dir,
            args.window_size,
            args.step_size,
            args.max_windows,
        )

        runs = []
        for window in windows:
            for config_path, config in config_pairs:
                output_path = runtime_output_path(args.work_dir, window, config_path)
                runtime_path = runtime_config_path(args.work_dir, window, config_path)
                write_runtime_config(runtime_path, config, window, output_path)
                print(f"Running {window.name} {config_path.stem}", flush=True)
                run_config(runtime_path)
                runs.append(summarize_output(window, config_path, config, output_path))

        aggregates = aggregate_runs(runs)
        write_run_summaries(args.runs_output, runs)
        write_aggregate_summaries(args.aggregate_output, aggregates)

        print(render_aggregate_table(aggregates))
        print()
        print(f"wrote {project_path(args.runs_output)}")
        print(f"wrote {project_path(args.aggregate_output)}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

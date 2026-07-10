#!/usr/bin/env python3
"""Test policy conclusions under alternative paper fill assumptions."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATTERN = "data/quotes/*.meta.json"
DEFAULT_STATIC_CONFIG = Path("configs/runs/kraken_solusd_maker_fee_paper_session.json")
DEFAULT_ADAPTIVE_CONFIG = Path("configs/runs/kraken_solusd_adaptive_maker_fee_paper_session.json")
DEFAULT_WORK_DIR = Path("target/research/fill_assumption_sensitivity")
DEFAULT_OUTPUT = Path("target/research/fill_assumption_sensitivity.csv")


@dataclass(frozen=True)
class Assumption:
    name: str
    base_intensity: float
    distance_decay: float
    volatility_boost: float


@dataclass(frozen=True)
class SensitivitySummary:
    assumption: Assumption
    datasets: int
    windows: int
    pnl_wins: int
    drawdown_reductions: int
    fee_reductions: int
    low_participation_windows: int
    mean_static_fills: float
    mean_adaptive_fills: float
    mean_pnl_delta: float
    mean_drawdown_delta: float
    mean_fee_delta: float
    mean_fill_delta: float


@dataclass(frozen=True)
class RunRow:
    window: str
    policy: str
    fills: int
    final_pnl: float
    total_fees: float
    max_drawdown: float


@dataclass(frozen=True)
class PairDelta:
    static_fills: int
    adaptive_fills: int
    pnl_delta: float
    drawdown_delta: float
    fee_delta: float
    fill_delta: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate static/adaptive conclusions under multiple fill assumptions."
    )
    parser.add_argument(
        "metadata_paths",
        nargs="*",
        type=Path,
        help="Quote-dataset metadata files. Defaults to data/quotes/*.meta.json.",
    )
    parser.add_argument("--window-size", type=int, default=30, help="Rows per window.")
    parser.add_argument("--step-size", type=int, default=30, help="Rows between window starts.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum windows per dataset. Use 0 for all windows.",
    )
    parser.add_argument(
        "--base-intensities",
        default="0.2,0.35,0.5",
        help="Comma-separated base intensity values.",
    )
    parser.add_argument(
        "--distance-decays",
        default="100,160,240",
        help="Comma-separated distance decay values.",
    )
    parser.add_argument(
        "--volatility-boosts",
        default="0,1",
        help="Comma-separated volatility boost values.",
    )
    parser.add_argument("--static-config", type=Path, default=DEFAULT_STATIC_CONFIG)
    parser.add_argument("--adaptive-config", type=Path, default=DEFAULT_ADAPTIVE_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=12, help="Rows to print.")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def discover_metadata(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(Path(path) for path in glob.glob(DEFAULT_METADATA_PATTERN))


def parse_float_list(value: str, name: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} must contain non-negative values")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.max_windows < 0:
        raise ValueError("--max-windows must be non-negative")
    if args.top <= 0:
        raise ValueError("--top must be positive")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("type") != "paper_session":
        raise ValueError(f"{path}: config type must be paper_session")
    return config


def fill_assumptions(args: argparse.Namespace) -> list[Assumption]:
    assumptions = []
    for base in parse_float_list(args.base_intensities, "--base-intensities"):
        for decay in parse_float_list(args.distance_decays, "--distance-decays"):
            for boost in parse_float_list(args.volatility_boosts, "--volatility-boosts"):
                name = f"base_{token(base)}_decay_{token(decay)}_boost_{token(boost)}"
                assumptions.append(
                    Assumption(
                        name=name,
                        base_intensity=base,
                        distance_decay=decay,
                        volatility_boost=boost,
                    )
                )
    return assumptions


def token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def config_with_fill_model(config: dict[str, Any], assumption: Assumption) -> dict[str, Any]:
    updated = dict(config)
    updated["fill_model"] = {
        "type": "touch_intensity",
        "base_intensity": assumption.base_intensity,
        "distance_decay": assumption.distance_decay,
        "volatility_boost": assumption.volatility_boost,
    }
    return updated


def metadata_run_id(path: Path, metadata: dict[str, Any]) -> str:
    value = metadata.get("run_id")
    if value:
        return str(value)
    return path.name.replace(".meta.json", "")


def metadata_csv_path(path: Path, metadata: dict[str, Any]) -> Path:
    value = metadata.get("csv_path")
    if not value:
        raise ValueError(f"{path}: metadata missing csv_path")
    csv_path = project_path(Path(str(value)))
    if not csv_path.exists():
        raise ValueError(f"{path}: quote CSV not found: {csv_path}")
    return csv_path


def write_runtime_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def run_evaluation(
    *,
    assumption: Assumption,
    run_id: str,
    data_path: Path,
    static_config: dict[str, Any],
    adaptive_config: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    base = project_path(args.work_dir) / assumption.name / run_id
    config_dir = base / "configs"
    static_path = config_dir / "static.json"
    adaptive_path = config_dir / "adaptive.json"
    write_runtime_config(static_path, static_config)
    write_runtime_config(adaptive_path, adaptive_config)

    runs_output = base / "paper_policy_window_runs.csv"
    aggregate_output = base / "paper_policy_evaluation.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "research" / "evaluate_paper_policies.py"),
        str(static_path),
        str(adaptive_path),
        "--data",
        str(data_path),
        "--window-size",
        str(args.window_size),
        "--step-size",
        str(args.step_size),
        "--max-windows",
        str(args.max_windows),
        "--work-dir",
        str(base / "paper_policy_windows"),
        "--runs-output",
        str(runs_output),
        "--aggregate-output",
        str(aggregate_output),
    ]
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return runs_output


def read_run_rows(path: Path) -> list[RunRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        return [
            RunRow(
                window=row["window"],
                policy=row["policy"],
                fills=parse_int(row, "fills"),
                final_pnl=parse_float(row, "final_pnl"),
                total_fees=parse_float(row, "total_fees"),
                max_drawdown=parse_float(row, "max_drawdown"),
            )
            for row in reader
        ]


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in {column}: {row[column]!r}") from exc


def parse_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid int in {column}: {row[column]!r}") from exc


def pair_deltas(rows: list[RunRow]) -> list[PairDelta]:
    by_window: dict[str, list[RunRow]] = {}
    for row in rows:
        by_window.setdefault(row.window, []).append(row)

    deltas = []
    for group in by_window.values():
        static = first_policy(group, "static")
        adaptive = first_policy(group, "adaptive")
        if static is None or adaptive is None:
            continue
        deltas.append(
            PairDelta(
                static_fills=static.fills,
                adaptive_fills=adaptive.fills,
                pnl_delta=adaptive.final_pnl - static.final_pnl,
                drawdown_delta=adaptive.max_drawdown - static.max_drawdown,
                fee_delta=adaptive.total_fees - static.total_fees,
                fill_delta=adaptive.fills - static.fills,
            )
        )
    return deltas


def first_policy(rows: list[RunRow], policy: str) -> RunRow | None:
    for row in rows:
        if row.policy == policy:
            return row
    return None


def summarize_assumption(assumption: Assumption, datasets: int, deltas: list[PairDelta]) -> SensitivitySummary:
    if not deltas:
        raise ValueError(f"{assumption.name}: no paired windows")
    return SensitivitySummary(
        assumption=assumption,
        datasets=datasets,
        windows=len(deltas),
        pnl_wins=sum(1 for delta in deltas if delta.pnl_delta > 0.0),
        drawdown_reductions=sum(1 for delta in deltas if delta.drawdown_delta < 0.0),
        fee_reductions=sum(1 for delta in deltas if delta.fee_delta < 0.0),
        low_participation_windows=sum(
            1
            for delta in deltas
            if delta.static_fills > 0 and delta.adaptive_fills / delta.static_fills < 0.5
        ),
        mean_static_fills=mean([float(delta.static_fills) for delta in deltas]),
        mean_adaptive_fills=mean([float(delta.adaptive_fills) for delta in deltas]),
        mean_pnl_delta=mean([delta.pnl_delta for delta in deltas]),
        mean_drawdown_delta=mean([delta.drawdown_delta for delta in deltas]),
        mean_fee_delta=mean([delta.fee_delta for delta in deltas]),
        mean_fill_delta=mean([float(delta.fill_delta) for delta in deltas]),
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_summaries(path: Path, summaries: list[SensitivitySummary]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "assumption",
                "base_intensity",
                "distance_decay",
                "volatility_boost",
                "datasets",
                "windows",
                "pnl_wins",
                "drawdown_reductions",
                "fee_reductions",
                "low_participation_windows",
                "mean_static_fills",
                "mean_adaptive_fills",
                "mean_pnl_delta",
                "mean_drawdown_delta",
                "mean_fee_delta",
                "mean_fill_delta",
            ]
        )
        for summary in summaries:
            writer.writerow(summary_row(summary))


def summary_row(summary: SensitivitySummary) -> list[str]:
    assumption = summary.assumption
    return [
        assumption.name,
        format_float(assumption.base_intensity),
        format_float(assumption.distance_decay),
        format_float(assumption.volatility_boost),
        str(summary.datasets),
        str(summary.windows),
        str(summary.pnl_wins),
        str(summary.drawdown_reductions),
        str(summary.fee_reductions),
        str(summary.low_participation_windows),
        format_float(summary.mean_static_fills),
        format_float(summary.mean_adaptive_fills),
        format_float(summary.mean_pnl_delta),
        format_float(summary.mean_drawdown_delta),
        format_float(summary.mean_fee_delta),
        format_float(summary.mean_fill_delta),
    ]


def format_float(value: float) -> str:
    return f"{value:.6f}"


def render_table(summaries: list[SensitivitySummary], top: int) -> str:
    headers = ["fill_model", "pnl_wins", "dd_red", "fee_red", "pnl_delta", "dd_delta", "fill_delta"]
    rows = [
        [
            summary.assumption.name,
            f"{summary.pnl_wins}/{summary.windows}",
            f"{summary.drawdown_reductions}/{summary.windows}",
            f"{summary.fee_reductions}/{summary.windows}",
            f"{summary.mean_pnl_delta:.5f}",
            f"{summary.mean_drawdown_delta:.5f}",
            f"{summary.mean_fill_delta:.2f}",
        ]
        for summary in summaries[:top]
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = ["Fill assumption sensitivity"]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        metadata_paths = discover_metadata(args.metadata_paths)
        if not metadata_paths:
            raise ValueError("no quote-dataset metadata files found")

        static_template = load_config(project_path(args.static_config))
        adaptive_template = load_config(project_path(args.adaptive_config))
        assumptions = fill_assumptions(args)
        summaries = []
        for index, assumption in enumerate(assumptions, start=1):
            print(f"Running fill assumption {index}/{len(assumptions)} {assumption.name}", flush=True)
            static_config = config_with_fill_model(static_template, assumption)
            adaptive_config = config_with_fill_model(adaptive_template, assumption)
            deltas = []
            for metadata_path in metadata_paths:
                metadata = load_json(metadata_path)
                run_id = metadata_run_id(metadata_path, metadata)
                data_path = metadata_csv_path(metadata_path, metadata)
                runs_output = run_evaluation(
                    assumption=assumption,
                    run_id=run_id,
                    data_path=data_path,
                    static_config=static_config,
                    adaptive_config=adaptive_config,
                    args=args,
                )
                deltas.extend(pair_deltas(read_run_rows(runs_output)))
            summaries.append(summarize_assumption(assumption, len(metadata_paths), deltas))

        summaries.sort(
            key=lambda summary: (
                summary.drawdown_reductions / summary.windows,
                summary.fee_reductions / summary.windows,
                summary.mean_pnl_delta,
            ),
            reverse=True,
        )
        write_summaries(args.output, summaries)
        print(render_table(summaries, args.top))
        print()
        print(f"wrote {project_path(args.output)}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

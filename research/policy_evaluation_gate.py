#!/usr/bin/env python3
"""Run a policy evaluation gate across collected quote datasets."""

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
DEFAULT_HYBRID_CONFIG = Path("configs/runs/kraken_solusd_hybrid_maker_fee_paper_session.json")
DEFAULT_SELECTOR_CONFIG = Path("configs/runs/kraken_solusd_selector_maker_fee_paper_session.json")
DEFAULT_LEARNED_CONFIG = Path("configs/runs/kraken_solusd_learned_selector_maker_fee_paper_session.json")
DEFAULT_LINEAR_CONFIG = Path("configs/runs/kraken_solusd_linear_agent_maker_fee_paper_session.json")
DEFAULT_BANDIT_CONFIG = Path("configs/runs/kraken_solusd_bandit_agent_maker_fee_paper_session.json")
DEFAULT_WORK_DIR = Path("target/research/policy_gate")
DEFAULT_DATASET_OUTPUT = Path("target/research/policy_gate_dataset_summary.csv")
DEFAULT_WINDOW_OUTPUT = Path("target/research/policy_gate_window_results.csv")
DEFAULT_POLICY_OUTPUT = Path("target/research/policy_gate_policy_summary.csv")
DEFAULT_REPORT_OUTPUT = Path("target/research/policy_gate_report.md")


@dataclass(frozen=True)
class FillAssumption:
    name: str
    base_intensity: float | None = None
    distance_decay: float | None = None
    volatility_boost: float | None = None


@dataclass(frozen=True)
class Dataset:
    run_id: str
    pair: str
    samples: int
    interval_seconds: float
    csv_path: Path


@dataclass(frozen=True)
class AggregateRow:
    assumption: str
    run_id: str
    pair: str
    samples: int
    interval_seconds: float
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
    utility: float


@dataclass(frozen=True)
class WindowRow:
    assumption: str
    run_id: str
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
    mean_abs_inventory: float
    policy_static_steps: int
    policy_adaptive_steps: int
    trigger_none_steps: int
    trigger_configured_steps: int
    trigger_inventory_steps: int
    trigger_drawdown_steps: int
    trigger_volatility_steps: int
    trigger_spread_steps: int
    trigger_multiple_steps: int
    avg_spread: float
    avg_quote_distance: float
    utility: float


@dataclass(frozen=True)
class PolicySummary:
    assumption: str
    policy: str
    datasets: int
    windows: int
    mean_utility: float
    mean_pnl: float
    mean_drawdown: float
    mean_fees: float
    mean_fills: float
    mean_abs_inventory: float
    adaptive_step_pct: float
    trigger_configured_steps: int
    trigger_inventory_steps: int
    trigger_drawdown_steps: int
    trigger_volatility_steps: int
    trigger_spread_steps: int
    trigger_multiple_steps: int
    dataset_utility_wins: int
    window_utility_wins: int


@dataclass(frozen=True)
class RunDiagnostics:
    mean_abs_inventory: float
    policy_static_steps: int
    policy_adaptive_steps: int
    trigger_none_steps: int
    trigger_configured_steps: int
    trigger_inventory_steps: int
    trigger_drawdown_steps: int
    trigger_volatility_steps: int
    trigger_spread_steps: int
    trigger_multiple_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether current paper policies are worth studying further."
    )
    parser.add_argument(
        "metadata_paths",
        nargs="*",
        type=Path,
        help="Quote-dataset metadata files. Defaults to data/quotes/*.meta.json.",
    )
    parser.add_argument("--window-size", type=int, default=30, help="Rows per evaluation window.")
    parser.add_argument("--step-size", type=int, default=30, help="Rows between window starts.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum windows per dataset. Use 0 for all windows.",
    )
    parser.add_argument("--static-config", type=Path, default=DEFAULT_STATIC_CONFIG)
    parser.add_argument("--adaptive-config", type=Path, default=DEFAULT_ADAPTIVE_CONFIG)
    parser.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--selector-config", type=Path, default=DEFAULT_SELECTOR_CONFIG)
    parser.add_argument("--learned-config", type=Path, default=DEFAULT_LEARNED_CONFIG)
    parser.add_argument("--linear-config", type=Path, default=DEFAULT_LINEAR_CONFIG)
    parser.add_argument("--bandit-config", type=Path, default=DEFAULT_BANDIT_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET_OUTPUT)
    parser.add_argument("--window-output", type=Path, default=DEFAULT_WINDOW_OUTPUT)
    parser.add_argument("--policy-output", type=Path, default=DEFAULT_POLICY_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--drawdown-weight",
        type=float,
        default=2.0,
        help="Utility penalty per unit of max drawdown.",
    )
    parser.add_argument(
        "--inventory-weight",
        type=float,
        default=0.02,
        help="Utility penalty per unit of average absolute inventory.",
    )
    parser.add_argument(
        "--fee-weight",
        type=float,
        default=0.0,
        help="Extra utility penalty per unit of fees. Fees are already in PnL.",
    )
    parser.add_argument(
        "--include-fill-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also rerun the gate under conservative and liquid fill assumptions.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def discover_metadata(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(project_path(path) for path in paths)
    return sorted(Path(path) for path in glob.glob(str(PROJECT_ROOT / DEFAULT_METADATA_PATTERN)))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_dataset(path: Path) -> Dataset:
    metadata = load_json(path)
    csv_value = metadata.get("csv_path")
    if not csv_value:
        raise ValueError(f"{path}: metadata missing csv_path")
    csv_path = project_path(Path(str(csv_value)))
    if not csv_path.exists():
        raise ValueError(f"{path}: quote CSV not found: {csv_path}")

    return Dataset(
        run_id=str(metadata.get("run_id") or path.name.replace(".meta.json", "")),
        pair=str(metadata.get("pair") or ""),
        samples=parse_metadata_int(metadata, "samples"),
        interval_seconds=parse_metadata_float(metadata, "interval_seconds"),
        csv_path=csv_path,
    )


def parse_metadata_int(metadata: dict[str, Any], key: str) -> int:
    try:
        return int(metadata.get(key, 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata field {key!r} must be an integer") from exc


def parse_metadata_float(metadata: dict[str, Any], key: str) -> float:
    try:
        return float(metadata.get(key, 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata field {key!r} must be numeric") from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.max_windows < 0:
        raise ValueError("--max-windows must be non-negative")
    if args.drawdown_weight < 0.0:
        raise ValueError("--drawdown-weight must be non-negative")
    if args.inventory_weight < 0.0:
        raise ValueError("--inventory-weight must be non-negative")
    if args.fee_weight < 0.0:
        raise ValueError("--fee-weight must be non-negative")


def fill_assumptions(include_sensitivity: bool) -> list[FillAssumption]:
    assumptions = [FillAssumption(name="configured")]
    if include_sensitivity:
        assumptions.extend(
            [
                FillAssumption(
                    name="conservative_fill",
                    base_intensity=0.2,
                    distance_decay=240.0,
                    volatility_boost=0.0,
                ),
                FillAssumption(
                    name="liquid_fill",
                    base_intensity=0.5,
                    distance_decay=100.0,
                    volatility_boost=1.0,
                ),
            ]
        )
    return assumptions


def load_policy_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    paths = [
        project_path(args.static_config),
        project_path(args.adaptive_config),
        project_path(args.hybrid_config),
        project_path(args.selector_config),
        project_path(args.learned_config),
        project_path(args.linear_config),
        project_path(args.bandit_config),
    ]
    configs = []
    for path in paths:
        config = load_json(path)
        if config.get("type") != "paper_session":
            raise ValueError(f"{path}: config type must be paper_session")
        if should_skip_missing_policy_model(path, config):
            continue
        configs.append((path.stem, config))
    return configs


def should_skip_missing_policy_model(path: Path, config: dict[str, Any]) -> bool:
    policy = config.get("policy", {})
    if not isinstance(policy, dict) or policy.get("type") not in {
        "learned_selector",
        "linear_agent",
        "bandit_agent",
    }:
        return False
    model_path = project_path(Path(str(policy.get("model_path", ""))))
    if model_path.exists():
        return False
    print(
        f"Skipping {path.name}: {policy.get('type')} model not found at {model_path}",
        file=sys.stderr,
    )
    return True


def config_for_assumption(config: dict[str, Any], assumption: FillAssumption) -> dict[str, Any]:
    updated = dict(config)
    if assumption.base_intensity is not None:
        updated["fill_model"] = {
            "type": "touch_intensity",
            "base_intensity": assumption.base_intensity,
            "distance_decay": assumption.distance_decay,
            "volatility_boost": assumption.volatility_boost,
        }
    return updated


def write_runtime_configs(
    base_dir: Path,
    configs: list[tuple[str, dict[str, Any]]],
    assumption: FillAssumption,
) -> list[Path]:
    config_paths = []
    for name, config in configs:
        runtime_config = config_for_assumption(config, assumption)
        path = base_dir / "configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(runtime_config, indent=2, sort_keys=True) + "\n")
        config_paths.append(path)
    return config_paths


def run_policy_evaluation(
    dataset: Dataset,
    assumption: FillAssumption,
    config_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    base = project_path(args.work_dir) / assumption.name / dataset.run_id
    runs_output = base / "paper_policy_window_runs.csv"
    aggregate_output = base / "paper_policy_evaluation.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "research" / "evaluate_paper_policies.py"),
        *[str(path) for path in config_paths],
        "--data",
        str(dataset.csv_path),
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
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
    return runs_output, aggregate_output


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in column {column!r}: {row[column]!r}") from exc


def parse_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid integer in column {column!r}: {row[column]!r}") from exc


def read_window_rows(path: Path, assumption: FillAssumption, dataset: Dataset, args: argparse.Namespace) -> list[WindowRow]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        for row in reader:
            min_inventory = parse_float(row, "min_inventory")
            max_inventory = parse_float(row, "max_inventory")
            diagnostics = read_run_diagnostics(Path(row["output"]))
            final_pnl = parse_float(row, "final_pnl")
            total_fees = parse_float(row, "total_fees")
            max_drawdown = parse_float(row, "max_drawdown")
            rows.append(
                WindowRow(
                    assumption=assumption.name,
                    run_id=dataset.run_id,
                    window=row["window"],
                    config=row["config"],
                    policy=row["policy"],
                    steps=parse_int(row, "steps"),
                    fills=parse_int(row, "fills"),
                    final_pnl=final_pnl,
                    total_fees=total_fees,
                    max_drawdown=max_drawdown,
                    min_inventory=min_inventory,
                    max_inventory=max_inventory,
                    mean_abs_inventory=diagnostics.mean_abs_inventory,
                    policy_static_steps=diagnostics.policy_static_steps,
                    policy_adaptive_steps=diagnostics.policy_adaptive_steps,
                    trigger_none_steps=diagnostics.trigger_none_steps,
                    trigger_configured_steps=diagnostics.trigger_configured_steps,
                    trigger_inventory_steps=diagnostics.trigger_inventory_steps,
                    trigger_drawdown_steps=diagnostics.trigger_drawdown_steps,
                    trigger_volatility_steps=diagnostics.trigger_volatility_steps,
                    trigger_spread_steps=diagnostics.trigger_spread_steps,
                    trigger_multiple_steps=diagnostics.trigger_multiple_steps,
                    avg_spread=parse_float(row, "avg_spread"),
                    avg_quote_distance=parse_float(row, "avg_quote_distance"),
                    utility=utility(
                        final_pnl,
                        max_drawdown,
                        total_fees,
                        diagnostics.mean_abs_inventory,
                        args,
                    ),
                )
            )
    return rows


def read_run_diagnostics(path: Path) -> RunDiagnostics:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        missing = {"inventory", "policy_mode", "policy_trigger"}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: CSV missing columns: {', '.join(sorted(missing))}")

        inventories = []
        policy_static_steps = 0
        policy_adaptive_steps = 0
        trigger_counts = {
            "none": 0,
            "configured": 0,
            "inventory": 0,
            "drawdown": 0,
            "volatility": 0,
            "spread": 0,
            "multiple": 0,
        }
        for row in reader:
            inventories.append(abs(parse_float(row, "inventory")))
            if row["policy_mode"] == "adaptive":
                policy_adaptive_steps += 1
            else:
                policy_static_steps += 1
            trigger = row["policy_trigger"]
            if trigger not in trigger_counts:
                raise ValueError(f"{path}: unknown policy_trigger {trigger!r}")
            trigger_counts[trigger] += 1

    return RunDiagnostics(
        mean_abs_inventory=mean(inventories),
        policy_static_steps=policy_static_steps,
        policy_adaptive_steps=policy_adaptive_steps,
        trigger_none_steps=trigger_counts["none"],
        trigger_configured_steps=trigger_counts["configured"],
        trigger_inventory_steps=trigger_counts["inventory"],
        trigger_drawdown_steps=trigger_counts["drawdown"],
        trigger_volatility_steps=trigger_counts["volatility"],
        trigger_spread_steps=trigger_counts["spread"],
        trigger_multiple_steps=trigger_counts["multiple"],
    )


def read_aggregate_rows(
    path: Path,
    assumption: FillAssumption,
    dataset: Dataset,
    window_rows: list[WindowRow],
    args: argparse.Namespace,
) -> list[AggregateRow]:
    mean_abs_inventory_by_policy = {
        policy: mean([row.mean_abs_inventory for row in policy_rows])
        for policy, policy_rows in group_windows_by_policy(window_rows).items()
    }
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        for row in reader:
            policy = row["policy"]
            mean_pnl = parse_float(row, "mean_pnl")
            mean_fees = parse_float(row, "mean_fees")
            mean_drawdown = parse_float(row, "mean_drawdown")
            rows.append(
                AggregateRow(
                    assumption=assumption.name,
                    run_id=dataset.run_id,
                    pair=dataset.pair,
                    samples=dataset.samples,
                    interval_seconds=dataset.interval_seconds,
                    config=row["config"],
                    policy=policy,
                    windows=parse_int(row, "windows"),
                    mean_fills=parse_float(row, "mean_fills"),
                    std_fills=parse_float(row, "std_fills"),
                    mean_pnl=mean_pnl,
                    std_pnl=parse_float(row, "std_pnl"),
                    mean_fees=mean_fees,
                    mean_drawdown=mean_drawdown,
                    mean_spread=parse_float(row, "mean_spread"),
                    mean_quote_distance=parse_float(row, "mean_quote_distance"),
                    utility=utility(
                        mean_pnl,
                        mean_drawdown,
                        mean_fees,
                        mean_abs_inventory_by_policy.get(policy, 0.0),
                        args,
                    ),
                )
            )
    return rows


def utility(
    pnl: float,
    drawdown: float,
    fees: float,
    mean_abs_inventory: float,
    args: argparse.Namespace,
) -> float:
    return (
        pnl
        - args.drawdown_weight * drawdown
        - args.inventory_weight * mean_abs_inventory
        - args.fee_weight * fees
    )


def group_windows_by_policy(rows: list[WindowRow]) -> dict[str, list[WindowRow]]:
    groups: dict[str, list[WindowRow]] = {}
    for row in rows:
        groups.setdefault(row.policy, []).append(row)
    return groups


def summarize_policies(aggregate_rows: list[AggregateRow], window_rows: list[WindowRow]) -> list[PolicySummary]:
    dataset_wins = dataset_win_counts(aggregate_rows)
    window_wins = window_win_counts(window_rows)
    groups: dict[tuple[str, str], list[AggregateRow]] = {}
    for row in aggregate_rows:
        groups.setdefault((row.assumption, row.policy), []).append(row)

    summaries = []
    for (assumption, policy), rows in sorted(groups.items()):
        policy_windows = [
            row
            for row in window_rows
            if row.assumption == assumption and row.policy == policy
        ]
        total_policy_steps = sum(
            row.policy_static_steps + row.policy_adaptive_steps for row in policy_windows
        )
        summaries.append(
            PolicySummary(
                assumption=assumption,
                policy=policy,
                datasets=len({row.run_id for row in rows}),
                windows=len(policy_windows),
                mean_utility=mean([row.utility for row in rows]),
                mean_pnl=mean([row.mean_pnl for row in rows]),
                mean_drawdown=mean([row.mean_drawdown for row in rows]),
                mean_fees=mean([row.mean_fees for row in rows]),
                mean_fills=mean([row.mean_fills for row in rows]),
                mean_abs_inventory=mean(
                    [row.mean_abs_inventory for row in policy_windows]
                ),
                adaptive_step_pct=percent(
                    sum(row.policy_adaptive_steps for row in policy_windows),
                    total_policy_steps,
                ),
                trigger_configured_steps=sum(
                    row.trigger_configured_steps for row in policy_windows
                ),
                trigger_inventory_steps=sum(row.trigger_inventory_steps for row in policy_windows),
                trigger_drawdown_steps=sum(row.trigger_drawdown_steps for row in policy_windows),
                trigger_volatility_steps=sum(
                    row.trigger_volatility_steps for row in policy_windows
                ),
                trigger_spread_steps=sum(row.trigger_spread_steps for row in policy_windows),
                trigger_multiple_steps=sum(row.trigger_multiple_steps for row in policy_windows),
                dataset_utility_wins=dataset_wins.get((assumption, policy), 0),
                window_utility_wins=window_wins.get((assumption, policy), 0),
            )
        )
    return summaries


def dataset_win_counts(rows: list[AggregateRow]) -> dict[tuple[str, str], int]:
    groups: dict[tuple[str, str], list[AggregateRow]] = {}
    for row in rows:
        groups.setdefault((row.assumption, row.run_id), []).append(row)
    counts: dict[tuple[str, str], int] = {}
    for (assumption, _run_id), group in groups.items():
        winner = max(group, key=lambda row: row.utility)
        counts[(assumption, winner.policy)] = counts.get((assumption, winner.policy), 0) + 1
    return counts


def window_win_counts(rows: list[WindowRow]) -> dict[tuple[str, str], int]:
    groups: dict[tuple[str, str, str], list[WindowRow]] = {}
    for row in rows:
        groups.setdefault((row.assumption, row.run_id, row.window), []).append(row)
    counts: dict[tuple[str, str], int] = {}
    for (assumption, _run_id, _window), group in groups.items():
        winner = max(group, key=lambda row: row.utility)
        counts[(assumption, winner.policy)] = counts.get((assumption, winner.policy), 0) + 1
    return counts


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percent(part: float, total: float) -> float:
    return 100.0 * part / total if total else 0.0


def write_dataset_rows(path: Path, rows: list[AggregateRow]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "assumption",
                "run_id",
                "pair",
                "samples",
                "interval_seconds",
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
                "utility",
            ]
        )
        for row in rows:
            writer.writerow(aggregate_row_to_csv(row))


def aggregate_row_to_csv(row: AggregateRow) -> list[str]:
    return [
        row.assumption,
        row.run_id,
        row.pair,
        str(row.samples),
        format_float(row.interval_seconds),
        row.config,
        row.policy,
        str(row.windows),
        format_float(row.mean_fills),
        format_float(row.std_fills),
        format_float(row.mean_pnl),
        format_float(row.std_pnl),
        format_float(row.mean_fees),
        format_float(row.mean_drawdown),
        format_float(row.mean_spread),
        format_float(row.mean_quote_distance),
        format_float(row.utility),
    ]


def write_window_rows(path: Path, rows: list[WindowRow]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "assumption",
                "run_id",
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
                "mean_abs_inventory",
                "policy_static_steps",
                "policy_adaptive_steps",
                "trigger_none_steps",
                "trigger_configured_steps",
                "trigger_inventory_steps",
                "trigger_drawdown_steps",
                "trigger_volatility_steps",
                "trigger_spread_steps",
                "trigger_multiple_steps",
                "avg_spread",
                "avg_quote_distance",
                "utility",
            ]
        )
        for row in rows:
            writer.writerow(window_row_to_csv(row))


def window_row_to_csv(row: WindowRow) -> list[str]:
    return [
        row.assumption,
        row.run_id,
        row.window,
        row.config,
        row.policy,
        str(row.steps),
        str(row.fills),
        format_float(row.final_pnl),
        format_float(row.total_fees),
        format_float(row.max_drawdown),
        format_float(row.min_inventory),
        format_float(row.max_inventory),
        format_float(row.mean_abs_inventory),
        str(row.policy_static_steps),
        str(row.policy_adaptive_steps),
        str(row.trigger_none_steps),
        str(row.trigger_configured_steps),
        str(row.trigger_inventory_steps),
        str(row.trigger_drawdown_steps),
        str(row.trigger_volatility_steps),
        str(row.trigger_spread_steps),
        str(row.trigger_multiple_steps),
        format_float(row.avg_spread),
        format_float(row.avg_quote_distance),
        format_float(row.utility),
    ]


def write_policy_summaries(path: Path, rows: list[PolicySummary]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "assumption",
                "policy",
                "datasets",
                "windows",
                "mean_utility",
                "mean_pnl",
                "mean_drawdown",
                "mean_fees",
                "mean_fills",
                "mean_abs_inventory",
                "adaptive_step_pct",
                "trigger_configured_steps",
                "trigger_inventory_steps",
                "trigger_drawdown_steps",
                "trigger_volatility_steps",
                "trigger_spread_steps",
                "trigger_multiple_steps",
                "dataset_utility_wins",
                "window_utility_wins",
            ]
        )
        for row in rows:
            writer.writerow(policy_summary_to_csv(row))


def policy_summary_to_csv(row: PolicySummary) -> list[str]:
    return [
        row.assumption,
        row.policy,
        str(row.datasets),
        str(row.windows),
        format_float(row.mean_utility),
        format_float(row.mean_pnl),
        format_float(row.mean_drawdown),
        format_float(row.mean_fees),
        format_float(row.mean_fills),
        format_float(row.mean_abs_inventory),
        format_float(row.adaptive_step_pct),
        str(row.trigger_configured_steps),
        str(row.trigger_inventory_steps),
        str(row.trigger_drawdown_steps),
        str(row.trigger_volatility_steps),
        str(row.trigger_spread_steps),
        str(row.trigger_multiple_steps),
        str(row.dataset_utility_wins),
        str(row.window_utility_wins),
    ]


def render_report(
    datasets: list[Dataset],
    assumptions: list[FillAssumption],
    summaries: list[PolicySummary],
    args: argparse.Namespace,
) -> str:
    configured = [row for row in summaries if row.assumption == "configured"]
    configured_winner = max(configured, key=lambda row: row.mean_utility) if configured else None
    configured_hybrid = first_summary(configured, "hybrid")
    configured_selector = first_summary(configured, "selector")
    stable_winner = stability_winner(summaries, assumptions)
    policies = ", ".join(sorted({summary.policy for summary in summaries}))
    lines = [
        "# Policy Evaluation Gate",
        "",
        "## Setup",
        "",
        f"- Datasets: {len(datasets)}",
        f"- Fill assumptions: {', '.join(assumption.name for assumption in assumptions)}",
        f"- Policies: {policies}",
        f"- Window size: {args.window_size}",
        f"- Step size: {args.step_size}",
        f"- Utility: `pnl - {format_float(args.drawdown_weight)} * drawdown - "
        f"{format_float(args.inventory_weight)} * mean_abs_inventory - "
        f"{format_float(args.fee_weight)} * fees`",
        "",
        "## Gate Result",
        "",
    ]
    if configured_winner is None:
        lines.append("- No configured-assumption winner could be computed.")
    else:
        lines.append(
            f"- Best configured policy by mean utility: `{configured_winner.policy}` "
            f"({format_float(configured_winner.mean_utility)})."
        )
    if stable_winner:
        lines.append(f"- Most stable utility winner across assumptions: `{stable_winner}`.")
    else:
        lines.append("- No single policy won every fill assumption.")
    if configured_hybrid:
        lines.append(
            f"- Configured hybrid adaptive-step rate: "
            f"{format_float(configured_hybrid.adaptive_step_pct)}%; "
            f"triggers: {trigger_summary(configured_hybrid)}."
        )
    if configured_selector:
        lines.append(
            f"- Configured selector adaptive-step rate: "
            f"{format_float(configured_selector.adaptive_step_pct)}%; "
            f"triggers: {trigger_summary(configured_selector)}."
        )
    lines.extend(
        [
            "- Interpretation: keep these as baselines; do not claim alpha yet.",
            "",
            "## Policy Summary",
            "",
            "| assumption | policy | utility | pnl | drawdown | fills | adaptive_step_pct | triggers | dataset_wins | window_wins |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in sorted(summaries, key=lambda item: (item.assumption, -item.mean_utility)):
        lines.append(
            "| "
            f"{row.assumption} | {row.policy} | {format_float(row.mean_utility)} | "
            f"{format_float(row.mean_pnl)} | {format_float(row.mean_drawdown)} | "
            f"{format_float(row.mean_fills)} | {format_float(row.adaptive_step_pct)} | "
            f"{trigger_summary(row)} | {row.dataset_utility_wins} | "
            f"{row.window_utility_wins} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- If static wins utility, the adaptive logic is still mainly a risk-control experiment.",
            "- If adaptive wins utility but loses PnL, it is reducing risk at a measurable cost.",
            "- If hybrid wins utility, the risk-trigger idea is worth tuning next.",
            "- If selector wins utility, the agentic policy-selection path is worth extending.",
            "- If selector loses, inspect its trigger attribution before changing weights.",
            "",
        ]
    )
    return "\n".join(lines)


def stability_winner(summaries: list[PolicySummary], assumptions: list[FillAssumption]) -> str | None:
    winners = []
    for assumption in assumptions:
        rows = [row for row in summaries if row.assumption == assumption.name]
        if rows:
            winners.append(max(rows, key=lambda row: row.mean_utility).policy)
    if winners and len(set(winners)) == 1:
        return winners[0]
    return None


def first_summary(rows: list[PolicySummary], policy: str) -> PolicySummary | None:
    for row in rows:
        if row.policy == policy:
            return row
    return None


def trigger_summary(row: PolicySummary) -> str:
    parts = [
        ("configured", row.trigger_configured_steps),
        ("inventory", row.trigger_inventory_steps),
        ("drawdown", row.trigger_drawdown_steps),
        ("volatility", row.trigger_volatility_steps),
        ("spread", row.trigger_spread_steps),
        ("multiple", row.trigger_multiple_steps),
    ]
    active = [f"{name}:{count}" for name, count in parts if count]
    return ", ".join(active) if active else "none"


def format_float(value: float) -> str:
    return f"{value:.6f}"


def print_table(summaries: list[PolicySummary]) -> None:
    headers = [
        "assumption",
        "policy",
        "utility",
        "pnl",
        "drawdown",
        "fills",
        "adapt_pct",
        "triggers",
        "ds_wins",
        "win_wins",
    ]
    rows = [
        [
            row.assumption,
            row.policy,
            f"{row.mean_utility:.5f}",
            f"{row.mean_pnl:.5f}",
            f"{row.mean_drawdown:.5f}",
            f"{row.mean_fills:.2f}",
            f"{row.adaptive_step_pct:.1f}",
            trigger_summary(row),
            str(row.dataset_utility_wins),
            str(row.window_utility_wins),
        ]
        for row in sorted(summaries, key=lambda item: (item.assumption, -item.mean_utility))
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    print("Policy evaluation gate")
    print("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        metadata_paths = discover_metadata(args.metadata_paths)
        if not metadata_paths:
            raise ValueError("no quote-dataset metadata files found")

        datasets = [load_dataset(path) for path in metadata_paths]
        policy_configs = load_policy_configs(args)
        assumptions = fill_assumptions(args.include_fill_sensitivity)

        aggregate_rows: list[AggregateRow] = []
        window_rows: list[WindowRow] = []
        for assumption in assumptions:
            for dataset in datasets:
                print(f"Evaluating {dataset.run_id} under {assumption.name}", flush=True)
                base = project_path(args.work_dir) / assumption.name / dataset.run_id
                config_paths = write_runtime_configs(base, policy_configs, assumption)
                runs_output, aggregate_output = run_policy_evaluation(
                    dataset,
                    assumption,
                    config_paths,
                    args,
                )
                dataset_windows = read_window_rows(runs_output, assumption, dataset, args)
                window_rows.extend(dataset_windows)
                aggregate_rows.extend(
                    read_aggregate_rows(aggregate_output, assumption, dataset, dataset_windows, args)
                )

        summaries = summarize_policies(aggregate_rows, window_rows)
        write_dataset_rows(args.dataset_output, aggregate_rows)
        write_window_rows(args.window_output, window_rows)
        write_policy_summaries(args.policy_output, summaries)
        report_path = project_path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(datasets, assumptions, summaries, args))

        print_table(summaries)
        print()
        print(f"wrote {project_path(args.dataset_output)}")
        print(f"wrote {project_path(args.window_output)}")
        print(f"wrote {project_path(args.policy_output)}")
        print(f"wrote {report_path}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

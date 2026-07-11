#!/usr/bin/env python3
"""Sweep weighted selector paper-policy parameters across quote datasets."""

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

import evaluate_paper_policies


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATTERN = "data/quotes/*.meta.json"
DEFAULT_BASE_CONFIG = Path("configs/runs/kraken_solusd_selector_maker_fee_paper_session.json")
DEFAULT_ADAPTIVE_CONFIG = Path("configs/runs/kraken_solusd_adaptive_maker_fee_paper_session.json")
DEFAULT_STATIC_CONFIG = Path("configs/runs/kraken_solusd_maker_fee_paper_session.json")
DEFAULT_WORK_DIR = Path("target/research/selector_policy_sweep")
DEFAULT_VARIANTS_OUTPUT = Path("target/research/selector_policy_sweep.csv")
DEFAULT_RUNS_OUTPUT = Path("target/research/selector_policy_sweep_runs.csv")
DEFAULT_REPORT_OUTPUT = Path("target/research/selector_policy_sweep.md")


@dataclass(frozen=True)
class Dataset:
    run_id: str
    pair: str
    csv_path: Path


@dataclass(frozen=True)
class Variant:
    name: str
    volatility_weight: float
    spread_weight: float
    inventory_weight: float
    drawdown_weight: float
    activation_threshold: float


@dataclass(frozen=True)
class RunDiagnostics:
    mean_abs_inventory: float
    policy_static_steps: int
    policy_adaptive_steps: int
    trigger_inventory_steps: int
    trigger_drawdown_steps: int
    trigger_volatility_steps: int
    trigger_spread_steps: int
    trigger_multiple_steps: int


@dataclass(frozen=True)
class WindowResult:
    variant: str
    dataset: str
    window: str
    fills: int
    pnl: float
    fees: float
    drawdown: float
    mean_abs_inventory: float
    adaptive_step_pct: float
    trigger_inventory_steps: int
    trigger_drawdown_steps: int
    trigger_volatility_steps: int
    trigger_spread_steps: int
    trigger_multiple_steps: int
    utility: float
    output: Path


@dataclass(frozen=True)
class VariantSummary:
    variant: Variant
    datasets: int
    windows: int
    mean_utility: float
    mean_pnl: float
    pnl_std: float
    mean_fees: float
    mean_drawdown: float
    mean_fills: float
    mean_abs_inventory: float
    adaptive_step_pct: float
    trigger_inventory_steps: int
    trigger_drawdown_steps: int
    trigger_volatility_steps: int
    trigger_spread_steps: int
    trigger_multiple_steps: int
    score: float


@dataclass(frozen=True)
class BaselineSummary:
    name: str
    datasets: int
    windows: int
    mean_utility: float
    mean_pnl: float
    mean_drawdown: float
    mean_fills: float
    adaptive_step_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep selector policy weights across collected quote datasets."
    )
    parser.add_argument(
        "metadata_paths",
        nargs="*",
        type=Path,
        help="Quote-dataset metadata files. Defaults to data/quotes/*.meta.json.",
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--adaptive-config", type=Path, default=DEFAULT_ADAPTIVE_CONFIG)
    parser.add_argument("--static-config", type=Path, default=DEFAULT_STATIC_CONFIG)
    parser.add_argument("--window-size", type=int, default=30, help="Rows per window.")
    parser.add_argument("--step-size", type=int, default=30, help="Rows between window starts.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum windows per dataset. Use 0 for all windows.",
    )
    parser.add_argument(
        "--activation-thresholds",
        default="0.08,0.12,0.16",
        help="Comma-separated selector activation thresholds.",
    )
    parser.add_argument(
        "--volatility-weights",
        default="6,12",
        help="Comma-separated volatility weights.",
    )
    parser.add_argument(
        "--spread-weights",
        default="2,5",
        help="Comma-separated observed-spread weights.",
    )
    parser.add_argument(
        "--inventory-weights",
        default="0.2,0.4",
        help="Comma-separated inventory weights.",
    )
    parser.add_argument(
        "--drawdown-weights",
        default="10",
        help="Comma-separated drawdown weights.",
    )
    parser.add_argument("--utility-drawdown-weight", type=float, default=2.0)
    parser.add_argument("--utility-inventory-weight", type=float, default=0.02)
    parser.add_argument("--utility-fee-weight", type=float, default=0.0)
    parser.add_argument(
        "--min-adaptive-pct",
        type=float,
        default=20.0,
        help="Lower healthy adaptive-step percentage for scoring.",
    )
    parser.add_argument(
        "--max-adaptive-pct",
        type=float,
        default=80.0,
        help="Upper healthy adaptive-step percentage for scoring.",
    )
    parser.add_argument(
        "--adaptive-rate-penalty",
        type=float,
        default=0.00005,
        help="Score penalty per percentage point outside adaptive-rate bounds.",
    )
    parser.add_argument("--min-mean-fills", type=float, default=3.0)
    parser.add_argument("--fill-penalty", type=float, default=0.0005)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--variants-output", type=Path, default=DEFAULT_VARIANTS_OUTPUT)
    parser.add_argument("--runs-output", type=Path, default=DEFAULT_RUNS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
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
    csv_path = project_path(Path(str(metadata.get("csv_path", ""))))
    if not csv_path.exists():
        raise ValueError(f"{path}: quote CSV not found: {csv_path}")
    return Dataset(
        run_id=str(metadata.get("run_id") or path.name.replace(".meta.json", "")),
        pair=str(metadata.get("pair") or ""),
        csv_path=csv_path,
    )


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
    if args.min_adaptive_pct < 0.0 or args.max_adaptive_pct > 100.0:
        raise ValueError("adaptive percentage bounds must be within [0, 100]")
    if args.min_adaptive_pct > args.max_adaptive_pct:
        raise ValueError("--min-adaptive-pct must be <= --max-adaptive-pct")


def token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants = []
    for threshold in parse_float_list(args.activation_thresholds, "--activation-thresholds"):
        for volatility in parse_float_list(args.volatility_weights, "--volatility-weights"):
            for spread in parse_float_list(args.spread_weights, "--spread-weights"):
                for inventory in parse_float_list(args.inventory_weights, "--inventory-weights"):
                    for drawdown in parse_float_list(args.drawdown_weights, "--drawdown-weights"):
                        name = (
                            f"selector_thr_{token(threshold)}"
                            f"_vol_{token(volatility)}"
                            f"_spr_{token(spread)}"
                            f"_inv_{token(inventory)}"
                            f"_dd_{token(drawdown)}"
                        )
                        variants.append(
                            Variant(
                                name=name,
                                volatility_weight=volatility,
                                spread_weight=spread,
                                inventory_weight=inventory,
                                drawdown_weight=drawdown,
                                activation_threshold=threshold,
                            )
                        )
    return variants


def variant_config(base_config: dict[str, Any], variant: Variant) -> dict[str, Any]:
    config = dict(base_config)
    policy = dict(config["policy"])
    policy.update(
        {
            "type": "selector",
            "volatility_weight": variant.volatility_weight,
            "spread_weight": variant.spread_weight,
            "inventory_weight": variant.inventory_weight,
            "drawdown_weight": variant.drawdown_weight,
            "activation_threshold": variant.activation_threshold,
        }
    )
    config["policy"] = policy
    return config


def dataset_work_dir(args: argparse.Namespace, dataset: Dataset) -> Path:
    return project_path(args.work_dir) / dataset.run_id


def runtime_config_path(work_dir: Path, window: evaluate_paper_policies.Window, name: str) -> Path:
    return work_dir / "configs" / f"{window.name}_{name}.json"


def runtime_output_path(work_dir: Path, window: evaluate_paper_policies.Window, name: str) -> Path:
    return work_dir / "runs" / f"{window.name}_{name}.csv"


def write_runtime_config(
    path: Path,
    config: dict[str, Any],
    window: evaluate_paper_policies.Window,
    output_path: Path,
) -> None:
    runtime_config = dict(config)
    runtime_config["data"] = str(window.path)
    runtime_config["output"] = str(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_config, indent=2, sort_keys=True) + "\n")


def run_config(path: Path, verbose: bool) -> None:
    command = ["cargo", "run", "--quiet", "--", "run", str(path)]
    if verbose:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    else:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)


def read_run_diagnostics(path: Path) -> RunDiagnostics:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        missing = {"inventory", "policy_mode", "policy_trigger"}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

        inventories = []
        static_steps = 0
        adaptive_steps = 0
        triggers = {
            "inventory": 0,
            "drawdown": 0,
            "volatility": 0,
            "spread": 0,
            "multiple": 0,
        }
        for row in reader:
            inventories.append(abs(parse_float(row, "inventory")))
            if row["policy_mode"] == "adaptive":
                adaptive_steps += 1
            else:
                static_steps += 1
            trigger = row["policy_trigger"]
            if trigger in triggers:
                triggers[trigger] += 1

    return RunDiagnostics(
        mean_abs_inventory=mean(inventories),
        policy_static_steps=static_steps,
        policy_adaptive_steps=adaptive_steps,
        trigger_inventory_steps=triggers["inventory"],
        trigger_drawdown_steps=triggers["drawdown"],
        trigger_volatility_steps=triggers["volatility"],
        trigger_spread_steps=triggers["spread"],
        trigger_multiple_steps=triggers["multiple"],
    )


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in {column}: {row[column]!r}") from exc


def run_policy_on_windows(
    *,
    name: str,
    config: dict[str, Any],
    dataset: Dataset,
    windows: list[evaluate_paper_policies.Window],
    args: argparse.Namespace,
) -> list[WindowResult]:
    work_dir = dataset_work_dir(args, dataset)
    results = []
    for window in windows:
        output_path = runtime_output_path(work_dir, window, name)
        runtime_path = runtime_config_path(work_dir, window, name)
        write_runtime_config(runtime_path, config, window, output_path)
        run_config(runtime_path, args.verbose)
        summary = evaluate_paper_policies.summarize_output(
            window,
            Path(name),
            config,
            output_path,
        )
        diagnostics = read_run_diagnostics(output_path)
        total_steps = diagnostics.policy_static_steps + diagnostics.policy_adaptive_steps
        utility = run_utility(
            summary.final_pnl,
            summary.max_drawdown,
            summary.total_fees,
            diagnostics.mean_abs_inventory,
            args,
        )
        results.append(
            WindowResult(
                variant=name,
                dataset=dataset.run_id,
                window=window.name,
                fills=summary.fills,
                pnl=summary.final_pnl,
                fees=summary.total_fees,
                drawdown=summary.max_drawdown,
                mean_abs_inventory=diagnostics.mean_abs_inventory,
                adaptive_step_pct=percent(diagnostics.policy_adaptive_steps, total_steps),
                trigger_inventory_steps=diagnostics.trigger_inventory_steps,
                trigger_drawdown_steps=diagnostics.trigger_drawdown_steps,
                trigger_volatility_steps=diagnostics.trigger_volatility_steps,
                trigger_spread_steps=diagnostics.trigger_spread_steps,
                trigger_multiple_steps=diagnostics.trigger_multiple_steps,
                utility=utility,
                output=output_path,
            )
        )
    return results


def run_utility(
    pnl: float,
    drawdown: float,
    fees: float,
    mean_abs_inventory: float,
    args: argparse.Namespace,
) -> float:
    return (
        pnl
        - args.utility_drawdown_weight * drawdown
        - args.utility_inventory_weight * mean_abs_inventory
        - args.utility_fee_weight * fees
    )


def summarize_variant(variant: Variant, results: list[WindowResult], args: argparse.Namespace) -> VariantSummary:
    if not results:
        raise ValueError(f"{variant.name}: no results")
    utilities = [row.utility for row in results]
    adaptive_pct = mean([row.adaptive_step_pct for row in results])
    adaptive_penalty = adaptive_rate_penalty(adaptive_pct, args)
    fill_shortfall = max(0.0, args.min_mean_fills - mean([float(row.fills) for row in results]))
    fill_penalty = args.fill_penalty * fill_shortfall
    score = mean(utilities) - adaptive_penalty - fill_penalty
    return VariantSummary(
        variant=variant,
        datasets=len({row.dataset for row in results}),
        windows=len(results),
        mean_utility=mean(utilities),
        mean_pnl=mean([row.pnl for row in results]),
        pnl_std=sample_std([row.pnl for row in results]),
        mean_fees=mean([row.fees for row in results]),
        mean_drawdown=mean([row.drawdown for row in results]),
        mean_fills=mean([float(row.fills) for row in results]),
        mean_abs_inventory=mean([row.mean_abs_inventory for row in results]),
        adaptive_step_pct=adaptive_pct,
        trigger_inventory_steps=sum(row.trigger_inventory_steps for row in results),
        trigger_drawdown_steps=sum(row.trigger_drawdown_steps for row in results),
        trigger_volatility_steps=sum(row.trigger_volatility_steps for row in results),
        trigger_spread_steps=sum(row.trigger_spread_steps for row in results),
        trigger_multiple_steps=sum(row.trigger_multiple_steps for row in results),
        score=score,
    )


def adaptive_rate_penalty(adaptive_pct: float, args: argparse.Namespace) -> float:
    if adaptive_pct < args.min_adaptive_pct:
        return (args.min_adaptive_pct - adaptive_pct) * args.adaptive_rate_penalty
    if adaptive_pct > args.max_adaptive_pct:
        return (adaptive_pct - args.max_adaptive_pct) * args.adaptive_rate_penalty
    return 0.0


def summarize_baseline(name: str, results: list[WindowResult]) -> BaselineSummary:
    return BaselineSummary(
        name=name,
        datasets=len({row.dataset for row in results}),
        windows=len(results),
        mean_utility=mean([row.utility for row in results]),
        mean_pnl=mean([row.pnl for row in results]),
        mean_drawdown=mean([row.drawdown for row in results]),
        mean_fills=mean([float(row.fills) for row in results]),
        adaptive_step_pct=mean([row.adaptive_step_pct for row in results]),
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def percent(part: float, total: float) -> float:
    return 100.0 * part / total if total else 0.0


def format_float(value: float) -> str:
    return f"{value:.6f}"


def trigger_summary(summary: VariantSummary) -> str:
    parts = [
        ("inventory", summary.trigger_inventory_steps),
        ("drawdown", summary.trigger_drawdown_steps),
        ("volatility", summary.trigger_volatility_steps),
        ("spread", summary.trigger_spread_steps),
        ("multiple", summary.trigger_multiple_steps),
    ]
    active = [f"{name}:{count}" for name, count in parts if count]
    return ", ".join(active) if active else "none"


def write_variant_summaries(path: Path, summaries: list[VariantSummary]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "variant",
                "datasets",
                "windows",
                "activation_threshold",
                "volatility_weight",
                "spread_weight",
                "inventory_weight",
                "drawdown_weight",
                "score",
                "mean_utility",
                "mean_pnl",
                "pnl_std",
                "mean_fees",
                "mean_drawdown",
                "mean_fills",
                "mean_abs_inventory",
                "adaptive_step_pct",
                "trigger_inventory_steps",
                "trigger_drawdown_steps",
                "trigger_volatility_steps",
                "trigger_spread_steps",
                "trigger_multiple_steps",
            ]
        )
        for rank, summary in enumerate(summaries, start=1):
            writer.writerow(variant_summary_row(rank, summary))


def variant_summary_row(rank: int, summary: VariantSummary) -> list[str]:
    variant = summary.variant
    return [
        str(rank),
        variant.name,
        str(summary.datasets),
        str(summary.windows),
        format_float(variant.activation_threshold),
        format_float(variant.volatility_weight),
        format_float(variant.spread_weight),
        format_float(variant.inventory_weight),
        format_float(variant.drawdown_weight),
        format_float(summary.score),
        format_float(summary.mean_utility),
        format_float(summary.mean_pnl),
        format_float(summary.pnl_std),
        format_float(summary.mean_fees),
        format_float(summary.mean_drawdown),
        format_float(summary.mean_fills),
        format_float(summary.mean_abs_inventory),
        format_float(summary.adaptive_step_pct),
        str(summary.trigger_inventory_steps),
        str(summary.trigger_drawdown_steps),
        str(summary.trigger_volatility_steps),
        str(summary.trigger_spread_steps),
        str(summary.trigger_multiple_steps),
    ]


def write_window_results(path: Path, results: list[WindowResult]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "dataset",
                "window",
                "fills",
                "pnl",
                "fees",
                "drawdown",
                "mean_abs_inventory",
                "adaptive_step_pct",
                "trigger_inventory_steps",
                "trigger_drawdown_steps",
                "trigger_volatility_steps",
                "trigger_spread_steps",
                "trigger_multiple_steps",
                "utility",
                "output",
            ]
        )
        for row in results:
            writer.writerow(window_result_row(row))


def window_result_row(row: WindowResult) -> list[str]:
    return [
        row.variant,
        row.dataset,
        row.window,
        str(row.fills),
        format_float(row.pnl),
        format_float(row.fees),
        format_float(row.drawdown),
        format_float(row.mean_abs_inventory),
        format_float(row.adaptive_step_pct),
        str(row.trigger_inventory_steps),
        str(row.trigger_drawdown_steps),
        str(row.trigger_volatility_steps),
        str(row.trigger_spread_steps),
        str(row.trigger_multiple_steps),
        format_float(row.utility),
        str(row.output),
    ]


def render_report(
    summaries: list[VariantSummary],
    baselines: list[BaselineSummary],
    args: argparse.Namespace,
) -> str:
    best = summaries[0] if summaries else None
    adaptive = next((baseline for baseline in baselines if baseline.name == "adaptive"), None)
    lines = [
        "# Selector Policy Sweep",
        "",
        "## Setup",
        "",
        f"- Variants: {len(summaries)}",
        f"- Healthy adaptive-step band: {format_float(args.min_adaptive_pct)}% to {format_float(args.max_adaptive_pct)}%",
        f"- Score: mean utility minus adaptive-rate and low-fill penalties",
        "",
        "## Result",
        "",
    ]
    if best:
        lines.append(
            f"- Best selector: `{best.variant.name}` with score {format_float(best.score)} "
            f"and utility {format_float(best.mean_utility)}."
        )
        lines.append(
            f"- Best selector adaptive-step rate: {format_float(best.adaptive_step_pct)}%; "
            f"triggers: {trigger_summary(best)}."
        )
    if best and adaptive:
        delta = best.mean_utility - adaptive.mean_utility
        lines.append(f"- Best selector utility minus adaptive baseline: {format_float(delta)}.")
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| baseline | utility | pnl | drawdown | fills | adaptive_step_pct |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in baselines:
        lines.append(
            "| "
            f"{baseline.name} | {format_float(baseline.mean_utility)} | "
            f"{format_float(baseline.mean_pnl)} | {format_float(baseline.mean_drawdown)} | "
            f"{format_float(baseline.mean_fills)} | {format_float(baseline.adaptive_step_pct)} |"
        )
    lines.extend(
        [
            "",
            "## Top Variants",
            "",
            "| rank | variant | score | utility | pnl | drawdown | fills | adaptive_step_pct | triggers |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, summary in enumerate(summaries[: args.top], start=1):
        lines.append(
            "| "
            f"{rank} | {summary.variant.name} | {format_float(summary.score)} | "
            f"{format_float(summary.mean_utility)} | {format_float(summary.mean_pnl)} | "
            f"{format_float(summary.mean_drawdown)} | {format_float(summary.mean_fills)} | "
            f"{format_float(summary.adaptive_step_pct)} | {trigger_summary(summary)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A good selector should beat adaptive utility while using adaptive less than 100% of the time.",
            "- If the best selector is close to adaptive but not better, the selector surface is useful but needs richer features or learned weights.",
            "- Treat this as model selection evidence, not a production trading claim.",
            "",
        ]
    )
    return "\n".join(lines)


def print_table(summaries: list[VariantSummary], baselines: list[BaselineSummary], top: int) -> None:
    print("Selector policy sweep baselines")
    for baseline in baselines:
        print(
            f"{baseline.name:>8} "
            f"utility={baseline.mean_utility:.5f} "
            f"pnl={baseline.mean_pnl:.5f} "
            f"drawdown={baseline.mean_drawdown:.5f} "
            f"fills={baseline.mean_fills:.2f} "
            f"adaptive={baseline.adaptive_step_pct:.1f}%"
        )
    print()

    headers = ["rank", "threshold", "vol", "spread", "inv", "score", "utility", "pnl", "dd", "fills", "adapt"]
    rows = []
    for rank, summary in enumerate(summaries[:top], start=1):
        variant = summary.variant
        rows.append(
            [
                str(rank),
                f"{variant.activation_threshold:.2f}",
                f"{variant.volatility_weight:.1f}",
                f"{variant.spread_weight:.1f}",
                f"{variant.inventory_weight:.1f}",
                f"{summary.score:.5f}",
                f"{summary.mean_utility:.5f}",
                f"{summary.mean_pnl:.5f}",
                f"{summary.mean_drawdown:.5f}",
                f"{summary.mean_fills:.2f}",
                f"{summary.adaptive_step_pct:.1f}%",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("Selector policy sweep")
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

        base_config = evaluate_paper_policies.load_config(project_path(args.base_config))
        adaptive_config = evaluate_paper_policies.load_config(project_path(args.adaptive_config))
        static_config = evaluate_paper_policies.load_config(project_path(args.static_config))
        variants = build_variants(args)

        windows_by_dataset = {
            dataset.run_id: evaluate_paper_policies.build_windows(
                dataset.csv_path,
                dataset_work_dir(args, dataset),
                args.window_size,
                args.step_size,
                args.max_windows,
            )
            for dataset in datasets
        }

        all_window_results: list[WindowResult] = []
        baseline_results: dict[str, list[WindowResult]] = {"adaptive": [], "static": []}
        for dataset in datasets:
            windows = windows_by_dataset[dataset.run_id]
            print(f"Running baselines for {dataset.run_id}", flush=True)
            baseline_results["adaptive"].extend(
                run_policy_on_windows(
                    name="adaptive",
                    config=adaptive_config,
                    dataset=dataset,
                    windows=windows,
                    args=args,
                )
            )
            baseline_results["static"].extend(
                run_policy_on_windows(
                    name="static",
                    config=static_config,
                    dataset=dataset,
                    windows=windows,
                    args=args,
                )
            )

        variant_summaries = []
        for index, variant in enumerate(variants, start=1):
            print(f"Running variant {index}/{len(variants)} {variant.name}", flush=True)
            config = variant_config(base_config, variant)
            results = []
            for dataset in datasets:
                results.extend(
                    run_policy_on_windows(
                        name=variant.name,
                        config=config,
                        dataset=dataset,
                        windows=windows_by_dataset[dataset.run_id],
                        args=args,
                    )
                )
            all_window_results.extend(results)
            variant_summaries.append(summarize_variant(variant, results, args))

        variant_summaries.sort(key=lambda summary: summary.score, reverse=True)
        baselines = [
            summarize_baseline("adaptive", baseline_results["adaptive"]),
            summarize_baseline("static", baseline_results["static"]),
        ]
        write_variant_summaries(args.variants_output, variant_summaries)
        write_window_results(args.runs_output, all_window_results)
        report_path = project_path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(variant_summaries, baselines, args))

        print_table(variant_summaries, baselines, args.top)
        print()
        print(f"wrote {project_path(args.variants_output)}")
        print(f"wrote {project_path(args.runs_output)}")
        print(f"wrote {report_path}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

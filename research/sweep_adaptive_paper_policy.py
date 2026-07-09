#!/usr/bin/env python3
"""Sweep adaptive paper-policy parameters across quote windows."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_paper_policies


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = Path("configs/runs/kraken_solusd_adaptive_maker_fee_paper_session.json")
DEFAULT_BASELINE_CONFIG = Path("configs/runs/kraken_solusd_maker_fee_paper_session.json")
DEFAULT_WORK_DIR = Path("target/research/adaptive_policy_sweep")
DEFAULT_VARIANTS_OUTPUT = Path("target/research/adaptive_policy_sweep_variants.csv")
DEFAULT_RUNS_OUTPUT = Path("target/research/adaptive_policy_sweep_runs.csv")


@dataclass(frozen=True)
class Variant:
    name: str
    min_spread: float
    max_spread: float
    volatility_spread_multiplier: float
    inventory_skew_multiplier: float
    touch_spread_multiplier: float


@dataclass(frozen=True)
class VariantSummary:
    variant: Variant
    windows: int
    mean_fills: float
    mean_pnl: float
    pnl_std: float
    mean_fees: float
    mean_drawdown: float
    mean_spread: float
    mean_quote_distance: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an adaptive paper-policy parameter sweep over quote windows."
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Quote CSV to window. Defaults to the data path in --base-config.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Adaptive paper_session config used as the template.",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG,
        help="Optional static baseline config to include in the run summary.",
    )
    parser.add_argument("--window-size", type=int, default=30, help="Rows per window.")
    parser.add_argument("--step-size", type=int, default=30, help="Rows between window starts.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum windows to evaluate. Use 0 for all windows.",
    )
    parser.add_argument(
        "--touch-spread-multipliers",
        default="0.2,0.4,0.6",
        help="Comma-separated touch-spread multipliers.",
    )
    parser.add_argument(
        "--volatility-spread-multipliers",
        default="0.0,0.5,1.0,2.0",
        help="Comma-separated volatility-spread multipliers.",
    )
    parser.add_argument(
        "--inventory-skew-multipliers",
        default="0.0,0.02,0.05",
        help="Comma-separated inventory-skew multipliers.",
    )
    parser.add_argument(
        "--max-spreads",
        default="0.04,0.08,0.12",
        help="Comma-separated max spread values.",
    )
    parser.add_argument("--min-spread", type=float, default=0.02, help="Minimum spread.")
    parser.add_argument(
        "--drawdown-weight",
        type=float,
        default=0.5,
        help="Score penalty per unit of mean max drawdown.",
    )
    parser.add_argument(
        "--min-mean-fills",
        type=float,
        default=3.0,
        help="Mean fills target below which the score is penalized.",
    )
    parser.add_argument(
        "--fill-penalty",
        type=float,
        default=0.001,
        help="Score penalty per missing mean fill below --min-mean-fills.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory for generated windows, configs, and run outputs.",
    )
    parser.add_argument(
        "--variants-output",
        type=Path,
        default=DEFAULT_VARIANTS_OUTPUT,
        help="Ranked variant summary CSV.",
    )
    parser.add_argument(
        "--runs-output",
        type=Path,
        default=DEFAULT_RUNS_OUTPUT,
        help="Per-window run summary CSV.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of ranked variants to print.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each paper-session run output.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_float_list(value: str, name: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(not item >= 0.0 for item in values):
        raise ValueError(f"{name} must contain non-negative values")
    return values


def format_token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_variants(args: argparse.Namespace) -> list[Variant]:
    if args.min_spread <= 0.0:
        raise ValueError("--min-spread must be positive")

    variants = []
    for touch in parse_float_list(args.touch_spread_multipliers, "--touch-spread-multipliers"):
        for vol in parse_float_list(args.volatility_spread_multipliers, "--volatility-spread-multipliers"):
            for skew in parse_float_list(args.inventory_skew_multipliers, "--inventory-skew-multipliers"):
                for max_spread in parse_float_list(args.max_spreads, "--max-spreads"):
                    if max_spread < args.min_spread:
                        raise ValueError("--max-spreads values must be greater than or equal to --min-spread")
                    name = (
                        f"adaptive_touch_{format_token(touch)}"
                        f"_vol_{format_token(vol)}"
                        f"_skew_{format_token(skew)}"
                        f"_max_{format_token(max_spread)}"
                    )
                    variants.append(
                        Variant(
                            name=name,
                            min_spread=args.min_spread,
                            max_spread=max_spread,
                            volatility_spread_multiplier=vol,
                            inventory_skew_multiplier=skew,
                            touch_spread_multiplier=touch,
                        )
                    )
    return variants


def variant_config(base_config: dict[str, Any], variant: Variant) -> dict[str, Any]:
    config = dict(base_config)
    config["policy"] = {
        "type": "adaptive",
        "min_spread": variant.min_spread,
        "max_spread": variant.max_spread,
        "volatility_spread_multiplier": variant.volatility_spread_multiplier,
        "inventory_skew_multiplier": variant.inventory_skew_multiplier,
        "touch_spread_multiplier": variant.touch_spread_multiplier,
    }
    return config


def runtime_config_path(work_dir: Path, window: evaluate_paper_policies.Window, name: str) -> Path:
    return project_path(work_dir) / "configs" / f"{window.name}_{name}.json"


def runtime_output_path(work_dir: Path, window: evaluate_paper_policies.Window, name: str) -> Path:
    return project_path(work_dir) / "runs" / f"{window.name}_{name}.csv"


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
    if verbose:
        subprocess.run(["cargo", "run", "--quiet", "--", "run", str(path)], cwd=PROJECT_ROOT, check=True)
    else:
        subprocess.run(
            ["cargo", "run", "--quiet", "--", "run", str(path)],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )


def summarize_variant(
    variant: Variant,
    runs: list[evaluate_paper_policies.RunSummary],
    drawdown_weight: float,
    min_mean_fills: float,
    fill_penalty: float,
) -> VariantSummary:
    aggregate = evaluate_paper_policies.aggregate_runs(runs)[0]
    fill_shortfall = max(0.0, min_mean_fills - aggregate.mean_fills)
    score = (
        aggregate.mean_pnl
        - drawdown_weight * aggregate.mean_drawdown
        - fill_penalty * fill_shortfall
    )
    return VariantSummary(
        variant=variant,
        windows=aggregate.windows,
        mean_fills=aggregate.mean_fills,
        mean_pnl=aggregate.mean_pnl,
        pnl_std=aggregate.std_pnl,
        mean_fees=aggregate.mean_fees,
        mean_drawdown=aggregate.mean_drawdown,
        mean_spread=aggregate.mean_spread,
        mean_quote_distance=aggregate.mean_quote_distance,
        score=score,
    )


def write_variant_summaries(path: Path, summaries: list[VariantSummary]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "variant",
                "windows",
                "touch_spread_multiplier",
                "volatility_spread_multiplier",
                "inventory_skew_multiplier",
                "min_spread",
                "max_spread",
                "mean_fills",
                "mean_pnl",
                "pnl_std",
                "mean_fees",
                "mean_drawdown",
                "mean_spread",
                "mean_quote_distance",
                "score",
            ]
        )
        for rank, summary in enumerate(summaries, start=1):
            writer.writerow(variant_summary_row(rank, summary))


def variant_summary_row(rank: int, summary: VariantSummary) -> list[str]:
    variant = summary.variant
    return [
        str(rank),
        variant.name,
        str(summary.windows),
        format_float(variant.touch_spread_multiplier),
        format_float(variant.volatility_spread_multiplier),
        format_float(variant.inventory_skew_multiplier),
        format_float(variant.min_spread),
        format_float(variant.max_spread),
        format_float(summary.mean_fills),
        format_float(summary.mean_pnl),
        format_float(summary.pnl_std),
        format_float(summary.mean_fees),
        format_float(summary.mean_drawdown),
        format_float(summary.mean_spread),
        format_float(summary.mean_quote_distance),
        format_float(summary.score),
    ]


def write_run_summaries(path: Path, runs: list[evaluate_paper_policies.RunSummary]) -> None:
    evaluate_paper_policies.write_run_summaries(project_path(path), runs)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def render_table(summaries: list[VariantSummary], top: int) -> str:
    headers = ["rank", "touch", "vol", "skew", "max", "fills", "pnl", "pnl_std", "drawdown", "spread", "score"]
    rows = []
    for rank, summary in enumerate(summaries[:top], start=1):
        variant = summary.variant
        rows.append(
            [
                str(rank),
                f"{variant.touch_spread_multiplier:.2f}",
                f"{variant.volatility_spread_multiplier:.2f}",
                f"{variant.inventory_skew_multiplier:.2f}",
                f"{variant.max_spread:.2f}",
                f"{summary.mean_fills:.2f}",
                f"{summary.mean_pnl:.4f}",
                f"{summary.pnl_std:.4f}",
                f"{summary.mean_drawdown:.4f}",
                f"{summary.mean_spread:.4f}",
                f"{summary.score:.4f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = ["Adaptive policy sweep"]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        base_config_path = project_path(args.base_config)
        base_config = evaluate_paper_policies.load_config(base_config_path)
        data_path = project_path(args.data) if args.data else project_path(Path(base_config["data"]))
        windows = evaluate_paper_policies.build_windows(
            data_path,
            args.work_dir,
            args.window_size,
            args.step_size,
            args.max_windows,
        )
        variants = build_variants(args)
        if args.top <= 0:
            raise ValueError("--top must be positive")

        all_runs: list[evaluate_paper_policies.RunSummary] = []
        variant_summaries = []
        for index, variant in enumerate(variants, start=1):
            config = variant_config(base_config, variant)
            variant_runs = []
            print(f"Running variant {index}/{len(variants)} {variant.name}", flush=True)
            for window in windows:
                output_path = runtime_output_path(args.work_dir, window, variant.name)
                runtime_path = runtime_config_path(args.work_dir, window, variant.name)
                write_runtime_config(runtime_path, config, window, output_path)
                run_config(runtime_path, args.verbose)
                summary = evaluate_paper_policies.summarize_output(
                    window,
                    Path(variant.name),
                    config,
                    output_path,
                )
                variant_runs.append(summary)
                all_runs.append(summary)

            variant_summaries.append(
                summarize_variant(
                    variant,
                    variant_runs,
                    args.drawdown_weight,
                    args.min_mean_fills,
                    args.fill_penalty,
                )
            )

        variant_summaries.sort(key=lambda summary: summary.score, reverse=True)
        write_variant_summaries(args.variants_output, variant_summaries)
        write_run_summaries(args.runs_output, all_runs)

        baseline_config_path = project_path(args.baseline_config) if args.baseline_config else None
        if baseline_config_path is not None:
            baseline_config = evaluate_paper_policies.load_config(baseline_config_path)
            baseline_runs = []
            for window in windows:
                output_path = runtime_output_path(args.work_dir, window, baseline_config_path.stem)
                runtime_path = runtime_config_path(args.work_dir, window, baseline_config_path.stem)
                write_runtime_config(runtime_path, baseline_config, window, output_path)
                run_config(runtime_path, args.verbose)
                baseline_runs.append(
                    evaluate_paper_policies.summarize_output(
                        window,
                        baseline_config_path,
                        baseline_config,
                        output_path,
                    )
                )
            baseline = evaluate_paper_policies.aggregate_runs(baseline_runs)[0]
            print()
            print(
                "Static baseline "
                f"fills={baseline.mean_fills:.2f} "
                f"pnl={baseline.mean_pnl:.4f} "
                f"pnl_std={baseline.std_pnl:.4f} "
                f"drawdown={baseline.mean_drawdown:.4f} "
                f"spread={baseline.mean_spread:.4f}"
            )

        print()
        print(render_table(variant_summaries, args.top))
        print()
        print(f"wrote {project_path(args.variants_output)}")
        print(f"wrote {project_path(args.runs_output)}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

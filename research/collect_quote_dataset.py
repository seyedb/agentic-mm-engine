#!/usr/bin/env python3
"""Collect a timestamped public quote dataset and optionally evaluate policies."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fetch_public_quotes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path("data/quotes")
DEFAULT_STATIC_CONFIG = Path("configs/runs/kraken_solusd_maker_fee_paper_session.json")
DEFAULT_ADAPTIVE_CONFIG = Path("configs/runs/kraken_solusd_adaptive_maker_fee_paper_session.json")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect public top-of-book quotes into a timestamped research dataset."
    )
    parser.add_argument(
        "--pair",
        default="SOLUSD",
        help="Kraken asset pair, for example SOLUSD, ETHUSD, or XBTUSD.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=360,
        help="Number of top-of-book snapshots to fetch.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait between snapshots.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for collected quote datasets.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Dataset id. Defaults to kraken_<pair>_<utc timestamp>.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run paper policy evaluation after collection.",
    )
    parser.add_argument(
        "--eval-config",
        action="append",
        type=Path,
        default=[],
        help="paper_session config to evaluate. May be passed more than once.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Rows per policy-evaluation window.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=60,
        help="Rows between policy-evaluation window starts.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum policy-evaluation windows. Use 0 for all windows.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_args(args: argparse.Namespace) -> None:
    fetch_public_quotes.validate_args(args)
    if args.run_id is not None and not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("--run-id may contain only letters, numbers, underscores, and hyphens")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.max_windows < 0:
        raise ValueError("--max-windows must be non-negative")
    if args.evaluate and args.samples < args.window_size:
        raise ValueError("--samples must be greater than or equal to --window-size when using --evaluate")


def default_run_id(pair: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    pair_name = re.sub(r"[^a-z0-9]+", "_", pair.lower()).strip("_")
    return f"kraken_{pair_name}_{timestamp}"


def dataset_paths(out_dir: Path, run_id: str) -> tuple[Path, Path]:
    csv_path = project_path(out_dir) / f"{run_id}.csv"
    metadata_path = project_path(out_dir) / f"{run_id}.meta.json"
    return csv_path, metadata_path


def evaluation_configs(args: argparse.Namespace) -> list[Path]:
    if args.eval_config:
        return [project_path(path) for path in args.eval_config]
    return [project_path(DEFAULT_STATIC_CONFIG), project_path(DEFAULT_ADAPTIVE_CONFIG)]


def evaluation_paths(run_id: str) -> tuple[Path, Path, Path]:
    base = PROJECT_ROOT / "target" / "research" / "quote_datasets" / run_id
    return (
        base / "paper_policy_windows",
        base / "paper_policy_window_runs.csv",
        base / "paper_policy_evaluation.csv",
    )


def run_policy_evaluation(
    *,
    run_id: str,
    data_path: Path,
    config_paths: list[Path],
    window_size: int,
    step_size: int,
    max_windows: int,
) -> dict[str, str]:
    work_dir, runs_output, aggregate_output = evaluation_paths(run_id)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "research" / "evaluate_paper_policies.py"),
        *[str(path) for path in config_paths],
        "--data",
        str(data_path),
        "--window-size",
        str(window_size),
        "--step-size",
        str(step_size),
        "--max-windows",
        str(max_windows),
        "--work-dir",
        str(work_dir),
        "--runs-output",
        str(runs_output),
        "--aggregate-output",
        str(aggregate_output),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return {
        "work_dir": str(work_dir),
        "runs_output": str(runs_output),
        "aggregate_output": str(aggregate_output),
    }


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    run_id: str,
    csv_path: Path,
    started_at: str,
    ended_at: str,
    evaluation: dict[str, str] | None,
) -> None:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_type": "quote_dataset",
        "run_id": run_id,
        "pair": args.pair,
        "samples": args.samples,
        "interval_seconds": args.interval_seconds,
        "timeout": args.timeout,
        "started_at": started_at,
        "ended_at": ended_at,
        "csv_path": str(csv_path),
        "evaluation": evaluation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        run_id = args.run_id or default_run_id(args.pair)
        csv_path, metadata_path = dataset_paths(args.out_dir, run_id)

        started_at = utc_now()
        print(f"Collecting {args.samples} {args.pair} quote snapshots", flush=True)
        snapshots = fetch_public_quotes.fetch_snapshots(
            args.pair,
            args.samples,
            args.interval_seconds,
            args.timeout,
        )
        fetch_public_quotes.write_quote_events(csv_path, snapshots)

        evaluation = None
        if args.evaluate:
            print("Evaluating paper policies", flush=True)
            evaluation = run_policy_evaluation(
                run_id=run_id,
                data_path=csv_path,
                config_paths=evaluation_configs(args),
                window_size=args.window_size,
                step_size=args.step_size,
                max_windows=args.max_windows,
            )

        ended_at = utc_now()
        write_metadata(
            metadata_path,
            args=args,
            run_id=run_id,
            csv_path=csv_path,
            started_at=started_at,
            ended_at=ended_at,
            evaluation=evaluation,
        )

        print(f"CSV: {csv_path}")
        print(f"Metadata: {metadata_path}")
        if evaluation is not None:
            print(f"Policy runs: {evaluation['runs_output']}")
            print(f"Policy summary: {evaluation['aggregate_output']}")
    except (OSError, ValueError, TimeoutError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

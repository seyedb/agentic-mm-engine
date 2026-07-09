#!/usr/bin/env python3
"""Re-run policy evaluations for existing quote datasets with current configs."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATTERN = "data/quotes/*.meta.json"
DEFAULT_STATIC_CONFIG = Path("configs/runs/kraken_solusd_maker_fee_paper_session.json")
DEFAULT_ADAPTIVE_CONFIG = Path("configs/runs/kraken_solusd_adaptive_maker_fee_paper_session.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run policy evaluation for collected quote datasets."
    )
    parser.add_argument(
        "metadata_paths",
        nargs="*",
        type=Path,
        help="Quote-dataset metadata files. Defaults to data/quotes/*.meta.json.",
    )
    parser.add_argument(
        "--eval-config",
        action="append",
        type=Path,
        default=[],
        help="paper_session config to evaluate. May be passed more than once.",
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
        "--tag",
        default="current",
        help="Evaluation tag written under target/research/quote_datasets/<run_id>/<tag>/.",
    )
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="Point each metadata file's evaluation field to the new outputs.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def discover_metadata(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(Path(path) for path in glob.glob(DEFAULT_METADATA_PATTERN))


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    return metadata


def evaluation_configs(args: argparse.Namespace) -> list[Path]:
    if args.eval_config:
        return [project_path(path) for path in args.eval_config]
    return [project_path(DEFAULT_STATIC_CONFIG), project_path(DEFAULT_ADAPTIVE_CONFIG)]


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.max_windows < 0:
        raise ValueError("--max-windows must be non-negative")
    if not args.tag.strip():
        raise ValueError("--tag must not be empty")


def output_paths(run_id: str, tag: str) -> tuple[Path, Path, Path]:
    base = PROJECT_ROOT / "target" / "research" / "quote_datasets" / run_id / tag
    return (
        base / "paper_policy_windows",
        base / "paper_policy_window_runs.csv",
        base / "paper_policy_evaluation.csv",
    )


def run_evaluation(
    *,
    run_id: str,
    data_path: Path,
    config_paths: list[Path],
    window_size: int,
    step_size: int,
    max_windows: int,
    tag: str,
) -> dict[str, str]:
    work_dir, runs_output, aggregate_output = output_paths(run_id, tag)
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


def update_metadata(path: Path, metadata: dict[str, Any], evaluation: dict[str, str]) -> None:
    metadata["evaluation"] = evaluation
    metadata["evaluation_updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        metadata_paths = discover_metadata(args.metadata_paths)
        if not metadata_paths:
            raise ValueError("no quote-dataset metadata files found")

        configs = evaluation_configs(args)
        for metadata_path in metadata_paths:
            metadata = load_metadata(metadata_path)
            run_id = str(metadata.get("run_id") or metadata_path.stem.removesuffix(".meta"))
            data_path = project_path(Path(str(metadata.get("csv_path", ""))))
            if not data_path.exists():
                raise ValueError(f"{metadata_path}: quote CSV not found: {data_path}")

            print(f"Re-evaluating {run_id}", flush=True)
            evaluation = run_evaluation(
                run_id=run_id,
                data_path=data_path,
                config_paths=configs,
                window_size=args.window_size,
                step_size=args.step_size,
                max_windows=args.max_windows,
                tag=args.tag,
            )
            if args.update_metadata:
                update_metadata(metadata_path, metadata, evaluation)

    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

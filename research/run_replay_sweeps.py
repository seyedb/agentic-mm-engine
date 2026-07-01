#!/usr/bin/env python3
"""Run replay sweeps for one or more replay CSV files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cargo replay-sweep for multiple replay event CSVs."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="Replay event CSV files.",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="Comma-separated random seeds passed to replay-sweep.",
    )
    parser.add_argument(
        "--spreads",
        default="0.2,0.5,1.0",
        help="Comma-separated spreads passed to replay-sweep.",
    )
    parser.add_argument(
        "--skews",
        default="0.0,0.02,0.05",
        help="Comma-separated skew values passed to replay-sweep.",
    )
    parser.add_argument(
        "--quantities",
        default="0.05,0.1,0.2",
        help="Comma-separated order quantities passed to replay-sweep.",
    )
    parser.add_argument(
        "--fee-rate",
        default="0.001",
        help="Fee rate passed to replay-sweep.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    missing = [path for path in args.csv_paths if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise ValueError(f"missing replay CSV files: {paths}")


def replay_sweep_command(path: Path, args: argparse.Namespace) -> list[str]:
    return [
        "cargo",
        "run",
        "--",
        "replay-sweep",
        str(path),
        "--seeds",
        args.seeds,
        "--spreads",
        args.spreads,
        "--skews",
        args.skews,
        "--quantities",
        args.quantities,
        "--fee-rate",
        args.fee_rate,
    ]


def run_sweeps(args: argparse.Namespace) -> None:
    for path in args.csv_paths:
        command = replay_sweep_command(path, args)
        print(f"running replay sweep for {path}", flush=True)
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        run_sweeps(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

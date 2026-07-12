#!/usr/bin/env python3
"""Propose a paper fill_model block from a calibration CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "rank",
    "base_intensity",
    "distance_decay",
    "volatility_boost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a touch-intensity fill_model proposal from calibration results."
    )
    parser.add_argument(
        "calibration_csv",
        type=Path,
        help="Path to *_paper_fill_calibration.csv.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Calibration rank to propose. Default: 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the proposal JSON.",
    )
    return parser.parse_args()


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


def read_ranked_row(path: Path, rank: int) -> dict[str, str]:
    if rank <= 0:
        raise ValueError("--rank must be positive")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("calibration CSV is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        for row in reader:
            if parse_int(row, "rank") == rank:
                return row

    raise ValueError(f"rank {rank} not found in {path}")


def proposal_from_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "fill_model": {
            "type": "touch_intensity",
            "base_intensity": parse_float(row, "base_intensity"),
            "distance_decay": parse_float(row, "distance_decay"),
            "volatility_boost": parse_float(row, "volatility_boost"),
        }
    }


def main() -> int:
    args = parse_args()

    try:
        row = read_ranked_row(args.calibration_csv, args.rank)
        proposal = proposal_from_row(row)
        output = json.dumps(proposal, indent=2, sort_keys=True) + "\n"

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output)
            print(f"wrote {args.output}", file=sys.stderr)

        print(output, end="")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

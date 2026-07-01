#!/usr/bin/env python3
"""Fetch several public replay windows for cross-window experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import fetch_public_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch multiple Kraken OHLC windows into replay CSV files."
    )
    parser.add_argument(
        "--pair",
        default="SOLUSD",
        help="Kraken asset pair, for example SOLUSD, ETHUSD, or XBTUSD.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Candle interval in minutes.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=120,
        help="Candles per replay window.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="First window start as Unix seconds or an ISO UTC timestamp ending in Z.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="Number of windows to fetch.",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=120,
        help="Minutes between window start times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory for generated replay CSV files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.interval not in fetch_public_events.KRAKEN_INTERVALS:
        supported = ", ".join(str(value) for value in sorted(fetch_public_events.KRAKEN_INTERVALS))
        raise ValueError(f"--interval must be one of: {supported}")
    if not 1 <= args.bars <= fetch_public_events.MAX_KRAKEN_BARS:
        raise ValueError(f"--bars must be between 1 and {fetch_public_events.MAX_KRAKEN_BARS}")
    if args.windows <= 0:
        raise ValueError("--windows must be positive")
    if args.step_minutes <= 0:
        raise ValueError("--step-minutes must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    fetch_public_events.parse_since(args.start)


def window_starts(start: str, windows: int, step_minutes: int) -> list[int]:
    first_start = fetch_public_events.parse_since(start)
    step_seconds = step_minutes * 60
    return [first_start + index * step_seconds for index in range(windows)]


def replay_window_path(out_dir: Path, pair: str, since_seconds: int) -> Path:
    timestamp = dt.datetime.fromtimestamp(since_seconds, tz=dt.timezone.utc)
    pair_name = re.sub(r"[^a-z0-9]+", "_", pair.lower()).strip("_")
    return out_dir / f"kraken_{pair_name}_{timestamp:%Y%m%d_%H%M}.csv"


def fetch_window(
    pair: str,
    interval: int,
    bars: int,
    since_seconds: int,
    timeout: float,
    out_path: Path,
) -> None:
    events = fetch_public_events.fetch_kraken_ohlc(
        pair,
        interval,
        bars,
        str(since_seconds),
        timeout,
    )
    fetch_public_events.write_events(out_path, events)
    print(f"wrote {len(events)} events to {out_path}")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        for since_seconds in window_starts(args.start, args.windows, args.step_minutes):
            fetch_window(
                args.pair,
                args.interval,
                args.bars,
                since_seconds,
                args.timeout,
                replay_window_path(args.out_dir, args.pair, since_seconds),
            )
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

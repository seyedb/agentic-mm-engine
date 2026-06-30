#!/usr/bin/env python3
"""Fetch recent public OHLC data and write replay events.

This intentionally writes the engine's simple replay format:

    timestamp_ms,mid_price

OHLC candle closes are used as a mid-price proxy. This is useful for replay
plumbing and research workflow checks, not exchange-grade order book replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
KRAKEN_INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}
MAX_KRAKEN_BARS = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch recent Kraken OHLC data into mm_engine replay CSV format."
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
        help="Candle interval in minutes. Kraken supports 1, 5, 15, 30, 60, 240, 1440, 10080, 21600.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=120,
        help="Number of recent candles to write, up to Kraken's 720-bar response limit.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/kraken_solusd_events.csv"),
        help="Output replay CSV path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.interval not in KRAKEN_INTERVALS:
        supported = ", ".join(str(value) for value in sorted(KRAKEN_INTERVALS))
        raise ValueError(f"--interval must be one of: {supported}")
    if not 1 <= args.bars <= MAX_KRAKEN_BARS:
        raise ValueError(f"--bars must be between 1 and {MAX_KRAKEN_BARS}")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")


def fetch_kraken_ohlc(
    pair: str, interval: int, bars: int, timeout: float
) -> list[tuple[int, float]]:
    since = int(time.time()) - bars * interval * 60
    query = urlencode({"pair": pair, "interval": interval, "since": since})
    url = f"{KRAKEN_OHLC_URL}?{query}"

    with urlopen(url, timeout=timeout) as response:
        payload = json.load(response)

    errors = payload.get("error", [])
    if errors:
        raise ValueError(f"Kraken API error: {', '.join(errors)}")

    return parse_kraken_ohlc(payload, bars)


def parse_kraken_ohlc(payload: dict, bars: int) -> list[tuple[int, float]]:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Kraken response missing result object")

    series = None
    for key, value in result.items():
        if key != "last":
            series = value
            break

    if not isinstance(series, list) or not series:
        raise ValueError("Kraken response contains no OHLC rows")

    events = []
    for row in series[-bars:]:
        if not isinstance(row, list) or len(row) < 5:
            raise ValueError(f"invalid OHLC row: {row!r}")

        timestamp_ms = int(row[0]) * 1000
        close_price = float(row[4])
        events.append((timestamp_ms, close_price))

    events.sort(key=lambda event: event[0])
    return events


def write_events(path: Path, events: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ms", "mid_price"])
        for timestamp_ms, mid_price in events:
            writer.writerow([timestamp_ms, f"{mid_price:.8f}"])


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        events = fetch_kraken_ohlc(args.pair, args.interval, args.bars, args.timeout)
        write_events(args.out, events)
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(events)} events to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch public top-of-book snapshots and write replay quote events."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


KRAKEN_DEPTH_URL = "https://api.kraken.com/0/public/Depth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Kraken top-of-book snapshots into replay CSV format."
    )
    parser.add_argument(
        "--pair",
        default="SOLUSD",
        help="Kraken asset pair, for example SOLUSD, ETHUSD, or XBTUSD.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=60,
        help="Number of top-of-book snapshots to fetch.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait between snapshots. Use 0 for immediate test samples.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/kraken_solusd_quotes.csv"),
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
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds must be non-negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")


def fetch_top_of_book(pair: str, timeout: float) -> tuple[int, float, float, float]:
    query = urlencode({"pair": pair, "count": 1})
    url = f"{KRAKEN_DEPTH_URL}?{query}"

    with urlopen(url, timeout=timeout) as response:
        payload = json.load(response)

    errors = payload.get("error", [])
    if errors:
        raise ValueError(f"Kraken API error: {', '.join(errors)}")

    bid, ask = parse_top_of_book(payload)
    timestamp_ms = int(time.time() * 1000)
    mid_price = (bid + ask) / 2.0
    return timestamp_ms, mid_price, bid, ask


def parse_top_of_book(payload: dict) -> tuple[float, float]:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Kraken response missing result object")

    book = None
    for value in result.values():
        if isinstance(value, dict):
            book = value
            break

    if book is None:
        raise ValueError("Kraken response contains no order book")

    bids = book.get("bids")
    asks = book.get("asks")
    if not isinstance(bids, list) or not bids:
        raise ValueError("Kraken response contains no bids")
    if not isinstance(asks, list) or not asks:
        raise ValueError("Kraken response contains no asks")

    return parse_price(bids[0], "bid"), parse_price(asks[0], "ask")


def parse_price(level: object, side: str) -> float:
    if not isinstance(level, list) or not level:
        raise ValueError(f"invalid {side} level: {level!r}")

    try:
        return float(level[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {side} price: {level!r}") from exc


def fetch_snapshots(
    pair: str, samples: int, interval_seconds: float, timeout: float
) -> list[tuple[int, float, float, float]]:
    snapshots = []
    for index in range(samples):
        snapshots.append(fetch_top_of_book(pair, timeout))
        if index + 1 < samples and interval_seconds > 0.0:
            time.sleep(interval_seconds)

    return snapshots


def write_quote_events(path: Path, snapshots: list[tuple[int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ms", "mid_price", "bid", "ask"])
        for timestamp_ms, mid_price, bid, ask in snapshots:
            writer.writerow(
                [
                    timestamp_ms,
                    f"{mid_price:.8f}",
                    f"{bid:.8f}",
                    f"{ask:.8f}",
                ]
            )


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        snapshots = fetch_snapshots(
            args.pair, args.samples, args.interval_seconds, args.timeout
        )
        write_quote_events(args.out, snapshots)
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(snapshots)} quote events to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

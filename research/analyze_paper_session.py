#!/usr/bin/env python3
"""Analyze paper-session decision logs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = {
    "timestamp_ms",
    "mid_price",
    "observed_bid",
    "observed_ask",
    "estimated_volatility",
    "regime",
    "agent_mode",
    "bid",
    "ask",
    "spread",
    "inventory",
    "cash",
    "pnl",
    "drawdown",
    "fills",
    "buy_fills",
    "sell_fills",
    "fill_quantity",
    "fill_notional",
    "fees",
}


@dataclass
class PaperRow:
    timestamp_ms: int
    mid_price: float
    observed_bid: float | None
    observed_ask: float | None
    estimated_volatility: float
    regime: str
    agent_mode: str
    bid: float
    ask: float
    spread: float
    inventory: float
    cash: float
    pnl: float
    drawdown: float
    fills: int
    buy_fills: int
    sell_fills: int
    fill_quantity: float
    fill_notional: float
    fees: float

    @property
    def has_observed_quote(self) -> bool:
        return self.observed_bid is not None and self.observed_ask is not None

    @property
    def bid_distance_to_observed_ask(self) -> float | None:
        if self.observed_ask is None:
            return None
        return self.observed_ask - self.bid

    @property
    def ask_distance_to_observed_bid(self) -> float | None:
        if self.observed_bid is None:
            return None
        return self.ask - self.observed_bid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a paper-session CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to *_paper_session.csv")
    return parser.parse_args()


def parse_float(value: str, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float in column '{column}': {value!r}") from exc


def parse_optional_float(value: str, column: str) -> float | None:
    if value == "":
        return None
    return parse_float(value, column)


def parse_int(value: str, column: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer in column '{column}': {value!r}") from exc


def parse_row(row: dict[str, str]) -> PaperRow:
    return PaperRow(
        timestamp_ms=parse_int(row["timestamp_ms"], "timestamp_ms"),
        mid_price=parse_float(row["mid_price"], "mid_price"),
        observed_bid=parse_optional_float(row["observed_bid"], "observed_bid"),
        observed_ask=parse_optional_float(row["observed_ask"], "observed_ask"),
        estimated_volatility=parse_float(
            row["estimated_volatility"], "estimated_volatility"
        ),
        regime=row["regime"],
        agent_mode=row["agent_mode"],
        bid=parse_float(row["bid"], "bid"),
        ask=parse_float(row["ask"], "ask"),
        spread=parse_float(row["spread"], "spread"),
        inventory=parse_float(row["inventory"], "inventory"),
        cash=parse_float(row["cash"], "cash"),
        pnl=parse_float(row["pnl"], "pnl"),
        drawdown=parse_float(row["drawdown"], "drawdown"),
        fills=parse_int(row["fills"], "fills"),
        buy_fills=parse_int(row["buy_fills"], "buy_fills"),
        sell_fills=parse_int(row["sell_fills"], "sell_fills"),
        fill_quantity=parse_float(row["fill_quantity"], "fill_quantity"),
        fill_notional=parse_float(row["fill_notional"], "fill_notional"),
        fees=parse_float(row["fees"], "fees"),
    )


def read_rows(path: Path) -> list[PaperRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        rows = [parse_row(row) for row in reader]

    if not rows:
        raise ValueError("CSV contains no paper-session rows")

    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percent(part: float, total: float) -> float:
    return 100.0 * part / total if total else 0.0


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def render_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines = [title]
    lines.append("  ".join(header.rjust(width) for header, width in zip(headers, widths)))
    for row in rows:
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def count_rows(rows: list[PaperRow], field: str) -> list[list[str]]:
    counts = Counter(getattr(row, field) for row in rows)
    return [
        [
            str(label),
            str(count),
            format_number(percent(count, len(rows)), 1),
        ]
        for label, count in counts.most_common()
    ]


def quote_distance_summary(rows: list[PaperRow]) -> tuple[int, float, float, float]:
    bid_distances = [
        distance
        for row in rows
        if (distance := row.bid_distance_to_observed_ask) is not None
    ]
    ask_distances = [
        distance
        for row in rows
        if (distance := row.ask_distance_to_observed_bid) is not None
    ]
    avg_bid_distance = mean(bid_distances)
    avg_ask_distance = mean(ask_distances)
    avg_quote_distance = mean([avg_bid_distance, avg_ask_distance])
    return (len(bid_distances), avg_bid_distance, avg_ask_distance, avg_quote_distance)


def print_summary(path: Path, rows: list[PaperRow]) -> None:
    final_row = rows[-1]
    total_fills = sum(row.fills for row in rows)
    total_fees = sum(row.fees for row in rows)
    total_notional = sum(row.fill_notional for row in rows)
    max_drawdown = max(row.drawdown for row in rows)
    min_inventory = min(row.inventory for row in rows)
    max_inventory = max(row.inventory for row in rows)
    avg_spread = mean([row.spread for row in rows])
    avg_volatility = mean([row.estimated_volatility for row in rows])
    passive_rows = sum(1 for row in rows if row.fills == 0)
    quote_steps, avg_bid_dist, avg_ask_dist, avg_quote_dist = quote_distance_summary(rows)

    print(f"Paper session: {path}")
    print(f"steps: {len(rows)}")
    print(f"final_pnl: {format_number(final_row.pnl, 4)}")
    print(f"final_inventory: {format_number(final_row.inventory, 4)}")
    print(f"inventory_range: {format_number(min_inventory, 4)} to {format_number(max_inventory, 4)}")
    print(f"fills: {total_fills}")
    print(f"passive_steps: {passive_rows} ({format_number(percent(passive_rows, len(rows)), 1)}%)")
    print(f"fees: {format_number(total_fees, 4)}")
    print(f"traded_notional: {format_number(total_notional, 4)}")
    print(f"max_drawdown: {format_number(max_drawdown, 4)}")
    print(f"avg_spread: {format_number(avg_spread, 4)}")
    print(f"avg_volatility: {format_number(avg_volatility, 4)}")
    print(f"observed_quote_steps: {quote_steps}")
    print(f"avg_bid_distance_to_observed_ask: {format_number(avg_bid_dist, 4)}")
    print(f"avg_ask_distance_to_observed_bid: {format_number(avg_ask_dist, 4)}")
    print(f"avg_quote_distance: {format_number(avg_quote_dist, 4)}")
    print()
    print(render_table("By agent mode", ["mode", "steps", "pct"], count_rows(rows, "agent_mode")))
    print()
    print(render_table("By regime", ["regime", "steps", "pct"], count_rows(rows, "regime")))


def main() -> int:
    args = parse_args()

    try:
        rows = read_rows(args.csv_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_summary(args.csv_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

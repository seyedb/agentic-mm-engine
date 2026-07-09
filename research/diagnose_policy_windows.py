#!/usr/bin/env python3
"""Diagnose static-vs-adaptive policy behavior at the window level."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_METADATA_PATTERN = "data/quotes/*.meta.json"
DEFAULT_OUTPUT = Path("target/research/policy_window_diagnostics.csv")
DEFAULT_SUMMARY = Path("target/research/policy_window_diagnostics.md")
DEFAULT_HTML = Path("target/research/policy_window_diagnostics.html")


@dataclass(frozen=True)
class RunRow:
    run_id: str
    pair: str
    started_at: str
    window: str
    config: str
    policy: str
    steps: int
    fills: int
    final_pnl: float
    total_fees: float
    max_drawdown: float
    min_inventory: float
    max_inventory: float
    avg_spread: float
    avg_quote_distance: float
    output: Path


@dataclass(frozen=True)
class MarketFeatures:
    avg_observed_spread: float
    avg_estimated_volatility: float
    mean_abs_mid_change: float
    mid_range: float


@dataclass(frozen=True)
class WindowPair:
    run_id: str
    pair: str
    started_at: str
    window: str
    steps: int
    static_fills: int
    adaptive_fills: int
    fills_delta: int
    static_pnl: float
    adaptive_pnl: float
    pnl_delta: float
    static_drawdown: float
    adaptive_drawdown: float
    drawdown_delta: float
    static_fees: float
    adaptive_fees: float
    fees_delta: float
    static_spread: float
    adaptive_spread: float
    spread_delta: float
    static_inventory_min: float
    static_inventory_max: float
    adaptive_inventory_min: float
    adaptive_inventory_max: float
    avg_observed_spread: float
    avg_estimated_volatility: float
    mean_abs_mid_change: float
    mid_range: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build window-level diagnostics for policy evaluations."
    )
    parser.add_argument(
        "metadata_paths",
        nargs="*",
        type=Path,
        help="Quote-dataset metadata files. Defaults to data/quotes/*.meta.json.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=DEFAULT_HTML,
        help="Plotly HTML diagnostics output path.",
    )
    return parser.parse_args()


def discover_metadata(paths: list[Path]) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(Path(path) for path in glob.glob(DEFAULT_METADATA_PATTERN))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "" if value is None else str(value)


def runs_output(metadata: dict[str, Any], metadata_path: Path) -> Path:
    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"{metadata_path}: metadata contains no evaluation object")
    value = evaluation.get("runs_output")
    if not value:
        raise ValueError(f"{metadata_path}: evaluation contains no runs_output")
    path = Path(str(value))
    if not path.exists():
        raise ValueError(f"{metadata_path}: runs_output not found: {path}")
    return path


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in column {column!r}: {row[column]!r}") from exc


def parse_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid integer in column {column!r}: {row[column]!r}") from exc


def read_run_rows(metadata_path: Path) -> list[RunRow]:
    metadata = load_json(metadata_path)
    path = runs_output(metadata, metadata_path)
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        for row in reader:
            rows.append(
                RunRow(
                    run_id=metadata_value(metadata, "run_id"),
                    pair=metadata_value(metadata, "pair"),
                    started_at=metadata_value(metadata, "started_at"),
                    window=row["window"],
                    config=row["config"],
                    policy=row["policy"],
                    steps=parse_int(row, "steps"),
                    fills=parse_int(row, "fills"),
                    final_pnl=parse_float(row, "final_pnl"),
                    total_fees=parse_float(row, "total_fees"),
                    max_drawdown=parse_float(row, "max_drawdown"),
                    min_inventory=parse_float(row, "min_inventory"),
                    max_inventory=parse_float(row, "max_inventory"),
                    avg_spread=parse_float(row, "avg_spread"),
                    avg_quote_distance=parse_float(row, "avg_quote_distance"),
                    output=Path(row["output"]),
                )
            )
    return rows


def read_paper_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV contains no rows")
    return rows


def market_features(path: Path) -> MarketFeatures:
    rows = read_paper_rows(path)
    observed_spreads = [
        parse_float(row, "observed_ask") - parse_float(row, "observed_bid")
        for row in rows
        if row["observed_ask"] != "" and row["observed_bid"] != ""
    ]
    volatilities = [parse_float(row, "estimated_volatility") for row in rows]
    mids = [parse_float(row, "mid_price") for row in rows]
    mid_changes = [
        abs(current - previous)
        for previous, current in zip(mids, mids[1:])
    ]
    return MarketFeatures(
        avg_observed_spread=mean(observed_spreads),
        avg_estimated_volatility=mean(volatilities),
        mean_abs_mid_change=mean(mid_changes),
        mid_range=max(mids) - min(mids),
    )


def pair_windows(run_rows: list[RunRow]) -> list[WindowPair]:
    by_key: dict[tuple[str, str], list[RunRow]] = {}
    for row in run_rows:
        by_key.setdefault((row.run_id, row.window), []).append(row)

    pairs = []
    for (_run_id, _window), group in sorted(by_key.items()):
        static = first_policy(group, "static")
        adaptive = first_policy(group, "adaptive")
        if static is None or adaptive is None:
            continue
        features = market_features(static.output)
        pairs.append(
            WindowPair(
                run_id=static.run_id,
                pair=static.pair,
                started_at=static.started_at,
                window=static.window,
                steps=min(static.steps, adaptive.steps),
                static_fills=static.fills,
                adaptive_fills=adaptive.fills,
                fills_delta=adaptive.fills - static.fills,
                static_pnl=static.final_pnl,
                adaptive_pnl=adaptive.final_pnl,
                pnl_delta=adaptive.final_pnl - static.final_pnl,
                static_drawdown=static.max_drawdown,
                adaptive_drawdown=adaptive.max_drawdown,
                drawdown_delta=adaptive.max_drawdown - static.max_drawdown,
                static_fees=static.total_fees,
                adaptive_fees=adaptive.total_fees,
                fees_delta=adaptive.total_fees - static.total_fees,
                static_spread=static.avg_spread,
                adaptive_spread=adaptive.avg_spread,
                spread_delta=adaptive.avg_spread - static.avg_spread,
                static_inventory_min=static.min_inventory,
                static_inventory_max=static.max_inventory,
                adaptive_inventory_min=adaptive.min_inventory,
                adaptive_inventory_max=adaptive.max_inventory,
                avg_observed_spread=features.avg_observed_spread,
                avg_estimated_volatility=features.avg_estimated_volatility,
                mean_abs_mid_change=features.mean_abs_mid_change,
                mid_range=features.mid_range,
            )
        )
    return pairs


def first_policy(rows: list[RunRow], policy: str) -> RunRow | None:
    for row in rows:
        if row.policy == policy:
            return row
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else 0.0


def format_float(value: float) -> str:
    return f"{value:.6f}"


def write_pairs(path: Path, pairs: list[WindowPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "pair",
                "started_at",
                "window",
                "steps",
                "static_fills",
                "adaptive_fills",
                "fills_delta",
                "static_pnl",
                "adaptive_pnl",
                "pnl_delta",
                "static_drawdown",
                "adaptive_drawdown",
                "drawdown_delta",
                "static_fees",
                "adaptive_fees",
                "fees_delta",
                "static_spread",
                "adaptive_spread",
                "spread_delta",
                "static_inventory_min",
                "static_inventory_max",
                "adaptive_inventory_min",
                "adaptive_inventory_max",
                "avg_observed_spread",
                "avg_estimated_volatility",
                "mean_abs_mid_change",
                "mid_range",
            ]
        )
        for pair in pairs:
            writer.writerow(pair_to_csv(pair))


def pair_to_csv(pair: WindowPair) -> list[str]:
    return [
        pair.run_id,
        pair.pair,
        pair.started_at,
        pair.window,
        str(pair.steps),
        str(pair.static_fills),
        str(pair.adaptive_fills),
        str(pair.fills_delta),
        format_float(pair.static_pnl),
        format_float(pair.adaptive_pnl),
        format_float(pair.pnl_delta),
        format_float(pair.static_drawdown),
        format_float(pair.adaptive_drawdown),
        format_float(pair.drawdown_delta),
        format_float(pair.static_fees),
        format_float(pair.adaptive_fees),
        format_float(pair.fees_delta),
        format_float(pair.static_spread),
        format_float(pair.adaptive_spread),
        format_float(pair.spread_delta),
        format_float(pair.static_inventory_min),
        format_float(pair.static_inventory_max),
        format_float(pair.adaptive_inventory_min),
        format_float(pair.adaptive_inventory_max),
        format_float(pair.avg_observed_spread),
        format_float(pair.avg_estimated_volatility),
        format_float(pair.mean_abs_mid_change),
        format_float(pair.mid_range),
    ]


def render_summary(pairs: list[WindowPair]) -> str:
    if not pairs:
        raise ValueError("no paired policy windows to summarize")

    pnl_wins = sum(1 for pair in pairs if pair.pnl_delta > 0.0)
    drawdown_wins = sum(1 for pair in pairs if pair.drawdown_delta < 0.0)
    fee_wins = sum(1 for pair in pairs if pair.fees_delta < 0.0)
    low_participation = sum(
        1
        for pair in pairs
        if pair.static_fills > 0 and pair.adaptive_fills / pair.static_fills < 0.5
    )
    pnl_deltas = [pair.pnl_delta for pair in pairs]
    drawdown_deltas = [pair.drawdown_delta for pair in pairs]
    fill_deltas = [float(pair.fills_delta) for pair in pairs]
    observed_spreads = [pair.avg_observed_spread for pair in pairs]
    volatilities = [pair.avg_estimated_volatility for pair in pairs]
    mid_ranges = [pair.mid_range for pair in pairs]

    lines = [
        "# Policy Window Diagnostics",
        "",
        "## Overview",
        "",
        f"- Paired windows: {len(pairs)}",
        f"- Datasets: {len({pair.run_id for pair in pairs})}",
        f"- Adaptive PnL wins: {pnl_wins}/{len(pairs)}",
        f"- Adaptive drawdown reductions: {drawdown_wins}/{len(pairs)}",
        f"- Adaptive fee reductions: {fee_wins}/{len(pairs)}",
        f"- Windows where adaptive fills < 50% of static fills: {low_participation}/{len(pairs)}",
        f"- Mean PnL delta: {format_float(mean(pnl_deltas))}",
        f"- Mean drawdown delta: {format_float(mean(drawdown_deltas))}",
        f"- Mean fill delta: {format_float(mean(fill_deltas))}",
        "",
        "## Market Feature Correlations With PnL Delta",
        "",
        f"- Observed spread correlation: {format_float(pearson(observed_spreads, pnl_deltas))}",
        f"- Estimated volatility correlation: {format_float(pearson(volatilities, pnl_deltas))}",
        f"- Mid-price range correlation: {format_float(pearson(mid_ranges, pnl_deltas))}",
        "",
        "## Paired Windows",
        "",
        "| run_id | window | pnl_delta | drawdown_delta | fills_delta | observed_spread | volatility | mid_range |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            "| "
            f"{pair.run_id} | {pair.window} | {format_float(pair.pnl_delta)} | "
            f"{format_float(pair.drawdown_delta)} | {pair.fills_delta} | "
            f"{format_float(pair.avg_observed_spread)} | "
            f"{format_float(pair.avg_estimated_volatility)} | "
            f"{format_float(pair.mid_range)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A positive `pnl_delta` means adaptive beat static in that window.",
            "- A negative `drawdown_delta` or `fees_delta` means adaptive reduced that risk or cost.",
            "- Low participation can make risk look better without proving market-making quality.",
            "- With only a few short datasets, correlations are diagnostic hints, not evidence of causality.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(pairs: list[WindowPair]) -> str:
    labels = [f"{pair.run_id}<br>{pair.window}" for pair in pairs]
    payload = {
        "labels": labels,
        "pnl_delta": [pair.pnl_delta for pair in pairs],
        "drawdown_delta": [pair.drawdown_delta for pair in pairs],
        "fills_delta": [pair.fills_delta for pair in pairs],
        "avg_observed_spread": [pair.avg_observed_spread for pair in pairs],
        "avg_estimated_volatility": [pair.avg_estimated_volatility for pair in pairs],
        "mid_range": [pair.mid_range for pair in pairs],
        "run_id": [pair.run_id for pair in pairs],
    }
    data_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Policy Window Diagnostics</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    #chart {{ width: 100%; height: 920px; }}
  </style>
</head>
<body>
  <h1>Policy Window Diagnostics</h1>
  <div id="chart"></div>
  <script>
    const d = {data_json};
    const text = d.labels.map((label, i) =>
      `${{label}}<br>` +
      `pnl_delta=${{d.pnl_delta[i].toFixed(6)}}<br>` +
      `drawdown_delta=${{d.drawdown_delta[i].toFixed(6)}}<br>` +
      `fills_delta=${{d.fills_delta[i]}}<br>` +
      `observed_spread=${{d.avg_observed_spread[i].toFixed(6)}}<br>` +
      `volatility=${{d.avg_estimated_volatility[i].toFixed(6)}}`
    );
    const traces = [
      {{
        type: "bar",
        name: "PnL delta",
        x: d.labels,
        y: d.pnl_delta,
        text,
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{ color: d.pnl_delta.map(v => v >= 0 ? "#15803d" : "#b91c1c") }},
        xaxis: "x",
        yaxis: "y"
      }},
      {{
        type: "bar",
        name: "Drawdown delta",
        x: d.labels,
        y: d.drawdown_delta,
        text,
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{ color: d.drawdown_delta.map(v => v <= 0 ? "#2563eb" : "#ea580c") }},
        xaxis: "x2",
        yaxis: "y2"
      }},
      {{
        type: "bar",
        name: "Fills delta",
        x: d.labels,
        y: d.fills_delta,
        text,
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{ color: "#7c3aed" }},
        xaxis: "x3",
        yaxis: "y3"
      }},
      {{
        type: "scatter",
        mode: "markers",
        name: "Observed spread vs PnL delta",
        x: d.avg_observed_spread,
        y: d.pnl_delta,
        text,
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{ color: "#0f766e", size: 11 }},
        xaxis: "x4",
        yaxis: "y4"
      }},
      {{
        type: "scatter",
        mode: "markers",
        name: "Volatility vs PnL delta",
        x: d.avg_estimated_volatility,
        y: d.pnl_delta,
        text,
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{ color: "#9333ea", size: 11 }},
        xaxis: "x5",
        yaxis: "y5"
      }}
    ];
    const layout = {{
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      hovermode: "closest",
      margin: {{ l: 70, r: 35, t: 35, b: 95 }},
      legend: {{ orientation: "h", x: 0, y: -0.08 }},
      grid: {{ rows: 5, columns: 1, pattern: "independent" }},
      xaxis: {{ title: "Window" }},
      yaxis: {{ title: "Adaptive - static PnL" }},
      xaxis2: {{ title: "Window" }},
      yaxis2: {{ title: "Adaptive - static drawdown" }},
      xaxis3: {{ title: "Window" }},
      yaxis3: {{ title: "Adaptive - static fills" }},
      xaxis4: {{ title: "Average observed spread" }},
      yaxis4: {{ title: "Adaptive - static PnL" }},
      xaxis5: {{ title: "Average estimated volatility" }},
      yaxis5: {{ title: "Adaptive - static PnL" }}
    }};
    Plotly.newPlot("chart", traces, layout, {{ responsive: true }});
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()

    try:
        metadata_paths = discover_metadata(args.metadata_paths)
        if not metadata_paths:
            raise ValueError("no quote-dataset metadata files found")
        run_rows = []
        for metadata_path in metadata_paths:
            run_rows.extend(read_run_rows(metadata_path))
        pairs = pair_windows(run_rows)
        if not pairs:
            raise ValueError("no static/adaptive paired windows found")

        write_pairs(args.output, pairs)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(render_summary(pairs))
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_html(pairs))
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

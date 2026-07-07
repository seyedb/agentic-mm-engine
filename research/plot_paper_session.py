#!/usr/bin/env python3
"""Render a paper-session CSV as an interactive Plotly HTML report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("target/research")

REQUIRED_COLUMNS = {
    "timestamp_ms",
    "mid_price",
    "observed_bid",
    "observed_ask",
    "agent_mode",
    "bid",
    "ask",
    "inventory",
    "pnl",
    "drawdown",
    "fills",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a paper-session CSV to Plotly HTML.")
    parser.add_argument("csv_path", type=Path, help="Path to *_paper_session.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="HTML output path. Default: target/research/<csv_stem>.html",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty")

        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no paper-session rows")
    return rows


def output_path(csv_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    return OUTPUT_DIR / f"{csv_path.stem}.html"


def as_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in column '{column}': {row[column]!r}") from exc


def as_optional_float(row: dict[str, str], column: str) -> float | None:
    if row[column] == "":
        return None
    return as_float(row, column)


def as_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid integer in column '{column}': {row[column]!r}") from exc


def utc_timestamp(row: dict[str, str]) -> str:
    timestamp_ms = as_int(row, "timestamp_ms")
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def line_trace(
    name: str,
    x: list[int],
    timestamps: list[str],
    values: list[float | None],
    color: str,
    *,
    axis: str = "",
    width: float = 2.0,
    dash: str = "solid",
) -> dict[str, Any]:
    return {
        "type": "scatter",
        "mode": "lines",
        "name": name,
        "x": x,
        "y": values,
        "customdata": timestamps,
        "connectgaps": False,
        "line": {"color": color, "width": width, "dash": dash},
        "xaxis": f"x{axis}" if axis else "x",
        "yaxis": f"y{axis}" if axis else "y",
        "hovertemplate": (
            "step=%{x}<br>"
            "time=%{customdata}<br>"
            f"{name}=%{{y:.6f}}"
            "<extra></extra>"
        ),
    }


def fill_markers(
    x: list[int],
    timestamps: list[str],
    mid_prices: list[float],
    fills: list[int],
) -> dict[str, Any] | None:
    points = [
        (step, timestamps[index], mid_prices[index], fill_count)
        for index, (step, fill_count) in enumerate(zip(x, fills))
        if fill_count > 0
    ]
    if not points:
        return None

    marker_x = [point[0] for point in points]
    marker_y = [point[2] for point in points]
    customdata = [[point[1], point[3]] for point in points]
    sizes = [7 + min(point[3], 5) * 2 for point in points]
    return {
        "type": "scatter",
        "mode": "markers",
        "name": "fills",
        "x": marker_x,
        "y": marker_y,
        "customdata": customdata,
        "marker": {"color": "#7c3aed", "size": sizes, "opacity": 0.85},
        "hovertemplate": (
            "step=%{x}<br>"
            "time=%{customdata[0]}<br>"
            "fills=%{customdata[1]}<br>"
            "mid=%{y:.6f}"
            "<extra></extra>"
        ),
    }


def risk_managed_shapes(x: list[int], modes: list[str]) -> list[dict[str, Any]]:
    shapes = []
    start: int | None = None
    for index, mode in enumerate(modes + [""]):
        if mode == "RiskManaged" and start is None:
            start = index
        elif mode != "RiskManaged" and start is not None:
            end = index - 1
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": x[start] - 0.5,
                    "x1": x[end] + 0.5,
                    "y0": 0.52,
                    "y1": 0.95,
                    "fillcolor": "rgba(245, 158, 11, 0.16)",
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
            start = None
    return shapes


def build_report(rows: list[dict[str, str]], title: str) -> str:
    x = list(range(len(rows)))
    timestamps = [utc_timestamp(row) for row in rows]
    mid_prices = [as_float(row, "mid_price") for row in rows]
    observed_bid = [as_optional_float(row, "observed_bid") for row in rows]
    observed_ask = [as_optional_float(row, "observed_ask") for row in rows]
    bids = [as_float(row, "bid") for row in rows]
    asks = [as_float(row, "ask") for row in rows]
    pnl = [as_float(row, "pnl") for row in rows]
    drawdown = [as_float(row, "drawdown") for row in rows]
    inventory = [as_float(row, "inventory") for row in rows]
    fills = [as_int(row, "fills") for row in rows]
    modes = [row["agent_mode"] for row in rows]

    traces: list[dict[str, Any]] = [
        line_trace("observed bid", x, timestamps, observed_bid, "#94a3b8", width=1.5),
        line_trace("observed ask", x, timestamps, observed_ask, "#64748b", width=1.5),
        line_trace("mid", x, timestamps, mid_prices, "#111827", width=2.2),
        line_trace("agent bid", x, timestamps, bids, "#2563eb", dash="dot"),
        line_trace("agent ask", x, timestamps, asks, "#dc2626", dash="dot"),
        line_trace("pnl", x, timestamps, pnl, "#15803d", axis="2", width=2.2),
        line_trace("drawdown", x, timestamps, drawdown, "#ea580c", axis="2", width=2.0),
        line_trace("inventory", x, timestamps, inventory, "#7c3aed", axis="3", width=2.2),
    ]

    markers = fill_markers(x, timestamps, mid_prices, fills)
    if markers is not None:
        traces.append(markers)

    total_fills = sum(fills)
    final_pnl = pnl[-1]
    final_inventory = inventory[-1]
    annotations = [
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.0,
            "y": 1.02,
            "showarrow": False,
            "xanchor": "left",
            "font": {"color": "#475569", "size": 12},
            "text": (
                f"steps={len(rows)} | fills={total_fills} | "
                f"final_pnl={final_pnl:.4f} | final_inventory={final_inventory:.4f}"
            ),
        }
    ]

    if total_fills == 0:
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.02,
                "showarrow": False,
                "xanchor": "right",
                "font": {"color": "#b91c1c", "size": 12},
                "text": "No fills: agent quotes never crossed the observed top of book.",
            }
        )

    layout: dict[str, Any] = {
        "title": {"text": title, "x": 0.02, "xanchor": "left"},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "hovermode": "x unified",
        "margin": {"l": 70, "r": 35, "t": 95, "b": 70},
        "legend": {"orientation": "h", "x": 0.0, "y": -0.14},
        "xaxis": {"domain": [0.07, 0.98], "anchor": "y", "showgrid": True},
        "yaxis": {"domain": [0.52, 0.95], "title": "Price", "showgrid": True},
        "xaxis2": {
            "domain": [0.07, 0.98],
            "anchor": "y2",
            "matches": "x",
            "showticklabels": False,
            "showgrid": True,
        },
        "yaxis2": {"domain": [0.29, 0.45], "title": "PnL / Drawdown", "showgrid": True},
        "xaxis3": {
            "domain": [0.07, 0.98],
            "anchor": "y3",
            "matches": "x",
            "title": "Step",
            "showgrid": True,
        },
        "yaxis3": {"domain": [0.08, 0.21], "title": "Inventory", "showgrid": True},
        "annotations": annotations,
        "shapes": risk_managed_shapes(x, modes),
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #0f172a;
      background: #ffffff;
    }}
    #paper-session-chart {{
      width: min(1280px, 100vw);
      height: min(900px, 100vh);
      margin: 0 auto;
    }}
  </style>
</head>
<body>
  <div id="paper-session-chart"></div>
  <script>
    const traces = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot("paper-session-chart", traces, layout, {{
      responsive: true,
      scrollZoom: true,
      displaylogo: false
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv_path)
    out = output_path(args.csv_path, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(rows, args.csv_path.name))
    print(out)


if __name__ == "__main__":
    main()

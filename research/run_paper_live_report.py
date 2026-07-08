#!/usr/bin/env python3
"""Run a live paper config, then analyze and plot its CSV output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "research"
REPORT_DIR = PROJECT_ROOT / "target" / "research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and report a live paper session.")
    parser.add_argument("config_path", type=Path, help="Path to a paper_live run config.")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Analyze and plot the existing CSV without running the live session.",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Plotly HTML output path. Default: target/research/<csv_stem>.html",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = json.load(handle)

    if config.get("type") != "paper_live":
        raise ValueError("config type must be 'paper_live'")
    if not config.get("pair"):
        raise ValueError("paper_live config must include pair")

    return config


def paper_live_csv_path(config: dict[str, Any]) -> Path:
    output = config.get("output")
    if output:
        return project_path(Path(output))

    pair = str(config["pair"]).lower()
    return PROJECT_ROOT / "target" / "reports" / f"kraken_{pair}_paper_live.csv"


def plot_html_path(csv_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        return project_path(requested)
    return REPORT_DIR / f"{csv_path.stem}.html"


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_metadata(
    path: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    csv_path: Path,
    html_path: Path,
    started_at: str | None,
    ended_at: str,
    skipped_run: bool,
) -> None:
    metadata = {
        "schema_version": 1,
        "run_type": "paper_live",
        "skipped_run": skipped_run,
        "started_at": started_at,
        "ended_at": ended_at,
        "config_path": str(config_path),
        "csv_path": str(csv_path),
        "plotly_html_path": str(html_path),
        "config": config,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config_path)

    try:
        config = load_config(config_path)
        csv_path = paper_live_csv_path(config)
        html_path = plot_html_path(csv_path, args.html_out)
        meta_path = metadata_path(csv_path)
        started_at = None

        if not args.skip_run:
            started_at = utc_now()
            print("Running live paper session", flush=True)
            run_command(["cargo", "run", "--", "run", str(config_path)])
            print()

        if not csv_path.exists():
            raise FileNotFoundError(f"paper live CSV not found: {csv_path}")

        print("Analyzing paper session", flush=True)
        run_command(
            [
                sys.executable,
                str(RESEARCH_DIR / "analyze_paper_session.py"),
                str(csv_path),
            ]
        )
        print()

        print("Rendering Plotly report", flush=True)
        run_command(
            [
                sys.executable,
                str(RESEARCH_DIR / "plot_paper_session.py"),
                str(csv_path),
                "--out",
                str(html_path),
            ]
        )
        print()

        ended_at = utc_now()
        write_metadata(
            meta_path,
            config_path=config_path,
            config=config,
            csv_path=csv_path,
            html_path=html_path,
            started_at=started_at,
            ended_at=ended_at,
            skipped_run=args.skip_run,
        )

        print(f"CSV: {csv_path}")
        print(f"Plotly HTML: {html_path}")
        print(f"Metadata: {meta_path}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

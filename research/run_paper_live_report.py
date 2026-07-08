#!/usr/bin/env python3
"""Run a live paper config, then analyze and plot its CSV output."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "research"
REPORT_DIR = PROJECT_ROOT / "target" / "research"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


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
    parser.add_argument(
        "--run-id",
        default=None,
        help="Write this run to target/reports/paper_live/<run_id>.csv.",
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


def run_id_csv_path(run_id: str) -> Path:
    return PROJECT_ROOT / "target" / "reports" / "paper_live" / f"{run_id}.csv"


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


def validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("--run-id may contain only letters, numbers, underscores, and hyphens")


def config_for_run(config: dict[str, Any], run_id: str | None) -> tuple[dict[str, Any], Path]:
    if run_id is None:
        return config, paper_live_csv_path(config)

    runtime_config = dict(config)
    csv_path = run_id_csv_path(run_id)
    runtime_config["output"] = str(csv_path.relative_to(PROJECT_ROOT))
    return runtime_config, csv_path


def write_runtime_config(config: dict[str, Any], directory: Path) -> Path:
    path = directory / "paper_live_run_config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


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
    run_id: str | None,
) -> None:
    metadata = {
        "schema_version": 1,
        "run_type": "paper_live",
        "run_id": run_id,
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
        validate_run_id(args.run_id)
        config = load_config(config_path)
        runtime_config, csv_path = config_for_run(config, args.run_id)
        html_path = plot_html_path(csv_path, args.html_out)
        meta_path = metadata_path(csv_path)
        started_at = None

        with tempfile.TemporaryDirectory(prefix="mm_engine_paper_live_") as temp_dir:
            runtime_config_path = (
                write_runtime_config(runtime_config, Path(temp_dir))
                if args.run_id is not None
                else config_path
            )

            if not args.skip_run:
                started_at = utc_now()
                print("Running live paper session", flush=True)
                run_command(["cargo", "run", "--", "run", str(runtime_config_path)])
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
            config=runtime_config,
            csv_path=csv_path,
            html_path=html_path,
            started_at=started_at,
            ended_at=ended_at,
            skipped_run=args.skip_run,
            run_id=args.run_id,
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

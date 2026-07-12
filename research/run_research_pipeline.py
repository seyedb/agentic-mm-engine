#!/usr/bin/env python3
"""Run the reproducible offline research pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline research pipeline.")
    parser.add_argument(
        "--skip-policy-gates",
        action="store_true",
        help="Skip the expensive policy gate passes and regenerate learned/report artifacts only.",
    )
    parser.add_argument(
        "--skip-bandit",
        action="store_true",
        help="Skip the offline LinUCB diagnostic.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def pipeline_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    if not args.skip_policy_gates:
        commands.append([sys.executable, str(RESEARCH_DIR / "policy_evaluation_gate.py")])
    commands.append([sys.executable, str(RESEARCH_DIR / "train_policy_selector.py")])
    if not args.skip_policy_gates:
        commands.append([sys.executable, str(RESEARCH_DIR / "policy_evaluation_gate.py")])
    commands.append([sys.executable, str(RESEARCH_DIR / "train_linear_policy_agent.py")])
    if not args.skip_policy_gates:
        commands.append([sys.executable, str(RESEARCH_DIR / "policy_evaluation_gate.py")])
    if not args.skip_bandit:
        commands.append([sys.executable, str(RESEARCH_DIR / "train_bandit_selector.py")])
    commands.append([sys.executable, str(RESEARCH_DIR / "summarize_live_dataset_evaluation.py")])
    commands.append([sys.executable, str(RESEARCH_DIR / "write_project_report.py")])
    return commands


def display_command(command: list[str]) -> str:
    display = command[:]
    if display and Path(display[0]).name.startswith("python"):
        display[0] = "python3"
    return " ".join(str(Path(part).relative_to(PROJECT_ROOT)) if Path(part).is_absolute() and Path(part).is_relative_to(PROJECT_ROOT) else part for part in display)


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"$ {display_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    try:
        for command in pipeline_commands(args):
            run_command(command, args.dry_run)
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

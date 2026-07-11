#!/usr/bin/env python3
"""Write a brief project research report from latest evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_SUMMARY = Path("target/research/policy_gate_policy_summary.csv")
DEFAULT_SELECTOR_SWEEP = Path("target/research/selector_policy_sweep.csv")
DEFAULT_OUTPUT = Path("target/research/project_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a concise project report.")
    parser.add_argument("--policy-summary", type=Path, default=DEFAULT_POLICY_SUMMARY)
    parser.add_argument("--selector-sweep", type=Path, default=DEFAULT_SELECTOR_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        return list(reader)


def parse_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"invalid float in {column}: {row[column]!r}") from exc


def best_configured_policy(rows: list[dict[str, str]]) -> dict[str, str]:
    configured = [row for row in rows if row["assumption"] == "configured"]
    if not configured:
        raise ValueError("policy summary has no configured rows")
    return max(configured, key=lambda row: parse_float(row, "mean_utility"))


def selector_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row["assumption"] == "configured" and row["policy"] == "selector":
            return row
    return None


def adaptive_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row["assumption"] == "configured" and row["policy"] == "adaptive":
            return row
    return None


def best_sweep_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("selector sweep has no rows")
    return min(rows, key=lambda row: int(row["rank"]))


def fmt(value: str) -> str:
    return f"{float(value):.6f}"


def render_report(policy_rows: list[dict[str, str]], sweep_rows: list[dict[str, str]]) -> str:
    best_policy = best_configured_policy(policy_rows)
    selector = selector_row(policy_rows)
    adaptive = adaptive_row(policy_rows)
    best_sweep = best_sweep_row(sweep_rows)

    lines = [
        "# Agentic Market Making: Brief Research Report",
        "",
        "## Objective",
        "",
        "Build a small Rust research engine for testing market-making policies on public top-of-book data.",
        "",
        "## Current Design",
        "",
        "- Rust runs replay, quoting, fills, fees, inventory, and PnL accounting.",
        "- Python runs research sweeps, policy gates, and report generation.",
        "- Policies include static, adaptive, hybrid, and a weighted selector.",
        "",
        "## Agentic Status",
        "",
        "The project currently has an agentic control layer: the selector observes market state and chooses static or adaptive quoting. It is not yet a learned ML agent. The natural next proof of concept is a small Python-trained selector that exports weights back into the Rust config.",
        "",
        "## Latest Result",
        "",
        f"- Best configured policy: `{best_policy['policy']}` with utility `{fmt(best_policy['mean_utility'])}`.",
    ]
    if selector and adaptive:
        delta = parse_float(selector, "mean_utility") - parse_float(adaptive, "mean_utility")
        lines.extend(
            [
                f"- Selector utility: `{fmt(selector['mean_utility'])}`.",
                f"- Adaptive baseline utility: `{fmt(adaptive['mean_utility'])}`.",
                f"- Selector minus adaptive: `{delta:.6f}`.",
                f"- Selector adaptive-step rate: `{fmt(selector['adaptive_step_pct'])}%`.",
            ]
        )
    lines.extend(
        [
            f"- Best sweep variant: `{best_sweep['variant']}`.",
            f"- Sweep score: `{fmt(best_sweep['score'])}`.",
            "",
            "## Interpretation",
            "",
            "The selector is promising because it can beat the adaptive baseline under the configured evaluation while using adaptive quoting less than 100% of the time. It is not robust enough to claim a trading edge, because results still depend on fill assumptions.",
            "",
            "## Limitations",
            "",
            "- Public top-of-book snapshots are limited data.",
            "- Fill behavior is modeled, not exchange-verified.",
            "- Selector weights are swept, not learned.",
            "",
            "## Next Step",
            "",
            "Add a minimal ML-agent proof of concept: train a simple Python model or contextual bandit to choose selector weights or policy actions, then export the learned parameters into a Rust paper-session config.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        policy_rows = read_csv(project_path(args.policy_summary))
        sweep_rows = read_csv(project_path(args.selector_sweep))
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_report(policy_rows, sweep_rows))
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {project_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

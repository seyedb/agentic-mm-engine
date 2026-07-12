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
DEFAULT_LEARNED_SELECTOR_FOLDS = Path("target/research/learned_policy_selector_folds.csv")
DEFAULT_CONTEXTUAL_BANDIT_RUNS = Path("target/research/contextual_bandit_agent_runs.csv")
DEFAULT_CONTEXTUAL_BANDIT_FOLDS = Path("target/research/contextual_bandit_agent_folds.csv")
DEFAULT_LIVE_RUNS = Path("target/research/paper_live_runs.csv")
DEFAULT_OUTPUT = Path("docs/final_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a concise project report.")
    parser.add_argument("--policy-summary", type=Path, default=DEFAULT_POLICY_SUMMARY)
    parser.add_argument("--selector-sweep", type=Path, default=DEFAULT_SELECTOR_SWEEP)
    parser.add_argument(
        "--learned-selector-folds",
        type=Path,
        default=DEFAULT_LEARNED_SELECTOR_FOLDS,
    )
    parser.add_argument("--bandit-runs", type=Path, default=DEFAULT_CONTEXTUAL_BANDIT_RUNS)
    parser.add_argument("--bandit-folds", type=Path, default=DEFAULT_CONTEXTUAL_BANDIT_FOLDS)
    parser.add_argument("--live-runs", type=Path, default=DEFAULT_LIVE_RUNS)
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


def learned_selector_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row["assumption"] == "configured" and row["policy"] == "learned_selector":
            return row
    return None


def linear_agent_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row["assumption"] == "configured" and row["policy"] == "linear_agent":
            return row
    return None


def bandit_agent_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row["assumption"] == "configured" and row["policy"] == "bandit_agent":
            return row
    return None


def policy_row(
    rows: list[dict[str, str]], assumption: str, policy: str
) -> dict[str, str] | None:
    for row in rows:
        if row["assumption"] == assumption and row["policy"] == policy:
            return row
    return None


def best_assumption_policy(rows: list[dict[str, str]], assumption: str) -> dict[str, str]:
    candidates = [row for row in rows if row["assumption"] == assumption]
    if not candidates:
        raise ValueError(f"policy summary has no rows for assumption {assumption!r}")
    return max(candidates, key=lambda row: parse_float(row, "mean_utility"))


def best_sweep_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("selector sweep has no rows")
    return min(rows, key=lambda row: int(row["rank"]))


def fmt(value: str) -> str:
    return f"{float(value):.6f}"


def mean_column(rows: list[dict[str, str]], column: str) -> float:
    if not rows:
        return 0.0
    return sum(parse_float(row, column) for row in rows) / len(rows)


def latest_live_run(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(row.get("steps", "0") or "0"),
            row.get("ended_at", ""),
            row.get("run_id", ""),
        ),
    )


def optional_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def render_report(
    policy_rows: list[dict[str, str]],
    sweep_rows: list[dict[str, str]],
    learned_rows: list[dict[str, str]],
    bandit_runs: list[dict[str, str]],
    bandit_folds: list[dict[str, str]],
    live_rows: list[dict[str, str]],
) -> str:
    best_policy = best_configured_policy(policy_rows)
    selector = selector_row(policy_rows)
    adaptive = adaptive_row(policy_rows)
    learned_selector = learned_selector_row(policy_rows)
    linear_agent = linear_agent_row(policy_rows)
    bandit_agent = bandit_agent_row(policy_rows)
    conservative_best = best_assumption_policy(policy_rows, "conservative_fill")
    liquid_best = best_assumption_policy(policy_rows, "liquid_fill")
    conservative_learned = policy_row(policy_rows, "conservative_fill", "learned_selector")
    liquid_learned = policy_row(policy_rows, "liquid_fill", "learned_selector")
    best_sweep = best_sweep_row(sweep_rows)
    live_run = latest_live_run(live_rows)

    lines = [
        "# Agentic Market Making: Final Research Report",
        "",
        "## Scope",
        "",
        "This is an experimental market-making research project, not a production trading system. The goal is to show a clean, reproducible loop from public quote data to Rust paper execution and small learned policy controllers.",
        "",
        "## System",
        "",
        "- Rust runs replay, paper sessions, live public-data paper mode, fills, fees, inventory, and PnL accounting.",
        "- Python runs data collection, policy evaluation, learned-gate training, Plotly reporting, and summary reports.",
        "- Policies include static, adaptive, hybrid, selector, learned-selector, linear-agent, and bandit-agent variants.",
        "- The learned selector is a logistic-regression classifier trained in Python, exported as JSON, and loaded back into Rust for paper execution.",
        "- The linear agent is a multi-action ridge-regression utility model trained in Python and executed by Rust.",
        "- The bandit agent is a LinUCB contextual-bandit policy trained in Python and executed by Rust.",
        "",
        "## Research Result",
        "",
        f"- Datasets: `{best_policy['datasets']}`.",
        f"- Windows: `{best_policy['windows']}`.",
        "- Fill assumptions: `configured`, `conservative_fill`, and `liquid_fill`.",
        f"- Best configured policy: `{best_policy['policy']}` with utility `{fmt(best_policy['mean_utility'])}`.",
        f"- Best conservative-fill policy: `{conservative_best['policy']}` with utility `{fmt(conservative_best['mean_utility'])}`.",
        f"- Best liquid-fill policy: `{liquid_best['policy']}` with utility `{fmt(liquid_best['mean_utility'])}`.",
        "",
        "## Configured Evaluation",
        "",
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
    if learned_selector and adaptive and selector:
        rust_learned = parse_float(learned_selector, "mean_utility")
        rust_adaptive = parse_float(adaptive, "mean_utility")
        rust_selector = parse_float(selector, "mean_utility")
        lines.extend(
            [
                f"- Rust learned-selector utility: `{rust_learned:.6f}`.",
                f"- Rust learned minus adaptive: `{rust_learned - rust_adaptive:.6f}`.",
                f"- Rust learned minus selector: `{rust_learned - rust_selector:.6f}`.",
                f"- Rust learned-selector adaptive-step rate: `{fmt(learned_selector['adaptive_step_pct'])}%`.",
            ]
        )
    if linear_agent and adaptive and selector:
        rust_linear = parse_float(linear_agent, "mean_utility")
        rust_adaptive = parse_float(adaptive, "mean_utility")
        rust_selector = parse_float(selector, "mean_utility")
        lines.extend(
            [
                f"- Rust linear-agent utility: `{rust_linear:.6f}`.",
                f"- Rust linear-agent minus adaptive: `{rust_linear - rust_adaptive:.6f}`.",
                f"- Rust linear-agent minus selector: `{rust_linear - rust_selector:.6f}`.",
                f"- Rust linear-agent adaptive-step rate: `{fmt(linear_agent['adaptive_step_pct'])}%`.",
            ]
        )
    if bandit_agent and adaptive and selector:
        rust_bandit = parse_float(bandit_agent, "mean_utility")
        rust_adaptive = parse_float(adaptive, "mean_utility")
        rust_selector = parse_float(selector, "mean_utility")
        lines.extend(
            [
                f"- Rust bandit-agent utility: `{rust_bandit:.6f}`.",
                f"- Rust bandit-agent minus adaptive: `{rust_bandit - rust_adaptive:.6f}`.",
                f"- Rust bandit-agent minus selector: `{rust_bandit - rust_selector:.6f}`.",
                f"- Rust bandit-agent adaptive-step rate: `{fmt(bandit_agent['adaptive_step_pct'])}%`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Fill-Assumption Check",
            "",
            f"- Conservative-fill winner: `{conservative_best['policy']}`.",
        ]
    )
    if conservative_learned:
        lines.append(
            f"- Learned-selector conservative-fill utility: `{fmt(conservative_learned['mean_utility'])}`."
        )
    lines.append(f"- Liquid-fill winner: `{liquid_best['policy']}`.")
    if liquid_learned:
        lines.append(
            f"- Learned-selector liquid-fill utility: `{fmt(liquid_learned['mean_utility'])}`."
        )
    lines.extend(
        [
            "- No policy wins all assumptions, so the result should be read as a research signal rather than a robust trading claim.",
            "",
            "## Learned Gate",
            "",
        ]
    )
    if learned_rows:
        learned = mean_column(learned_rows, "learned_utility")
        learned_minus_adaptive = mean_column(learned_rows, "learned_minus_adaptive")
        learned_minus_selector = mean_column(learned_rows, "learned_minus_selector")
        lines.extend(
            [
                f"- Learned gate holdout utility: `{learned:.6f}`.",
                f"- Learned minus adaptive: `{learned_minus_adaptive:.6f}`.",
                f"- Learned minus selector: `{learned_minus_selector:.6f}`.",
            ]
        )
    else:
        lines.append("- Learned holdout fold output was not found.")
    lines.extend(
        [
            f"- Best rule-selector sweep variant: `{best_sweep['variant']}`.",
            f"- Best sweep score: `{fmt(best_sweep['score'])}`.",
            "",
            "## Contextual Bandit Check",
            "",
        ]
    )
    if bandit_runs:
        chronological_bandit = mean_column(bandit_runs, "reward")
        lines.append(f"- Chronological executable LinUCB utility: `{chronological_bandit:.6f}`.")
    if bandit_folds:
        fold_bandit = mean_column(bandit_folds, "bandit_utility")
        lines.append(f"- Leave-one-dataset-out LinUCB utility: `{fold_bandit:.6f}`.")
        for action in ["static", "adaptive", "selector"]:
            column = f"{action}_utility"
            if column in bandit_folds[0]:
                utility = mean_column(bandit_folds, column)
                lines.append(f"- Fold always-`{action}` utility: `{utility:.6f}`.")
    if bandit_agent and adaptive and selector:
        lines.append(
            f"- Rust policy-gate bandit utility: `{fmt(bandit_agent['mean_utility'])}`."
        )
    lines.append(
        "- The bandit is executable and useful as an ML-agent proof of concept, but it is not the best policy in the current gate."
    )
    if live_run:
        lines.extend(
            [
                "",
                "## Live Paper Demo",
                "",
                f"- Run ID: `{live_run['run_id']}`.",
                f"- Pair: `{live_run['pair']}`.",
                f"- Public quote samples: `{live_run['steps']}`.",
                f"- Paper fills: `{live_run['fills']}`.",
                f"- Final paper PnL: `{fmt(live_run['final_pnl'])}`.",
                f"- Inventory range: `{fmt(live_run['min_inventory'])}` to `{fmt(live_run['max_inventory'])}`.",
                f"- Max drawdown: `{fmt(live_run['max_drawdown'])}`.",
                f"- Average quote distance: `{fmt(live_run['avg_quote_distance'])}`.",
                f"- Adaptive steps: `{live_run['adaptive_steps']}`.",
                f"- Dominant trigger: `{live_run['main_trigger']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The project has reached a credible proof-of-concept state. The agentic loop is real: public data feeds Rust paper sessions, Python trains small logistic, linear, and contextual-bandit policy models, the models are exported to JSON, and Rust executes those learned decisions in replay.",
            "",
            "The result is useful because it is measurable and falsifiable, not because it proves a trading edge. The learned selector leads under the configured evaluation, while the linear and bandit agents are functional multi-action controllers with mixed results. The weakest point remains fill realism.",
            "",
            "## Limitations",
            "",
            "- Public top-of-book snapshots are limited data.",
            "- Fill behavior is modeled, not exchange-verified.",
            "- The learned models are small and trained on a limited number of quote windows.",
            "- Live paper mode polls public quotes and never places orders.",
            "",
            "## Wrap-Up Assessment",
            "",
            "This is a reasonable place to wrap the current phase. The next genuinely interesting phase would be more live public-data evaluation and a larger dataset before trying to improve the ML agents.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        policy_rows = read_csv(project_path(args.policy_summary))
        sweep_rows = read_csv(project_path(args.selector_sweep))
        learned_path = project_path(args.learned_selector_folds)
        bandit_runs_path = project_path(args.bandit_runs)
        bandit_folds_path = project_path(args.bandit_folds)
        live_path = project_path(args.live_runs)
        learned_rows = optional_csv(learned_path)
        bandit_runs = optional_csv(bandit_runs_path)
        bandit_folds = optional_csv(bandit_folds_path)
        live_rows = optional_csv(live_path)
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_report(
                policy_rows,
                sweep_rows,
                learned_rows,
                bandit_runs,
                bandit_folds,
                live_rows,
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {project_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

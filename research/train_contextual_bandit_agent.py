#!/usr/bin/env python3
"""Train an executable LinUCB contextual-bandit paper policy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from train_bandit_selector import (
    LinUcbArm,
    baseline_utility,
    chronological_examples,
    evaluate_greedy,
    invert,
    mat_vec,
    pretrain_arms,
    run_linucb,
    write_decisions,
)
from train_policy_selector import (
    FEATURES,
    feature_scale,
    normalize_examples,
    project_path,
    read_examples,
    safe_mean,
)


DEFAULT_RUNS_OUTPUT = Path("target/research/contextual_bandit_agent_runs.csv")
DEFAULT_FOLDS_OUTPUT = Path("target/research/contextual_bandit_agent_folds.csv")
DEFAULT_MODEL_OUTPUT = Path("target/research/contextual_bandit_agent_model.json")
DEFAULT_REPORT_OUTPUT = Path("target/research/contextual_bandit_agent.md")


@dataclass(frozen=True)
class FoldResult:
    holdout_run_id: str
    windows: int
    bandit_utility: float
    baseline_utilities: dict[str, float]
    oracle_utility: float
    regret: float
    action_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an executable LinUCB paper-policy agent.")
    parser.add_argument(
        "--window-results",
        type=Path,
        default=Path("target/research/policy_gate_window_results.csv"),
    )
    parser.add_argument(
        "--policy-gate-dir",
        type=Path,
        default=Path("target/research/policy_gate/configured"),
    )
    parser.add_argument("--assumption", default="configured")
    parser.add_argument("--actions", default="static,adaptive,selector")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--feature-policy", default="static")
    parser.add_argument("--feature-window", type=int, default=30)
    parser.add_argument("--runs-output", type=Path, default=DEFAULT_RUNS_OUTPUT)
    parser.add_argument("--folds-output", type=Path, default=DEFAULT_FOLDS_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def parse_actions(value: str) -> list[str]:
    actions = [action.strip() for action in value.split(",") if action.strip()]
    if len(actions) < 2:
        raise ValueError("--actions must contain at least two actions")
    return actions


def write_model(
    path: Path,
    *,
    arms: dict[str, LinUcbArm],
    actions: list[str],
    scale: dict[str, float],
    alpha: float,
    ridge: float,
    feature_window: int,
    training_examples: int,
    training_datasets: list[str],
) -> None:
    theta = {}
    inverse_covariance = {}
    for action in actions:
        inverse = invert(arms[action].matrix)
        theta[action] = weights_by_feature(mat_vec(inverse, arms[action].reward_vector))
        inverse_covariance[action] = inverse

    payload = {
        "model_type": "linucb_contextual_bandit_agent",
        "actions": actions,
        "features": list(FEATURES),
        "theta": theta,
        "inverse_covariance": inverse_covariance,
        "feature_scale": scale,
        "feature_window": feature_window,
        "alpha": alpha,
        "ridge": ridge,
        "training_examples": training_examples,
        "training_datasets": training_datasets,
        "training_method": "offline_linucb_full_information_replay",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def weights_by_feature(values: list[float]) -> dict[str, float]:
    return {feature: values[index] for index, feature in enumerate(FEATURES)}


def leave_one_dataset_out(
    examples,
    actions: list[str],
    alpha: float,
    ridge: float,
) -> list[FoldResult]:
    folds = []
    for holdout in sorted({example.run_id for example in examples}):
        train = [example for example in examples if example.run_id != holdout]
        test = [example for example in examples if example.run_id == holdout]
        arms = pretrain_arms(train, actions, alpha, ridge)
        decisions = evaluate_greedy(test, actions, arms)
        folds.append(
            FoldResult(
                holdout_run_id=holdout,
                windows=len(test),
                bandit_utility=safe_mean([decision.reward for decision in decisions]),
                baseline_utilities={action: baseline_utility(test, action) for action in actions},
                oracle_utility=safe_mean([decision.best_reward for decision in decisions]),
                regret=sum(decision.regret for decision in decisions),
                action_counts=action_counts(decisions, actions),
            )
        )
    return folds


def write_folds(path: Path, folds, actions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "holdout_run_id",
                "windows",
                "bandit_utility",
                *[f"{action}_utility" for action in actions],
                "oracle_utility",
                "regret",
                *[f"{action}_actions" for action in actions],
            ]
        )
        for fold in folds:
            writer.writerow(
                [
                    fold.holdout_run_id,
                    fold.windows,
                    f"{fold.bandit_utility:.12f}",
                    *[f"{fold.baseline_utilities[action]:.12f}" for action in actions],
                    f"{fold.oracle_utility:.12f}",
                    f"{fold.regret:.12f}",
                    *[fold.action_counts[action] for action in actions],
                ]
            )


def action_counts(decisions, actions: list[str]) -> dict[str, int]:
    return {action: sum(decision.action == action for decision in decisions) for action in actions}


def write_report(
    path: Path,
    *,
    examples,
    decisions,
    folds,
    actions: list[str],
    alpha: float,
    ridge: float,
) -> None:
    bandit = safe_mean([decision.reward for decision in decisions])
    fold_bandit = safe_mean([fold.bandit_utility for fold in folds])
    rows = [
        "# Contextual Bandit Agent",
        "",
        "## Method",
        "",
        "This is an executable LinUCB contextual-bandit agent. Python replays quote windows, updates an arm-specific linear reward model, exports posterior parameters, and Rust loads the model to score policy actions at runtime.",
        "",
        "## Setup",
        "",
        f"- Examples: `{len(examples)}`.",
        f"- Actions: `{', '.join(actions)}`.",
        f"- Alpha: `{alpha:.3f}`.",
        f"- Ridge: `{ridge:.3f}`.",
        f"- Features: `{', '.join(FEATURES)}`.",
        "",
        "## Chronological Replay",
        "",
        f"- Bandit utility: `{bandit:.6f}`.",
        *[
            f"- Always `{action}` utility: `{baseline_utility(examples, action):.6f}`."
            for action in actions
        ],
        "",
        "## Leave-One-Dataset-Out",
        "",
        f"- Fold bandit utility: `{fold_bandit:.6f}`.",
        *[
            f"- Fold `{action}` utility: `{safe_mean([fold.baseline_utilities[action] for fold in folds]):.6f}`."
            for action in actions
        ],
        "",
        "## Action Mix",
        "",
    ]
    for action, count in action_counts(decisions, actions).items():
        rows.append(f"- `{action}`: `{count}` windows.")
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the project's executable contextual-bandit proof of concept. Treat it as a research controller: useful for demonstrating the agent loop, but only credible if it survives out-of-sample comparison against simpler policies.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def main() -> int:
    args = parse_args()
    actions = parse_actions(args.actions)
    read_args = argparse.Namespace(
        window_results=args.window_results,
        policy_gate_dir=args.policy_gate_dir,
        assumption=args.assumption,
        action_on=actions[-1],
        action_off=actions[0],
        feature_policy=args.feature_policy,
    )
    examples = chronological_examples(read_examples(read_args))
    scale = feature_scale(examples)
    normalized = normalize_examples(examples, scale)
    decisions = run_linucb(normalized, actions, args.alpha, args.ridge)
    folds = leave_one_dataset_out(normalized, actions, args.alpha, args.ridge)
    arms = pretrain_arms(normalized, actions, args.alpha, args.ridge)

    write_decisions(project_path(args.runs_output), decisions)
    write_folds(project_path(args.folds_output), folds, actions)
    write_model(
        project_path(args.model_output),
        arms=arms,
        actions=actions,
        scale=scale,
        alpha=args.alpha,
        ridge=args.ridge,
        feature_window=args.feature_window,
        training_examples=len(examples),
        training_datasets=sorted({example.run_id for example in examples}),
    )
    write_report(
        project_path(args.report_output),
        examples=normalized,
        decisions=decisions,
        folds=folds,
        actions=actions,
        alpha=args.alpha,
        ridge=args.ridge,
    )

    print("Contextual bandit agent")
    print(f"examples: {len(examples)}")
    print(f"chronological utility: {safe_mean([decision.reward for decision in decisions]):.6f}")
    print(f"fold utility: {safe_mean([fold.bandit_utility for fold in folds]):.6f}")
    print(f"wrote {project_path(args.model_output)}")
    print(f"wrote {project_path(args.folds_output)}")
    print(f"wrote {project_path(args.report_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

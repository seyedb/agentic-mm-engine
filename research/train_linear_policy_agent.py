#!/usr/bin/env python3
"""Train a simple multi-action linear utility policy agent."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from train_policy_selector import (
    FEATURES,
    Example,
    feature_scale,
    normalize_examples,
    project_path,
    read_examples,
    safe_mean,
)


DEFAULT_MODEL_OUTPUT = Path("target/research/linear_policy_agent_model.json")
DEFAULT_FOLDS_OUTPUT = Path("target/research/linear_policy_agent_folds.csv")
DEFAULT_REPORT_OUTPUT = Path("target/research/linear_policy_agent.md")


@dataclass(frozen=True)
class LinearModel:
    actions: list[str]
    weights: dict[str, dict[str, float]]
    intercepts: dict[str, float]
    feature_scale: dict[str, float]
    feature_window: int
    ridge: float


@dataclass(frozen=True)
class FoldResult:
    holdout_run_id: str
    linear_utility: float
    static_utility: float
    adaptive_utility: float
    selector_utility: float
    oracle_utility: float
    action_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-action linear policy agent.")
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
    parser.add_argument("--feature-policy", default="static")
    parser.add_argument("--feature-window", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--folds-output", type=Path, default=DEFAULT_FOLDS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def parse_actions(value: str) -> list[str]:
    actions = [action.strip() for action in value.split(",") if action.strip()]
    if len(actions) < 2:
        raise ValueError("--actions must contain at least two actions")
    return actions


def train_model(
    examples: list[Example],
    actions: list[str],
    scale: dict[str, float],
    feature_window: int,
    ridge: float,
) -> LinearModel:
    weights = {}
    intercepts = {}
    for action in actions:
        coefficients = ridge_regression(
            [feature_row(example) for example in examples],
            [example.utilities[action] for example in examples],
            ridge,
        )
        intercepts[action] = coefficients[0]
        weights[action] = {
            feature: coefficients[index + 1]
            for index, feature in enumerate(FEATURES)
        }
    return LinearModel(
        actions=actions,
        weights=weights,
        intercepts=intercepts,
        feature_scale=scale,
        feature_window=feature_window,
        ridge=ridge,
    )


def feature_row(example: Example) -> list[float]:
    return [example.features[feature] for feature in FEATURES]


def ridge_regression(rows: list[list[float]], targets: list[float], ridge: float) -> list[float]:
    size = len(FEATURES) + 1
    matrix = [[0.0 for _col in range(size)] for _row in range(size)]
    vector = [0.0 for _row in range(size)]
    for features, target in zip(rows, targets):
        row = [1.0, *features]
        for left in range(size):
            vector[left] += row[left] * target
            for right in range(size):
                matrix[left][right] += row[left] * row[right]
    for index in range(1, size):
        matrix[index][index] += ridge
    return solve_linear_system(matrix, vector)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular linear-agent regression matrix")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [
                value - factor * augmented[col][index]
                for index, value in enumerate(augmented[row])
            ]
    return [row[-1] for row in augmented]


def choose_action(model: LinearModel, example: Example) -> str:
    return max(model.actions, key=lambda action: (score_action(model, action, example), action))


def score_action(model: LinearModel, action: str, example: Example) -> float:
    return model.intercepts[action] + sum(
        model.weights[action][feature] * example.features[feature]
        for feature in FEATURES
    )


def evaluate_model(model: LinearModel, examples: list[Example]) -> tuple[float, dict[str, int], float]:
    utilities = []
    oracle_utilities = []
    action_counts = {action: 0 for action in model.actions}
    for example in examples:
        action = choose_action(model, example)
        action_counts[action] += 1
        utilities.append(example.utilities[action])
        oracle_utilities.append(max(example.utilities[action] for action in model.actions))
    return safe_mean(utilities), action_counts, safe_mean(oracle_utilities)


def leave_one_dataset_out(
    examples: list[Example],
    actions: list[str],
    scale: dict[str, float],
    feature_window: int,
    ridge: float,
) -> list[FoldResult]:
    folds = []
    for holdout in sorted({example.run_id for example in examples}):
        train = [example for example in examples if example.run_id != holdout]
        test = [example for example in examples if example.run_id == holdout]
        model = train_model(train, actions, scale, feature_window, ridge)
        utility, action_counts, oracle = evaluate_model(model, test)
        folds.append(
            FoldResult(
                holdout_run_id=holdout,
                linear_utility=utility,
                static_utility=baseline_utility(test, "static"),
                adaptive_utility=baseline_utility(test, "adaptive"),
                selector_utility=baseline_utility(test, "selector"),
                oracle_utility=oracle,
                action_counts=action_counts,
            )
        )
    return folds


def baseline_utility(examples: list[Example], policy: str) -> float:
    return safe_mean([example.utilities[policy] for example in examples])


def write_model(path: Path, model: LinearModel, examples: list[Example]) -> None:
    payload = {
        "model_type": "linear_utility_policy_agent",
        "actions": model.actions,
        "features": list(FEATURES),
        "weights": model.weights,
        "intercepts": model.intercepts,
        "feature_scale": model.feature_scale,
        "feature_window": model.feature_window,
        "training_method": "ridge_linear_utility_regression",
        "ridge": model.ridge,
        "training_examples": len(examples),
        "training_datasets": sorted({example.run_id for example in examples}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_folds(path: Path, folds: list[FoldResult], actions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "holdout_run_id",
                "linear_utility",
                "static_utility",
                "adaptive_utility",
                "selector_utility",
                "oracle_utility",
                "linear_minus_static",
                "linear_minus_adaptive",
                "linear_minus_selector",
                *[f"{action}_actions" for action in actions],
            ]
        )
        for fold in folds:
            writer.writerow(
                [
                    fold.holdout_run_id,
                    f"{fold.linear_utility:.12f}",
                    f"{fold.static_utility:.12f}",
                    f"{fold.adaptive_utility:.12f}",
                    f"{fold.selector_utility:.12f}",
                    f"{fold.oracle_utility:.12f}",
                    f"{fold.linear_utility - fold.static_utility:.12f}",
                    f"{fold.linear_utility - fold.adaptive_utility:.12f}",
                    f"{fold.linear_utility - fold.selector_utility:.12f}",
                    *[fold.action_counts[action] for action in actions],
                ]
            )


def write_report(path: Path, model: LinearModel, folds: list[FoldResult], examples: list[Example]) -> None:
    linear = safe_mean([fold.linear_utility for fold in folds])
    static = safe_mean([fold.static_utility for fold in folds])
    adaptive = safe_mean([fold.adaptive_utility for fold in folds])
    selector = safe_mean([fold.selector_utility for fold in folds])
    oracle = safe_mean([fold.oracle_utility for fold in folds])
    action_counts = {
        action: sum(fold.action_counts[action] for fold in folds)
        for action in model.actions
    }
    lines = [
        "# Linear Policy Agent",
        "",
        "## Method",
        "",
        "This is a multi-action linear utility agent. It fits one ridge regression per action, predicts each action's utility from normalized market-state features, and chooses the highest-scored action.",
        "",
        "## Setup",
        "",
        f"- Training examples: `{len(examples)}`.",
        f"- Actions: `{', '.join(model.actions)}`.",
        f"- Ridge: `{model.ridge:.6f}`.",
        f"- Features: `{', '.join(FEATURES)}`.",
        "",
        "## Leave-One-Dataset-Out Result",
        "",
        f"- Linear agent utility: `{linear:.6f}`.",
        f"- Static utility: `{static:.6f}`.",
        f"- Adaptive utility: `{adaptive:.6f}`.",
        f"- Selector utility: `{selector:.6f}`.",
        f"- Oracle utility across actions: `{oracle:.6f}`.",
        f"- Linear minus adaptive: `{linear - adaptive:.6f}`.",
        f"- Linear minus selector: `{linear - selector:.6f}`.",
        "",
        "## Action Mix",
        "",
    ]
    for action in model.actions:
        lines.append(f"- `{action}`: `{action_counts[action]}` windows.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is an agent architecture step, not a trading claim. It gives Rust a generic multi-action learned policy format that can later be replaced by a stronger contextual bandit or reinforcement-style controller.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    actions = parse_actions(args.actions)
    args.action_on = "selector" if "selector" in actions else actions[-1]
    args.action_off = "adaptive" if "adaptive" in actions else actions[0]
    examples = read_examples(args)
    missing_actions = sorted(
        {
            action
            for action in actions
            if any(action not in example.utilities for example in examples)
        }
    )
    if missing_actions:
        raise ValueError(f"missing utilities for actions: {', '.join(missing_actions)}")
    scale = feature_scale(examples)
    normalized = normalize_examples(examples, scale)
    model = train_model(normalized, actions, scale, args.feature_window, args.ridge)
    folds = leave_one_dataset_out(normalized, actions, scale, args.feature_window, args.ridge)
    write_model(project_path(args.model_output), model, examples)
    write_folds(project_path(args.folds_output), folds, actions)
    write_report(project_path(args.report_output), model, folds, examples)

    print("Linear policy agent")
    print(f"examples: {len(examples)}")
    print(f"holdout utility: {safe_mean([fold.linear_utility for fold in folds]):.6f}")
    print(f"adaptive utility: {safe_mean([fold.adaptive_utility for fold in folds]):.6f}")
    print(f"selector utility: {safe_mean([fold.selector_utility for fold in folds]):.6f}")
    print(f"wrote {project_path(args.model_output)}")
    print(f"wrote {project_path(args.folds_output)}")
    print(f"wrote {project_path(args.report_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

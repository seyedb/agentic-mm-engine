#!/usr/bin/env python3
"""Evaluate a small LinUCB contextual-bandit policy selector offline."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_policy_selector import (
    FEATURES,
    Example,
    feature_scale,
    normalize_examples,
    project_path,
    read_examples,
    safe_mean,
)


DEFAULT_OUTPUT = Path("target/research/bandit_selector_runs.csv")
DEFAULT_REPORT = Path("target/research/bandit_selector.md")
DEFAULT_FOLDS_OUTPUT = Path("target/research/bandit_selector_folds.csv")


@dataclass
class LinUcbArm:
    matrix: list[list[float]]
    reward_vector: list[float]


@dataclass(frozen=True)
class Decision:
    run_id: str
    window: str
    action: str
    reward: float
    best_action: str
    best_reward: float
    regret: float


@dataclass(frozen=True)
class FoldResult:
    holdout_run_id: str
    windows: int
    bandit_utility: float
    adaptive_utility: float
    selector_utility: float
    learned_selector_utility: float
    oracle_utility: float
    regret: float
    selector_actions: int
    adaptive_actions: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline LinUCB selector study.")
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
    parser.add_argument("--actions", default="adaptive,selector")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--feature-policy", default="static")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds-output", type=Path, default=DEFAULT_FOLDS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def parse_actions(value: str) -> list[str]:
    actions = [action.strip() for action in value.split(",") if action.strip()]
    if len(actions) < 2:
        raise ValueError("--actions must contain at least two policies")
    return actions


def chronological_examples(examples: list[Example]) -> list[Example]:
    return sorted(examples, key=lambda example: (example.run_id, example.window))


def identity(size: int, value: float) -> list[list[float]]:
    return [[value if row == col else 0.0 for col in range(size)] for row in range(size)]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix]


def dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(len(left)))


def invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [row[:] + identity(size, 1.0)[index] for index, row in enumerate(matrix)]

    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular LinUCB matrix")
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

    return [row[size:] for row in augmented]


def feature_vector(example: Example) -> list[float]:
    return [example.features[feature] for feature in FEATURES]


def choose_action(
    arms: dict[str, LinUcbArm],
    actions: list[str],
    features: list[float],
    alpha: float,
) -> str:
    scores = []
    for action in actions:
        inverse = invert(arms[action].matrix)
        theta = mat_vec(inverse, arms[action].reward_vector)
        uncertainty = max(dot(features, mat_vec(inverse, features)), 0.0) ** 0.5
        scores.append((dot(theta, features) + alpha * uncertainty, action))
    return max(scores, key=lambda item: (item[0], item[1]))[1]


def update_arm(arm: LinUcbArm, features: list[float], reward: float) -> None:
    for row in range(len(features)):
        arm.reward_vector[row] += reward * features[row]
        for col in range(len(features)):
            arm.matrix[row][col] += features[row] * features[col]


def best_action(example: Example, actions: list[str]) -> str:
    return max(actions, key=lambda action: example.utilities[action])


def fresh_arms(actions: list[str], ridge: float) -> dict[str, LinUcbArm]:
    size = len(FEATURES)
    return {
        action: LinUcbArm(identity(size, ridge), [0.0 for _feature in FEATURES])
        for action in actions
    }


def run_linucb(examples: list[Example], actions: list[str], alpha: float, ridge: float) -> list[Decision]:
    arms = fresh_arms(actions, ridge)
    decisions = []

    for index, example in enumerate(examples):
        features = feature_vector(example)
        action = actions[index] if index < len(actions) else choose_action(arms, actions, features, alpha)
        reward = example.utilities[action]
        oracle = best_action(example, actions)
        oracle_reward = example.utilities[oracle]
        update_arm(arms[action], features, reward)
        decisions.append(
            Decision(
                run_id=example.run_id,
                window=example.window,
                action=action,
                reward=reward,
                best_action=oracle,
                best_reward=oracle_reward,
                regret=oracle_reward - reward,
            )
        )

    return decisions


def pretrain_arms(
    examples: list[Example],
    actions: list[str],
    alpha: float,
    ridge: float,
) -> dict[str, LinUcbArm]:
    arms = fresh_arms(actions, ridge)
    for index, example in enumerate(examples):
        features = feature_vector(example)
        action = actions[index] if index < len(actions) else choose_action(arms, actions, features, alpha)
        update_arm(arms[action], features, example.utilities[action])
    return arms


def evaluate_greedy(
    examples: list[Example],
    actions: list[str],
    arms: dict[str, LinUcbArm],
) -> list[Decision]:
    decisions = []
    for example in examples:
        features = feature_vector(example)
        action = choose_action(arms, actions, features, 0.0)
        reward = example.utilities[action]
        oracle = best_action(example, actions)
        oracle_reward = example.utilities[oracle]
        decisions.append(
            Decision(
                run_id=example.run_id,
                window=example.window,
                action=action,
                reward=reward,
                best_action=oracle,
                best_reward=oracle_reward,
                regret=oracle_reward - reward,
            )
        )
    return decisions


def leave_one_dataset_out(
    examples: list[Example],
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
                adaptive_utility=baseline_utility(test, "adaptive"),
                selector_utility=baseline_utility(test, "selector"),
                learned_selector_utility=baseline_utility(test, "learned_selector"),
                oracle_utility=safe_mean([decision.best_reward for decision in decisions]),
                regret=sum(decision.regret for decision in decisions),
                selector_actions=sum(decision.action == "selector" for decision in decisions),
                adaptive_actions=sum(decision.action == "adaptive" for decision in decisions),
            )
        )
    return folds


def baseline_utility(examples: list[Example], policy: str) -> float:
    return safe_mean([example.utilities[policy] for example in examples])


def write_decisions(path: Path, decisions: list[Decision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "window", "action", "reward", "best_action", "best_reward", "regret"])
        for decision in decisions:
            writer.writerow(
                [
                    decision.run_id,
                    decision.window,
                    decision.action,
                    f"{decision.reward:.12f}",
                    decision.best_action,
                    f"{decision.best_reward:.12f}",
                    f"{decision.regret:.12f}",
                ]
            )


def write_folds(path: Path, folds: list[FoldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "holdout_run_id",
                "windows",
                "bandit_utility",
                "adaptive_utility",
                "selector_utility",
                "learned_selector_utility",
                "bandit_minus_adaptive",
                "bandit_minus_selector",
                "bandit_minus_learned_selector",
                "oracle_utility",
                "regret",
                "selector_actions",
                "adaptive_actions",
            ]
        )
        for fold in folds:
            writer.writerow(
                [
                    fold.holdout_run_id,
                    fold.windows,
                    f"{fold.bandit_utility:.12f}",
                    f"{fold.adaptive_utility:.12f}",
                    f"{fold.selector_utility:.12f}",
                    f"{fold.learned_selector_utility:.12f}",
                    f"{fold.bandit_utility - fold.adaptive_utility:.12f}",
                    f"{fold.bandit_utility - fold.selector_utility:.12f}",
                    f"{fold.bandit_utility - fold.learned_selector_utility:.12f}",
                    f"{fold.oracle_utility:.12f}",
                    f"{fold.regret:.12f}",
                    fold.selector_actions,
                    fold.adaptive_actions,
                ]
            )


def write_report(
    path: Path,
    *,
    examples: list[Example],
    decisions: list[Decision],
    actions: list[str],
    alpha: float,
    ridge: float,
    folds: list[FoldResult],
) -> None:
    action_counts = {action: sum(decision.action == action for decision in decisions) for action in actions}
    bandit_utility = safe_mean([decision.reward for decision in decisions])
    oracle_utility = safe_mean([decision.best_reward for decision in decisions])
    regret = sum(decision.regret for decision in decisions)
    rows = [
        "# Contextual Bandit Selector",
        "",
        "## Method",
        "",
        "This is an offline LinUCB contextual-bandit study. It replays existing quote windows in chronological order, observes normalized market-state features, chooses a policy action, receives that policy's realized paper utility, and updates the selected arm.",
        "",
        "## Setup",
        "",
        f"- Examples: `{len(examples)}`.",
        f"- Actions: `{', '.join(actions)}`.",
        f"- Alpha: `{alpha:.3f}`.",
        f"- Ridge: `{ridge:.3f}`.",
        f"- Features: `{', '.join(FEATURES)}`.",
        "",
        "## Result",
        "",
        f"- LinUCB utility: `{bandit_utility:.6f}`.",
        f"- Always adaptive utility: `{baseline_utility(examples, 'adaptive'):.6f}`.",
        f"- Hand selector utility: `{baseline_utility(examples, 'selector'):.6f}`.",
        f"- Logistic learned-selector utility: `{baseline_utility(examples, 'learned_selector'):.6f}`.",
        f"- Oracle utility across available actions: `{oracle_utility:.6f}`.",
        f"- Cumulative regret: `{regret:.6f}`.",
        "",
        "## Leave-One-Dataset-Out",
        "",
        f"- Fold bandit utility: `{safe_mean([fold.bandit_utility for fold in folds]):.6f}`.",
        f"- Fold adaptive utility: `{safe_mean([fold.adaptive_utility for fold in folds]):.6f}`.",
        f"- Fold selector utility: `{safe_mean([fold.selector_utility for fold in folds]):.6f}`.",
        f"- Fold logistic learned-selector utility: `{safe_mean([fold.learned_selector_utility for fold in folds]):.6f}`.",
        f"- Fold wins versus adaptive: `{sum(fold.bandit_utility > fold.adaptive_utility for fold in folds)}/{len(folds)}`.",
        f"- Fold wins versus selector: `{sum(fold.bandit_utility > fold.selector_utility for fold in folds)}/{len(folds)}`.",
        f"- Fold wins versus logistic learned-selector: `{sum(fold.bandit_utility > fold.learned_selector_utility for fold in folds)}/{len(folds)}`.",
        "",
        "## Action Mix",
        "",
    ]
    for action in actions:
        rows.append(f"- `{action}`: `{action_counts[action]}` windows.")
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a research diagnostic, not a live trading policy. The chronological replay result is mildly encouraging, but the leave-one-dataset-out result is negative: LinUCB does not beat adaptive, the hand selector, or the logistic learned selector out of sample. The bandit should not be wired into Rust unless a later validation design beats these simpler selectors.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows))


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
    write_decisions(project_path(args.output), decisions)
    write_folds(project_path(args.folds_output), folds)
    write_report(
        project_path(args.report_output),
        examples=normalized,
        decisions=decisions,
        actions=actions,
        alpha=args.alpha,
        ridge=args.ridge,
        folds=folds,
    )

    print("Contextual bandit selector")
    print(f"examples: {len(examples)}")
    print(f"linucb utility: {safe_mean([decision.reward for decision in decisions]):.6f}")
    print(f"adaptive utility: {baseline_utility(normalized, 'adaptive'):.6f}")
    print(f"selector utility: {baseline_utility(normalized, 'selector'):.6f}")
    print(f"learned selector utility: {baseline_utility(normalized, 'learned_selector'):.6f}")
    print(f"fold bandit utility: {safe_mean([fold.bandit_utility for fold in folds]):.6f}")
    print(f"wrote {project_path(args.output)}")
    print(f"wrote {project_path(args.folds_output)}")
    print(f"wrote {project_path(args.report_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

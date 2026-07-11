#!/usr/bin/env python3
"""Train a small policy-selection gate from paper-session window results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOW_RESULTS = Path("target/research/policy_gate_window_results.csv")
DEFAULT_POLICY_GATE_DIR = Path("target/research/policy_gate/configured")
DEFAULT_MODEL_OUTPUT = Path("target/research/learned_policy_selector_model.json")
DEFAULT_FOLDS_OUTPUT = Path("target/research/learned_policy_selector_folds.csv")
DEFAULT_REPORT_OUTPUT = Path("target/research/learned_policy_selector.md")
FEATURES = (
    "estimated_volatility",
    "observed_spread",
    "max_observed_spread",
    "abs_mid_move",
    "abs_inventory",
    "drawdown",
)


@dataclass(frozen=True)
class Example:
    run_id: str
    window: str
    features: dict[str, float]
    utilities: dict[str, float]


@dataclass(frozen=True)
class GateModel:
    action_on: str
    action_off: str
    weights: dict[str, float]
    threshold: float
    feature_scale: dict[str, float]


@dataclass(frozen=True)
class FoldResult:
    holdout_run_id: str
    learned_utility: float
    static_utility: float
    adaptive_utility: float
    selector_utility: float
    action_on_count: int
    action_off_count: int
    opportunities: int
    missed_opportunities: int
    target_accuracy: float
    captured_advantage: float
    train_utility: float
    train_score: float
    model: GateModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a simple threshold gate that chooses between two paper policies "
            "from quote-window features."
        )
    )
    parser.add_argument("--window-results", type=Path, default=DEFAULT_WINDOW_RESULTS)
    parser.add_argument("--policy-gate-dir", type=Path, default=DEFAULT_POLICY_GATE_DIR)
    parser.add_argument("--assumption", default="configured")
    parser.add_argument("--action-on", default="selector")
    parser.add_argument("--action-off", default="adaptive")
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=0.00025,
        help="Require this much action-on utility advantage before labeling a selector opportunity.",
    )
    parser.add_argument(
        "--feature-policy",
        default="static",
        help="Policy output used to compute window features before training the gate.",
    )
    parser.add_argument(
        "--feature-window",
        type=int,
        default=30,
        help="Rolling feature window used by Rust when applying the learned gate.",
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--folds-output", type=Path, default=DEFAULT_FOLDS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_utilities(path: Path, assumption: str) -> dict[tuple[str, str], dict[str, float]]:
    utilities: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        missing = {"assumption", "run_id", "window", "policy", "utility"}.difference(
            reader.fieldnames
        )
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["assumption"] != assumption:
                continue
            key = (row["run_id"], row["window"])
            utilities.setdefault(key, {})[row["policy"]] = float(row["utility"])
    return utilities


def read_examples(args: argparse.Namespace) -> list[Example]:
    utilities = read_utilities(project_path(args.window_results), args.assumption)
    examples = []
    gate_dir = project_path(args.policy_gate_dir)
    for runs_path in sorted(gate_dir.glob("*/paper_policy_window_runs.csv")):
        run_id = runs_path.parent.name
        with runs_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{runs_path}: CSV file is empty")
            rows_by_window = {
                row["window"]: row for row in reader if row["policy"] == args.feature_policy
            }

        for window, row in rows_by_window.items():
            key = (run_id, window)
            window_utilities = utilities.get(key)
            if window_utilities is None:
                continue
            required = {
                "static",
                "adaptive",
                "selector",
                args.action_on,
                args.action_off,
                args.feature_policy,
            }
            if not required.issubset(window_utilities):
                continue
            examples.append(
                Example(
                    run_id=run_id,
                    window=window,
                    features=read_features(Path(row["output"])),
                    utilities=window_utilities,
                )
            )
    if not examples:
        raise ValueError("no training examples found; run policy_evaluation_gate.py first")
    return examples


def read_features(path: Path) -> dict[str, float]:
    if not path.exists():
        raise ValueError(f"missing paper-session output: {path}")
    volatilities = []
    observed_spreads = []
    mid_prices = []
    inventories = []
    drawdowns = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is empty")
        missing = {
            "mid_price",
            "observed_bid",
            "observed_ask",
            "estimated_volatility",
            "inventory",
            "drawdown",
        }.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            volatilities.append(float(row["estimated_volatility"]))
            mid_prices.append(float(row["mid_price"]))
            inventories.append(abs(float(row["inventory"])))
            drawdowns.append(float(row["drawdown"]))
            if row["observed_bid"] and row["observed_ask"]:
                observed_spreads.append(float(row["observed_ask"]) - float(row["observed_bid"]))

    mid_moves = [
        abs(mid_prices[index] - mid_prices[index - 1])
        for index in range(1, len(mid_prices))
    ]
    return {
        "estimated_volatility": safe_mean(volatilities),
        "observed_spread": safe_mean(observed_spreads),
        "max_observed_spread": max(observed_spreads) if observed_spreads else 0.0,
        "abs_mid_move": safe_mean(mid_moves),
        "abs_inventory": safe_mean(inventories),
        "drawdown": safe_mean(drawdowns),
    }


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def feature_scale(examples: list[Example]) -> dict[str, float]:
    return {
        feature: max(max(abs(example.features[feature]) for example in examples), 1e-12)
        for feature in FEATURES
    }


def normalize_examples(examples: list[Example], scale: dict[str, float]) -> list[Example]:
    normalized = []
    for example in examples:
        normalized.append(
            Example(
                run_id=example.run_id,
                window=example.window,
                features={
                    feature: example.features[feature] / scale[feature]
                    for feature in FEATURES
                },
                utilities=example.utilities,
            )
        )
    return normalized


def candidate_models(
    action_on: str,
    action_off: str,
    scale: dict[str, float],
) -> list[GateModel]:
    weight_grid = (-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    thresholds = (-2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
    models = [
        GateModel(action_on, action_off, {feature: 0.0 for feature in FEATURES}, 0.0, scale),
        GateModel(action_on, action_off, {feature: 0.0 for feature in FEATURES}, 1.0, scale),
    ]

    feature_sets = [(feature,) for feature in FEATURES]
    feature_sets.extend(itertools.combinations(FEATURES, 2))
    for selected_features in feature_sets:
        for weights in itertools.product(weight_grid, repeat=len(selected_features)):
            model_weights = {feature: 0.0 for feature in FEATURES}
            for feature, weight in zip(selected_features, weights):
                model_weights[feature] = weight
            for threshold in thresholds:
                models.append(
                    GateModel(action_on, action_off, model_weights, threshold, scale)
                )
    return models


def choose_action(model: GateModel, features: dict[str, float]) -> str:
    score = sum(model.weights[feature] * features[feature] for feature in FEATURES)
    if score >= model.threshold:
        return model.action_on
    return model.action_off


def target_action(
    example: Example,
    action_on: str,
    action_off: str,
    margin: float,
) -> str:
    if example.utilities[action_on] > example.utilities[action_off] + margin:
        return action_on
    return action_off


def selector_opportunity(
    example: Example,
    action_on: str,
    action_off: str,
    margin: float,
) -> bool:
    return example.utilities[action_on] > example.utilities[action_off] + margin


def evaluate_model(
    model: GateModel,
    examples: list[Example],
    margin: float,
) -> tuple[float, int, int, int, int, float, float]:
    utilities = []
    action_on_count = 0
    action_off_count = 0
    correct = 0
    opportunities = 0
    missed_opportunities = 0
    captured_advantage = 0.0
    for example in examples:
        action = choose_action(model, example.features)
        utilities.append(example.utilities[action])
        if action == model.action_on:
            action_on_count += 1
        else:
            action_off_count += 1
        target = target_action(example, model.action_on, model.action_off, margin)
        if action == target:
            correct += 1
        if selector_opportunity(example, model.action_on, model.action_off, margin):
            opportunities += 1
            if action == model.action_on:
                captured_advantage += example.utilities[model.action_on] - example.utilities[model.action_off]
            else:
                missed_opportunities += 1
    target_accuracy = correct / len(examples) if examples else 0.0
    return (
        safe_mean(utilities),
        action_on_count,
        action_off_count,
        opportunities,
        missed_opportunities,
        target_accuracy,
        captured_advantage,
    )


def training_score(model: GateModel, examples: list[Example], margin: float) -> float:
    score = 0.0
    for example in examples:
        action = choose_action(model, example.features)
        target = target_action(example, model.action_on, model.action_off, margin)
        advantage = abs(example.utilities[model.action_on] - example.utilities[model.action_off])
        weight = max(advantage, margin)
        if action == target:
            score += weight
        else:
            score -= weight
    return score / len(examples) if examples else 0.0


def train_model(
    examples: list[Example],
    action_on: str,
    action_off: str,
    scale: dict[str, float],
    margin: float,
) -> tuple[GateModel, float, float]:
    best_model = None
    best_score = None
    best_utility = 0.0
    for model in candidate_models(action_on, action_off, scale):
        score = training_score(model, examples, margin)
        (
            utility,
            _action_on_count,
            _action_off_count,
            _opportunities,
            _missed_opportunities,
            _target_accuracy,
            _captured_advantage,
        ) = evaluate_model(model, examples, margin)
        if best_score is None or score > best_score or (score == best_score and utility > best_utility):
            best_model = model
            best_score = score
            best_utility = utility
    assert best_model is not None
    assert best_score is not None
    return best_model, best_utility, best_score


def leave_one_dataset_out(
    examples: list[Example],
    action_on: str,
    action_off: str,
    scale: dict[str, float],
    margin: float,
) -> list[FoldResult]:
    folds = []
    run_ids = sorted({example.run_id for example in examples})
    for holdout in run_ids:
        train = [example for example in examples if example.run_id != holdout]
        test = [example for example in examples if example.run_id == holdout]
        model, train_utility, train_score = train_model(train, action_on, action_off, scale, margin)
        (
            learned_utility,
            action_on_count,
            action_off_count,
            opportunities,
            missed_opportunities,
            target_accuracy,
            captured_advantage,
        ) = evaluate_model(model, test, margin)
        folds.append(
            FoldResult(
                holdout_run_id=holdout,
                learned_utility=learned_utility,
                static_utility=baseline_utility(test, "static"),
                adaptive_utility=baseline_utility(test, "adaptive"),
                selector_utility=baseline_utility(test, "selector"),
                action_on_count=action_on_count,
                action_off_count=action_off_count,
                opportunities=opportunities,
                missed_opportunities=missed_opportunities,
                target_accuracy=target_accuracy,
                captured_advantage=captured_advantage,
                train_utility=train_utility,
                train_score=train_score,
                model=model,
            )
        )
    return folds


def baseline_utility(examples: list[Example], policy: str) -> float:
    return safe_mean([example.utilities[policy] for example in examples])


def write_model(
    path: Path,
    model: GateModel,
    examples: list[Example],
    train_utility: float,
    train_score: float,
    feature_policy: str,
    feature_window: int,
    selection_margin: float,
) -> None:
    payload = {
        "model_type": "threshold_policy_gate",
        "action_on": model.action_on,
        "action_off": model.action_off,
        "feature_policy": feature_policy,
        "feature_window": feature_window,
        "selection_margin": selection_margin,
        "features": list(FEATURES),
        "weights": model.weights,
        "threshold": model.threshold,
        "feature_scale": model.feature_scale,
        "training_examples": len(examples),
        "training_datasets": sorted({example.run_id for example in examples}),
        "training_utility": train_utility,
        "training_score": train_score,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_folds(path: Path, folds: list[FoldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "holdout_run_id",
                "learned_utility",
                "static_utility",
                "adaptive_utility",
                "selector_utility",
                "learned_minus_static",
                "learned_minus_adaptive",
                "learned_minus_selector",
                "action_on_count",
                "action_off_count",
                "opportunities",
                "missed_opportunities",
                "target_accuracy",
                "captured_advantage",
                "train_utility",
                "train_score",
                "weights",
                "threshold",
            ]
        )
        for fold in folds:
            writer.writerow(
                [
                    fold.holdout_run_id,
                    f"{fold.learned_utility:.12f}",
                    f"{fold.static_utility:.12f}",
                    f"{fold.adaptive_utility:.12f}",
                    f"{fold.selector_utility:.12f}",
                    f"{fold.learned_utility - fold.static_utility:.12f}",
                    f"{fold.learned_utility - fold.adaptive_utility:.12f}",
                    f"{fold.learned_utility - fold.selector_utility:.12f}",
                    fold.action_on_count,
                    fold.action_off_count,
                    fold.opportunities,
                    fold.missed_opportunities,
                    f"{fold.target_accuracy:.12f}",
                    f"{fold.captured_advantage:.12f}",
                    f"{fold.train_utility:.12f}",
                    f"{fold.train_score:.12f}",
                    json.dumps(fold.model.weights, sort_keys=True),
                    f"{fold.model.threshold:.12f}",
                ]
            )


def write_report(
    path: Path,
    model: GateModel,
    train_utility: float,
    train_score: float,
    folds: list[FoldResult],
    examples: list[Example],
    feature_policy: str,
    selection_margin: float,
) -> None:
    learned = safe_mean([fold.learned_utility for fold in folds])
    static = safe_mean([fold.static_utility for fold in folds])
    adaptive = safe_mean([fold.adaptive_utility for fold in folds])
    selector = safe_mean([fold.selector_utility for fold in folds])
    adaptive_wins = sum(fold.learned_utility > fold.adaptive_utility for fold in folds)
    selector_wins = sum(fold.learned_utility > fold.selector_utility for fold in folds)
    action_on_count = sum(fold.action_on_count for fold in folds)
    action_off_count = sum(fold.action_off_count for fold in folds)
    total_actions = action_on_count + action_off_count
    action_on_rate = action_on_count / total_actions * 100.0 if total_actions else 0.0
    opportunities = sum(fold.opportunities for fold in folds)
    missed_opportunities = sum(fold.missed_opportunities for fold in folds)
    captured_advantage = sum(fold.captured_advantage for fold in folds)
    target_accuracy = safe_mean([fold.target_accuracy for fold in folds])

    lines = [
        "# Learned Policy Selector",
        "",
        "## Method",
        "",
        (
            "A small threshold gate chooses between "
            f"`{model.action_off}` and `{model.action_on}` from normalized market-state "
            f"features measured from `{feature_policy}` runs. Training uses "
            f"a `{selection_margin:.6f}` utility margin before labeling `{model.action_on}` "
            f"better than `{model.action_off}`. "
            "Validation leaves one quote dataset out, so each reported fold is "
            "evaluated on a dataset that was not used to choose the gate."
        ),
        "",
        "## Full-Data Model",
        "",
        f"- Training examples: `{len(examples)}`.",
        f"- Training utility: `{train_utility:.6f}`.",
        f"- Training target score: `{train_score:.6f}`.",
        f"- Threshold: `{model.threshold:.6f}`.",
        f"- Weights: `{json.dumps(model.weights, sort_keys=True)}`.",
        "",
        "## Holdout Result",
        "",
        f"- Learned gate utility: `{learned:.6f}`.",
        f"- Static utility: `{static:.6f}`.",
        f"- Adaptive utility: `{adaptive:.6f}`.",
        f"- Hand-tuned selector utility: `{selector:.6f}`.",
        f"- Learned minus adaptive: `{learned - adaptive:.6f}`.",
        f"- Learned minus selector: `{learned - selector:.6f}`.",
        f"- Holdout wins versus adaptive: `{adaptive_wins}/{len(folds)}`.",
        f"- Holdout wins versus selector: `{selector_wins}/{len(folds)}`.",
        f"- `{model.action_on}` action rate: `{action_on_rate:.2f}%`.",
        f"- Selector opportunities: `{opportunities}`.",
        f"- Missed selector opportunities: `{missed_opportunities}`.",
        f"- Captured selector advantage: `{captured_advantage:.6f}`.",
        f"- Target accuracy: `{target_accuracy:.2%}`.",
        "",
        "## Interpretation",
        "",
        (
            "This is a first ML-agent proof of concept, not a trading edge. Passing this "
            "check means the project has a reproducible learned control loop; failing "
            "against the hand-tuned selector means the selector remains the stronger "
            "baseline for now."
        ),
        "",
        "## Folds",
        "",
        "| holdout | learned | static | adaptive | selector | on | off | opp | missed | target_acc | captured_adv |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in folds:
        lines.append(
            "| "
            f"{fold.holdout_run_id} | "
            f"{fold.learned_utility:.6f} | "
            f"{fold.static_utility:.6f} | "
            f"{fold.adaptive_utility:.6f} | "
            f"{fold.selector_utility:.6f} | "
            f"{fold.action_on_count} | "
            f"{fold.action_off_count} | "
            f"{fold.opportunities} | "
            f"{fold.missed_opportunities} | "
            f"{fold.target_accuracy:.2%} | "
            f"{fold.captured_advantage:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    examples = read_examples(args)
    scale = feature_scale(examples)
    normalized_examples = normalize_examples(examples, scale)
    model, train_utility, train_score = train_model(
        normalized_examples,
        args.action_on,
        args.action_off,
        scale,
        args.selection_margin,
    )
    folds = leave_one_dataset_out(
        normalized_examples,
        args.action_on,
        args.action_off,
        scale,
        args.selection_margin,
    )
    write_model(
        project_path(args.model_output),
        model,
        examples,
        train_utility,
        train_score,
        args.feature_policy,
        args.feature_window,
        args.selection_margin,
    )
    write_folds(project_path(args.folds_output), folds)
    write_report(
        project_path(args.report_output),
        model,
        train_utility,
        train_score,
        folds,
        examples,
        args.feature_policy,
        args.selection_margin,
    )

    print("Learned policy selector")
    print(f"examples: {len(examples)}")
    print(f"training utility: {train_utility:.6f}")
    print(f"training target score: {train_score:.6f}")
    print(f"holdout utility: {safe_mean([fold.learned_utility for fold in folds]):.6f}")
    print(f"holdout selector action rate: {safe_mean([fold.action_on_count / (fold.action_on_count + fold.action_off_count) for fold in folds if fold.action_on_count + fold.action_off_count > 0]) * 100.0:.2f}%")
    print(f"wrote {project_path(args.model_output)}")
    print(f"wrote {project_path(args.folds_output)}")
    print(f"wrote {project_path(args.report_output)}")


if __name__ == "__main__":
    main()

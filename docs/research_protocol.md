# Research Protocol

This project is an experimental market-making research engine, not a production trading system. The current goal is to test whether a small learned policy gate can improve paper-session quoting decisions on public top-of-book data.

## Research Question

Can a Python-trained policy gate choose between `adaptive` and `selector` quoting in a way that improves risk-adjusted paper-session utility when Rust executes the learned policy?

The learned policy is considered interesting only if it improves over simple baselines after being exported to JSON and loaded back into the Rust paper engine.

## Current Policies

- `static`: uses the controller quote unchanged.
- `adaptive`: widens and skews quotes from observed spread, volatility, and inventory.
- `hybrid`: switches from static to adaptive on risk triggers.
- `selector`: weighted rule-based selector between static and adaptive behavior.
- `learned_selector`: Rust-executed learned gate trained by Python.

## Evaluation Loop

Run the gate once to produce window-level policy outcomes:

```bash
python3 research/policy_evaluation_gate.py
```

Train the learned gate from those outcomes:

```bash
python3 research/train_policy_selector.py
```

Rerun the gate so Rust loads and evaluates the learned model:

```bash
python3 research/policy_evaluation_gate.py
python3 research/write_project_report.py
```

The learned model artifact is written to:

```text
target/research/learned_policy_selector_model.json
```

## Metrics

The main utility is:

```text
pnl - 2.0 * drawdown - 0.02 * mean_abs_inventory
```

Reports also track fills, fees, adaptive-step percentage, policy triggers, dataset wins, and window wins.

## Fill Assumptions

The policy gate evaluates each policy under:

- `configured`: the fill model in the committed run configs.
- `conservative_fill`: a stricter touch-intensity fill assumption.
- `liquid_fill`: a more permissive touch-intensity fill assumption.

A result is not robust if it only works under one fill assumption.

## Success Criteria

A useful learned-policy result should satisfy most of these:

- Rust-executed `learned_selector` beats `adaptive` under configured assumptions.
- Rust-executed `learned_selector` is competitive with or better than the hand-tuned `selector`.
- The result does not collapse under conservative fill assumptions.
- Python leave-one-dataset-out validation does not contradict the Rust gate result.
- Trigger attribution shows the learned policy is not simply always-adaptive or never-adaptive.

These criteria are research checks, not proof of a trading edge.

## Current Interpretation

The latest result is promising because the Rust-executed learned selector leads under configured assumptions after adding fresh quote datasets. It is still a small-sample result, and liquid-fill assumptions continue to favor adaptive quoting. The next validation step is a live public-data paper demonstration, not real trading.

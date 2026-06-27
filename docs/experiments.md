# Experiments

Experiments are configured with JSON files in `configs/`.

## Run

```bash
cargo run
cargo run -- configs/baseline_sweep.json
cargo run -- configs/high_volatility_sweep.json
```

The default config is `configs/baseline_sweep.json`.

## Sweep Config

A sweep config contains:

- simulation settings
- spread values
- skew coefficient values
- scoring settings

The runner evaluates every spread/skew combination and ranks the results.

## Output

The terminal prints the top sweep results. The full ranked sweep is written to:

```text
target/reports/sweep_results.csv
```

## Score

The current score is intentionally simple:

```text
score = final_pnl
      - drawdown_weight * max_drawdown
      - inventory_weight * max_abs_inventory
      - inactivity_penalty
```

The inactivity penalty discourages strategies that avoid trading entirely.

## Current Regimes

- `baseline_sweep.json`: lower-volatility baseline environment
- `high_volatility_sweep.json`: higher volatility, noisier fills, stronger adverse selection

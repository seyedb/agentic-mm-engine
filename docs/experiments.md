# Experiments

Experiments are configured with JSON files in `configs/`.

## Run

```bash
cargo run
cargo run -- configs/baseline_sweep.json
cargo run -- configs/high_volatility_sweep.json
cargo run -- configs/baseline_sweep.json configs/high_volatility_sweep.json
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

The terminal prints the top sweep results for each config. Full ranked sweeps are written to:

```text
target/reports/<config_name>.csv
```

The best result from each config is also written to:

```text
target/reports/regime_summary.csv
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

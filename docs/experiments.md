# Experiments

Experiments are configured with JSON files in `configs/`.

## Run

```bash
cargo run
cargo run -- configs/baseline_sweep.json
cargo run -- configs/high_volatility_sweep.json
cargo run -- configs/volatility_aware_sweep.json
cargo run -- configs/baseline_sweep.json configs/high_volatility_sweep.json
```

The default config is `configs/baseline_sweep.json`.

## Sweep Config

A sweep config contains:

- an optional experiment name
- simulation settings
- optional seed values for multi-seed aggregation
- strategy sweep settings
- scoring settings

The runner evaluates every spread/skew combination across the configured seeds, averages the metrics, and ranks the aggregate results.

## Output

The terminal prints the top sweep results for each config. Full ranked sweeps are written using the config `name` when present:

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
- `volatility_aware_sweep.json`: high-volatility environment using volatility-aware quoting

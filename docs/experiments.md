# Experiments

Experiments are configured with JSON files in `configs/`.

## Run

```bash
cargo run
cargo run -- configs/baseline_sweep.json
cargo run -- configs/baseline_volatility_aware_sweep.json
cargo run -- configs/baseline_inventory_risk_sweep.json
cargo run -- configs/high_volatility_sweep.json
cargo run -- configs/volatility_aware_sweep.json
cargo run -- configs/high_volatility_inventory_risk_sweep.json
cargo run -- configs/baseline_sweep.json configs/baseline_volatility_aware_sweep.json configs/baseline_inventory_risk_sweep.json configs/high_volatility_sweep.json configs/volatility_aware_sweep.json configs/high_volatility_inventory_risk_sweep.json
```

The default config is `configs/baseline_sweep.json`.

## Sweep Config

A sweep config contains:

- an optional experiment name
- simulation settings
- fill model settings
- optional seed values for multi-seed aggregation
- strategy sweep settings
- scoring settings

The runner evaluates every configured strategy parameter combination across the configured seeds, averages the metrics, tracks stability statistics, and ranks the aggregate results.

## Output

The terminal prints the top sweep results for each config. Full ranked sweeps are written using the config `name` when present:

```text
target/reports/<config_name>.csv
```

The best result from each config is also written to:

```text
target/reports/regime_summary.csv
```

Multi-seed outputs include standard deviation fields such as `score_std`, `final_pnl_std`, and `max_drawdown_std`.

## Score

The raw score is intentionally simple:

```text
score = final_pnl
      - drawdown_weight * max_drawdown
      - inventory_weight * max_abs_inventory
      - inactivity_penalty
```

The inactivity penalty discourages strategies that avoid trading entirely.

Rankings use a stability-adjusted score:

```text
stable_score = score - stability_weight * score_std
```

This keeps high-PnL candidates visible while favoring parameter sets that behave more consistently across seeds.

## Current Regimes

- `baseline_sweep.json`: lower-volatility baseline environment
- `baseline_volatility_aware_sweep.json`: lower-volatility baseline using volatility-aware quoting
- `baseline_inventory_risk_sweep.json`: lower-volatility baseline using inventory-risk quoting
- `high_volatility_sweep.json`: higher volatility, noisier fills, stronger adverse selection
- `volatility_aware_sweep.json`: high-volatility environment using volatility-aware quoting
- `high_volatility_inventory_risk_sweep.json`: high-volatility environment using inventory-risk quoting

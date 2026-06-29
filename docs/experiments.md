# Experiments

Experiments are configured with JSON files in `configs/`.

## Run

```bash
cargo run
cargo run -- configs/baseline_sweep.json
cargo run -- configs/baseline_volatility_aware_sweep.json
cargo run -- configs/baseline_inventory_risk_sweep.json
cargo run -- configs/baseline_regime_adaptive_sweep.json
cargo run -- configs/high_volatility_sweep.json
cargo run -- configs/volatility_aware_sweep.json
cargo run -- configs/high_volatility_inventory_risk_sweep.json
cargo run -- configs/high_volatility_regime_adaptive_sweep.json
cargo run -- configs/mixed_regime_volatility_aware_sweep.json
cargo run -- configs/mixed_regime_adaptive_sweep.json
cargo run -- configs/baseline_sweep.json configs/baseline_volatility_aware_sweep.json configs/baseline_inventory_risk_sweep.json configs/baseline_regime_adaptive_sweep.json configs/high_volatility_sweep.json configs/volatility_aware_sweep.json configs/high_volatility_inventory_risk_sweep.json configs/high_volatility_regime_adaptive_sweep.json configs/mixed_regime_volatility_aware_sweep.json configs/mixed_regime_adaptive_sweep.json
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

The best-ranked strategy for each config is replayed across the configured seeds and written as a per-step dataset:

```text
target/reports/<config_name>_best_steps.csv
```

The best result from each config is also written to:

```text
target/reports/regime_summary.csv
```

Multi-seed outputs include standard deviation fields such as `score_std`, `final_pnl_std`, and `max_drawdown_std`, plus average regime step counts and execution attribution for low, normal, and high volatility. Step datasets include quote state, inventory, PnL, fills, fees, and adverse selection for ML/calibration work.

Step datasets can be inspected with the Python research utility:

```bash
python3 research/analyze_steps.py target/reports/mixed_regime_adaptive_volatility_aware_best_steps.csv
python3 research/calibrate_fill_model.py
python3 research/compare_calibrations.py
python3 research/validate_fill_model.py
```

The calibration utility estimates empirical fill probability and fill intensity by regime, spread bucket, and volatility bucket. It writes a JSON report to `target/research/` for later model comparison or calibration work.

The comparison utility reads calibration reports and writes a compact cross-experiment CSV summary to `target/research/fill_calibration_comparison.csv`.

The validation utility reads that comparison CSV and writes a pass/warn model-behavior report to `target/research/fill_model_validation.txt`.

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
- `baseline_regime_adaptive_sweep.json`: lower-volatility baseline using regime-conditioned volatility-aware quoting
- `high_volatility_sweep.json`: higher volatility, noisier fills, stronger adverse selection
- `volatility_aware_sweep.json`: high-volatility environment using volatility-aware quoting
- `high_volatility_inventory_risk_sweep.json`: high-volatility environment using inventory-risk quoting
- `high_volatility_regime_adaptive_sweep.json`: high-volatility environment using regime-conditioned volatility-aware quoting
- `mixed_regime_volatility_aware_sweep.json`: scheduled low/normal/high-volatility environment using volatility-aware quoting
- `mixed_regime_adaptive_sweep.json`: scheduled low/normal/high-volatility environment using regime-conditioned volatility-aware quoting

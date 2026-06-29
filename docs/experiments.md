# Experiments

Experiments are configured with JSON files in `configs/`.

## Run

```bash
cargo run
cargo run -- replay data/events.csv
cargo run -- configs/mixed_regime_adaptive_sweep.json
cargo run -- configs/*.json
```

The default config is `configs/baseline_sweep.json`. Use `configs/mixed_regime_adaptive_sweep.json` as the main research run, and `configs/*.json` when you want the full comparison set. Replay CSVs require `timestamp_ms` and `mid_price`; `bid` and `ask` are optional.

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

Replay runs write a step dataset to:

```text
target/reports/<csv_stem>_replay_steps.csv
```

Multi-seed outputs include standard deviation fields such as `score_std`, `final_pnl_std`, and `max_drawdown_std`, plus average regime step counts and execution attribution for low, normal, and high volatility. Step datasets include quote state, inventory, PnL, fills, fees, and adverse selection for ML/calibration work.

Research utilities consume the generated step datasets:

```bash
python3 research/calibrate_fill_model.py
python3 research/compare_calibrations.py
python3 research/validate_fill_model.py
```

Use `research/analyze_steps.py <path>` when you want a detailed look at one exported step dataset.

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

## Configs

See [the config guide](../configs/README.md) for the current experiment set and notes on which configs are appropriate for regime-level conclusions.

# Configs

The configs are grouped by market environment. The main research run is the mixed-regime adaptive sweep because it exercises low, normal, and high volatility in one experiment.

## Core

- `baseline_sweep.json`: fixed-spread baseline in a mostly normal-volatility environment.
- `baseline_volatility_aware_sweep.json`: volatility-aware quoting in the baseline environment.
- `mixed_regime_fixed_spread_sweep.json`: fixed-spread baseline in the mixed-regime environment.
- `mixed_regime_inventory_risk_sweep.json`: inventory-risk heuristic in the mixed-regime environment.
- `mixed_regime_rule_based_controller_sweep.json`: simple controller that switches between fixed-spread and risk-managed quoting.
- `mixed_regime_avellaneda_stoikov_sweep.json`: finite-horizon Avellaneda-Stoikov baseline in the mixed-regime environment.
- `mixed_regime_adaptive_sweep.json`: main mixed-regime experiment with regime-conditioned quoting.

## Additional Sweeps

- `baseline_inventory_risk_sweep.json`: baseline environment with inventory-risk quoting.
- `baseline_regime_adaptive_sweep.json`: baseline environment with regime-conditioned quoting.
- `high_volatility_sweep.json`: high-volatility fixed-spread stress test.
- `volatility_aware_sweep.json`: high-volatility stress test with volatility-aware quoting.
- `high_volatility_inventory_risk_sweep.json`: high-volatility stress test with inventory-risk quoting.
- `high_volatility_regime_adaptive_sweep.json`: high-volatility stress test with regime-conditioned quoting.
- `mixed_regime_volatility_aware_sweep.json`: mixed-regime environment with one volatility-aware parameter set.

## Regime Coverage

- Baseline configs are useful for strategy comparisons in a calmer market, but they do not strongly test regime transitions.
- High-volatility configs are stress tests. They spend most of their time in high volatility, so low/normal regime statistics can be too sparse for strong conclusions.
- Mixed-regime configs are the right place to evaluate regime-adaptive behavior because they intentionally move through low, normal, and high volatility.

Run all configs with:

```bash
cargo run -- configs/*.json
```

Replay run specs live under `configs/runs/` and use the `run` command:

```bash
cargo run -- run configs/runs/sample_replay.json
cargo run -- run configs/runs/sample_replay_sweep.json
```

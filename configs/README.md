# Configs

The configs are grouped by market environment. The main research run is the mixed-regime adaptive sweep because it exercises low, normal, and high volatility in one experiment.

## Core

- `baseline_sweep.json`: fixed-spread baseline.
- `baseline_volatility_aware_sweep.json`: baseline environment with volatility-aware quoting.
- `mixed_regime_adaptive_sweep.json`: main mixed-regime experiment with regime-conditioned quoting.

## Additional Sweeps

- `baseline_inventory_risk_sweep.json`: baseline environment with inventory-risk quoting.
- `baseline_regime_adaptive_sweep.json`: baseline environment with regime-conditioned quoting.
- `high_volatility_sweep.json`: high-volatility fixed-spread baseline.
- `volatility_aware_sweep.json`: high-volatility environment with volatility-aware quoting.
- `high_volatility_inventory_risk_sweep.json`: high-volatility environment with inventory-risk quoting.
- `high_volatility_regime_adaptive_sweep.json`: high-volatility environment with regime-conditioned quoting.
- `mixed_regime_volatility_aware_sweep.json`: mixed-regime environment with one volatility-aware parameter set.

Run all configs with:

```bash
cargo run -- configs/*.json
```

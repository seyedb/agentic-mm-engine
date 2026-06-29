## Agentic Market Making
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Rust-based experimental market-making engine for studying inventory-aware quoting and, eventually, agent-driven strategy control.

### Status

This is a learning and research project, not a production trading system. The current focus is a clean simulation core with configurable parameter sweeps and reproducible experiment output.

The long-term goal is an agentic market maker: a system where a controller can observe market state, evaluate risk, and adapt strategy parameters over time.

### What It Does

- Simulates fixed-spread, volatility-aware, inventory-risk, and regime-adaptive market-making strategies.
- Models fills, fees, adverse selection, inventory, cash, and mark-to-market PnL.
- Runs configurable multi-seed parameter sweeps.
- Aggregates sweep results across multiple random seeds.
- Writes ranked sweep results and best-strategy step datasets to CSV.

### Run

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

Analyze an exported best-strategy step dataset:

```bash
python3 research/analyze_steps.py target/reports/mixed_regime_adaptive_volatility_aware_best_steps.csv
python3 research/calibrate_fill_model.py target/reports/mixed_regime_adaptive_volatility_aware_best_steps.csv
python3 research/compare_calibrations.py
```

### Verify

```bash
cargo test
cargo clippy -- -D warnings
```

### Docs

- [Model assumptions](docs/model.md)
- [Experiments and sweeps](docs/experiments.md)

### Roadmap

- Improve the fill model with arrival probabilities and volatility-aware behavior.
- Add multi-regime comparison reports.
- Add live public market data in paper-trading mode.
- Add an agent/control layer after the core simulator is stable.

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
cargo run -- replay data/sample_events.csv
cargo run -- configs/mixed_regime_adaptive_sweep.json
cargo run -- configs/*.json
```

The default config is `configs/baseline_sweep.json`.

Run the research checks after generating reports:

```bash
python3 research/calibrate_fill_model.py
python3 research/compare_calibrations.py
python3 research/validate_fill_model.py
```

### Verify

```bash
cargo test
cargo clippy -- -D warnings
```

### Docs

- [Model assumptions](docs/model.md)
- [Experiments and sweeps](docs/experiments.md)
- [Config guide](configs/README.md)

### Roadmap

- Improve the fill model using validation warnings.
- Add live public market data in paper-trading mode.
- Add an agent/control layer after the core simulator is stable.

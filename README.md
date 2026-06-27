## Agentic Market Making
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Rust-based experimental market-making engine for studying inventory-aware quoting and, eventually, agent-driven strategy control.

### Status

This is a learning and research project, not a production trading system. The current focus is a clean simulation core with configurable parameter sweeps and reproducible experiment output.

The long-term goal is an agentic market maker: a system where a controller can observe market state, evaluate risk, and adapt strategy parameters over time.

### What It Does

- Simulates a market-making strategy with inventory-aware quote skew.
- Models fills, fees, adverse selection, inventory, cash, and mark-to-market PnL.
- Runs configurable spread/skew parameter sweeps.
- Aggregates sweep results across multiple random seeds.
- Writes ranked sweep results to CSV.

### Run

```bash
cargo run
cargo run -- configs/baseline_sweep.json
cargo run -- configs/high_volatility_sweep.json
cargo run -- configs/baseline_sweep.json configs/high_volatility_sweep.json
```

The default config is `configs/baseline_sweep.json`.

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

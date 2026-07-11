## Agentic Market Making
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Rust-based experimental market-making engine for studying inventory-aware quoting and, eventually, agent-driven strategy control.

### Status

This is a learning and research project, not a production trading system. The current focus is a clean simulation core with configurable parameter sweeps and reproducible experiment output.

The long-term goal is an agentic market maker: a system where a controller can observe market state, evaluate risk, and adapt strategy parameters over time.

### What It Does

- Simulates fixed-spread, volatility-aware, inventory-risk, rule-based controller, regime-adaptive, and Avellaneda-Stoikov strategies.
- Models fills, fees, adverse selection, inventory, cash, and mark-to-market PnL.
- Runs configurable multi-seed parameter sweeps.
- Aggregates sweep results across multiple random seeds.
- Writes ranked sweep results and best-strategy step datasets to CSV.

### Run

```bash
cargo run
cargo run -- run configs/runs/sample_replay.json
cargo run -- run configs/runs/sample_replay_sweep.json
cargo run -- run configs/runs/sample_paper_session.json
cargo run -- replay data/sample_events.csv
cargo run -- replay-sweep data/sample_events.csv
cargo run -- configs/mixed_regime_adaptive_sweep.json
cargo run -- configs/*.json
python3 research/compare_strategy_sweeps.py
```

The default config is `configs/baseline_sweep.json`.

Fetch recent public candle data for replay:

```bash
python3 research/fetch_public_events.py --pair SOLUSD --bars 120 --out data/kraken_solusd_events.csv
```

Run configs live in `configs/runs/`. Set the `data` field to the CSV you want to replay, then run:

```bash
cargo run -- run configs/runs/sample_replay.json
cargo run -- run configs/runs/sample_replay_sweep.json
cargo run -- run configs/runs/sample_paper_session.json
```

Run the current paper-policy research loop after collecting quote datasets:

```bash
python3 research/policy_evaluation_gate.py
python3 research/sweep_selector_policy.py
python3 research/train_policy_selector.py
python3 research/write_project_report.py
```

### Verify

```bash
cargo test
cargo clippy -- -D warnings
```

### Docs

- [Model assumptions](docs/model.md)
- [Experiments and sweeps](docs/experiments.md)
- [Research workflow](docs/workflow.md)
- [Config guide](configs/README.md)

### Development Note

This project is developed as a learning and research effort with AI assistance for code generation, refactoring, and documentation. Design decisions, review, testing, and project direction are handled by the author.

### Roadmap

- Improve the fill model using validation warnings.
- Add live public market data in paper-trading mode.
- Add an agent/control layer after the core simulator is stable.

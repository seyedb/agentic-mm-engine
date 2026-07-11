## Agentic Market Making
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Rust-based experimental market-making research engine for studying inventory-aware quoting, public quote replay, and learned policy selection.

### Status

This is a learning and research project, not a production trading system. The current focus is a clean Rust paper engine with a reproducible Python research loop.

The project now includes an agentic proof of concept: Python trains a small policy-selection gate from paper-session results, exports a JSON model, and Rust executes that learned policy during replay.

### What It Does

- Replays public top-of-book quote data through Rust paper-session policies.
- Compares static, adaptive, hybrid, selector, and learned-selector policies.
- Models fills, fees, inventory, cash, mark-to-market PnL, and drawdown.
- Trains a small Python learned gate and exports it back to Rust.
- Evaluates policy robustness across multiple fill assumptions.

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
python3 research/train_policy_selector.py
python3 research/policy_evaluation_gate.py
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
- [Research protocol](docs/research_protocol.md)
- [Research workflow](docs/workflow.md)
- [Config guide](configs/README.md)

### Development Note

This project is developed as a learning and research effort with AI assistance for code generation, refactoring, and documentation. Design decisions, review, testing, and project direction are handled by the author.

### Roadmap

- Consolidate the final research protocol and result bundle.
- Run a live public-data paper demonstration with the learned selector.
- Keep improving the learned policy only when new data exposes a concrete weakness.

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

### Quickstart

```bash
python3 research/policy_evaluation_gate.py
python3 research/train_policy_selector.py
python3 research/policy_evaluation_gate.py
python3 research/write_project_report.py
```

Run the learned selector against live public quotes in paper mode:

```bash
python3 research/run_paper_live_report.py configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live.json \
  --run-id solusd_demo
```

### Current Result

Latest seven-dataset research report:

- `learned_selector` configured utility: `0.000611`
- `selector` configured utility: `0.000521`
- `adaptive` configured utility: `0.000177`
- `learned_selector` remains behind `adaptive` under the liquid-fill assumption.

This is a small-sample research result, not evidence of a live trading edge.

### Other Runs

```bash
cargo run -- run configs/runs/sample_replay.json
cargo run -- run configs/runs/sample_replay_sweep.json
cargo run -- run configs/runs/sample_paper_session.json
python3 research/collect_quote_dataset.py --pair SOLUSD --samples 120 --interval-seconds 2 --evaluate
```

### Verify

```bash
cargo test
cargo clippy -- -D warnings
```

### Docs

- [Model assumptions](docs/model.md)
- [Experiments and sweeps](docs/experiments.md)
- [Live paper demo](docs/live_demo.md)
- [Research protocol](docs/research_protocol.md)
- [Research workflow](docs/workflow.md)
- [Config guide](configs/README.md)

### Development Note

This project is developed as a learning and research effort with AI assistance for code generation, refactoring, and documentation. Design decisions, review, testing, and project direction are handled by the author.

### Roadmap

- Preserve the final research result bundle.
- Compare one longer live-paper learned-selector run against the short demo.
- Keep improving the learned policy only when new data exposes a concrete weakness.

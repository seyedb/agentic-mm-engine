## Agentic Market Making
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Rust-based experimental market-making research engine for studying inventory-aware quoting, public quote replay, and learned policy selection.

### Status

This is a learning and research project, not a production trading system. The current focus is a clean Rust paper engine with a reproducible Python research loop.

The project now includes agentic proof-of-concept policies: Python trains small policy models from paper-session results, exports JSON models, and Rust executes those learned decisions during replay.

### What It Does

- Replays public top-of-book quote data through Rust paper-session policies.
- Compares static, adaptive, hybrid, selector, and learned-selector policies.
- Models fills, fees, inventory, cash, mark-to-market PnL, and drawdown.
- Trains a small Python logistic-regression gate and exports it back to Rust.
- Trains a multi-action linear policy agent and exports it back to Rust.
- Trains an executable LinUCB contextual-bandit agent and exports it back to Rust.
- Evaluates policy robustness across multiple fill assumptions.
- Includes small checked-in public quote datasets for reproducing the current report.

### Quickstart

```bash
python3 research/run_research_pipeline.py
```

For quick report regeneration from existing policy-gate outputs:

```bash
python3 research/run_research_pipeline.py --skip-policy-gates
```

The checked-in SOLUSD quote captures are enough to reproduce the current offline report. Collecting fresh public data is optional.

Run the learned selector against live public quotes in paper mode:

```bash
python3 research/run_paper_live_report.py configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live_long.json \
  --run-id solusd_demo
```

### Current Result

Latest seven-dataset research report:

- Public quote samples: `1440`
- Replay windows: `48`
- `learned_selector` configured utility: `0.000572`
- `selector` configured utility: `0.000521`
- `adaptive` configured utility: `0.000177`
- `linear_agent` liquid-fill utility: `0.004511`
- `linear_agent` is functional but mixed: it wins liquid-fill sensitivity and lags configured/conservative assumptions.
- `bandit_agent` is executable now, but remains research-only: configured utility `-0.003747`.

This is a small-sample research result, not evidence of a live trading edge.

Latest live-paper demonstration: `300` public quote samples, `70` paper fills, final inventory `-0.2000`, and max drawdown `0.0059`.

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

- [Final research report](docs/final_report.md)
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
- Improve the contextual-bandit agent only if more data shows it can beat simpler selectors.
- Keep improving the learned policy only when new data exposes a concrete weakness.

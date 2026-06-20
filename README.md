## Agentic Market Making
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Rust-based experimental market-making engine for studying inventory-aware quoting and, eventually, agent-driven strategy control.

### Overview

This project is a learning and research environment for market-making ideas. It is intentionally not a production trading system. The current engine simulates a mid-price process, generates bid/ask quotes from an inventory-aware strategy, simulates fills, and tracks cash, inventory, and mark-to-market PnL.

The long-term goal is an agentic market maker: a system where a controller can observe simulation state and adapt strategy parameters over time. The current focus is building a clean, testable simulation core before adding agent or API layers.

### Architecture

```text
src/
  main.rs              # small runnable demo
  lib.rs               # reusable library entry point
  engine/
    simulation.rs      # deterministic simulation loop
    state.rs           # accounting state and PnL updates
  market/
    mod.rs             # quotes, fills, and market-side types
  strategy/
    market_maker.rs    # inventory-skew market-making strategy
    mod.rs             # strategy trait
```

### Current Model

- Price follows a seeded random walk.
- The strategy quotes around mid-price with a fixed spread.
- Inventory skews quotes lower when inventory is positive and higher when inventory is negative.
- Fills occur when a noisy simulated market price crosses the bid or ask.
- PnL is marked to market as `cash + inventory * mid_price`.

### Run

```bash
cargo run
```

### Test

```bash
cargo test
```

### Roadmap

- Add explicit experiment configuration.
- Improve the fill model with arrival probabilities and volatility-aware behavior.
- Track richer performance metrics such as drawdown, fill count, turnover, and inventory risk.
- Add parameter sweeps for comparing strategies.
- Add an agent/control layer after the core simulator is stable.

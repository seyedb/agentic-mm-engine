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
  experiment.rs        # named experiment configs and reports
  lib.rs               # reusable library entry point
  sweep.rs             # grid search over strategy parameters
  engine/
    metrics.rs         # simulation summary statistics
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
- Each fill pays a configurable notional fee.
- Filled quotes move the mid-price slightly against the market maker to model adverse selection.
- PnL is marked to market as `cash + inventory * mid_price`.
- Simulation metrics summarize fills, turnover, inventory exposure, and drawdown.
- Named experiments compare strategy settings under the same simulation conditions.
- Parameter sweeps rank spread/skew combinations with a simple risk-adjusted score.
- Sweep runs write CSV results to `target/reports/sweep_results.csv`.

### Run

```bash
cargo run
```

### Test

```bash
cargo test
```

### Sweep Score

Parameter sweeps currently use a simple placeholder objective:

```text
score = final_pnl
      - 2.0 * max_drawdown
      - max_abs_inventory
      - inactivity_penalty
```

This rewards PnL while penalizing drawdown, inventory exposure, and strategies that do not trade enough to be useful.

### Roadmap

- Add explicit experiment configuration.
- Improve the fill model with arrival probabilities and volatility-aware behavior.
- Add CSV or JSON output for experiment results.
- Add an agent/control layer after the core simulator is stable.

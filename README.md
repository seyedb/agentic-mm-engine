## Agentic Market Making
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
A Rust-based market-making engine simulating inventory-aware quoting with a future extension toward agent-driven strategy control.

### Overview
This project is an experimental market-making simulator built in Rust, designed to explore adaptive trading strategies and agent-based control of execution parameters.
The system is intentionally simplified and serves as a research and learning environment rather than a production trading system.

### Notes
- This project is experimental in nature.
- AI-assisted tools were used during development for design exploration and code scaffolding.
- Core system design, architecture decisions, and implementation logic are owned and reviewed by the author.

### Components
- Rust execution engine
- Simulated price process
- Market making logic
- PnL + inventory tracking

### Architecture
The system is structured into:
- Execution engine (deterministic simulation)
- Strategy layer (parameterized decision logic)
- Future agent layer (external control of strategy parameters)

## Future work
- Python agent interface
- ML-based parameter control
- Real market data integration


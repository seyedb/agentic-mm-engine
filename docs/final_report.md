# Agentic Market Making: Final Research Report

## Scope

This is an experimental market-making research project, not a production trading system. The goal is to show a clean, reproducible loop from public quote data to Rust paper execution and a small learned policy controller.

## System

- Rust runs replay, paper sessions, live public-data paper mode, fills, fees, inventory, and PnL accounting.
- Python runs data collection, policy evaluation, learned-gate training, Plotly reporting, and summary reports.
- Policies include static, adaptive, hybrid, selector, and learned-selector variants.
- The learned selector is trained in Python, exported as JSON, and loaded back into Rust for paper execution.

## Research Result

- Datasets: `7`.
- Windows: `48`.
- Fill assumptions: `configured`, `conservative_fill`, and `liquid_fill`.
- Best configured policy: `learned_selector` with utility `0.000611`.
- Best conservative-fill policy: `selector` with utility `-0.000636`.
- Best liquid-fill policy: `adaptive` with utility `0.004290`.

## Configured Evaluation

- Selector utility: `0.000521`.
- Adaptive baseline utility: `0.000177`.
- Selector minus adaptive: `0.000344`.
- Selector adaptive-step rate: `78.055556%`.
- Rust learned-selector utility: `0.000611`.
- Rust learned minus adaptive: `0.000434`.
- Rust learned minus selector: `0.000090`.
- Rust learned-selector adaptive-step rate: `82.291667%`.

## Fill-Assumption Check

- Conservative-fill winner: `selector`.
- Learned-selector conservative-fill utility: `-0.000766`.
- Liquid-fill winner: `adaptive`.
- Learned-selector liquid-fill utility: `0.004172`.
- No policy wins all assumptions, so the result should be read as a research signal rather than a robust trading claim.

## Learned Gate

- Learned gate holdout utility: `0.000451`.
- Learned minus adaptive: `0.000274`.
- Learned minus selector: `-0.000070`.
- Best rule-selector sweep variant: `selector_thr_0p08_vol_6_spr_5_inv_0p4_dd_10`.
- Best sweep score: `0.003067`.

## Live Paper Demo

- Run ID: `solusd_long_20260711_001`.
- Pair: `SOLUSD`.
- Public quote samples: `300`.
- Paper fills: `70`.
- Final paper PnL: `0.080736`.
- Inventory range: `-0.300000` to `0.400000`.
- Max drawdown: `0.005944`.
- Average quote distance: `0.019005`.
- Adaptive steps: `282`.
- Dominant trigger: `configured`.

## Interpretation

The project has reached a credible proof-of-concept state. The agentic loop is real: public data feeds Rust paper sessions, Python trains a small policy gate, the model is exported to JSON, and Rust executes that learned selector in replay and live public-data paper mode.

The result is useful because it is measurable and falsifiable, not because it proves a trading edge. The learned selector leads under the configured evaluation, remains close under other assumptions, and produces a coherent live-paper run with bounded inventory. The weakest point remains fill realism.

## Limitations

- Public top-of-book snapshots are limited data.
- Fill behavior is modeled, not exchange-verified.
- The learned gate is small and trained on a limited number of quote windows.
- Live paper mode polls public quotes and never places orders.

## Wrap-Up Assessment

This is a reasonable place to wrap the current phase. The next genuinely interesting phase would be a small contextual-bandit or reinforcement-style selector, but that should be treated as a separate research extension after preserving this result.

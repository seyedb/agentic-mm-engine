# Agentic Market Making: Final Research Report

## Scope

This is an experimental market-making research project, not a production trading system. The goal is to show a clean, reproducible loop from public quote data to Rust paper execution and small learned policy controllers.

## System

- Rust runs replay, paper sessions, live public-data paper mode, fills, fees, inventory, and PnL accounting.
- Python runs data collection, policy evaluation, learned-gate training, Plotly reporting, and summary reports.
- Policies include static, adaptive, hybrid, selector, learned-selector, linear-agent, and bandit-agent variants.
- The learned selector is a logistic-regression classifier trained in Python, exported as JSON, and loaded back into Rust for paper execution.
- The linear agent is a multi-action ridge-regression utility model trained in Python and executed by Rust.
- The bandit agent is a LinUCB contextual-bandit policy trained in Python and executed by Rust.

## Research Result

- Datasets: `7`.
- Windows: `48`.
- Fill assumptions: `configured`, `conservative_fill`, and `liquid_fill`.
- Best configured policy: `learned_selector` with utility `0.000572`.
- Best conservative-fill policy: `selector` with utility `-0.000636`.
- Best liquid-fill policy: `linear_agent` with utility `0.004511`.

## Configured Evaluation

- Selector utility: `0.000521`.
- Adaptive baseline utility: `0.000177`.
- Selector minus adaptive: `0.000344`.
- Selector adaptive-step rate: `78.055556%`.
- Rust learned-selector utility: `0.000572`.
- Rust learned minus adaptive: `0.000395`.
- Rust learned minus selector: `0.000051`.
- Rust learned-selector adaptive-step rate: `80.902778%`.
- Rust linear-agent utility: `-0.000961`.
- Rust linear-agent minus adaptive: `-0.001138`.
- Rust linear-agent minus selector: `-0.001482`.
- Rust linear-agent adaptive-step rate: `58.402778%`.
- Rust bandit-agent utility: `-0.003747`.
- Rust bandit-agent minus adaptive: `-0.003924`.
- Rust bandit-agent minus selector: `-0.004268`.
- Rust bandit-agent adaptive-step rate: `40.138889%`.

## Fill-Assumption Check

- Conservative-fill winner: `selector`.
- Learned-selector conservative-fill utility: `-0.000865`.
- Liquid-fill winner: `linear_agent`.
- Learned-selector liquid-fill utility: `0.004082`.
- No policy wins all assumptions, so the result should be read as a research signal rather than a robust trading claim.

## Learned Gate

- Learned gate holdout utility: `0.000296`.
- Learned minus adaptive: `0.000118`.
- Learned minus selector: `-0.000226`.
- Best rule-selector sweep variant: `selector_thr_0p08_vol_6_spr_5_inv_0p4_dd_10`.
- Best sweep score: `0.003067`.

## Contextual Bandit Check

- Chronological executable LinUCB utility: `-0.000844`.
- Leave-one-dataset-out LinUCB utility: `0.001074`.
- Fold always-`static` utility: `-0.004370`.
- Fold always-`adaptive` utility: `0.000178`.
- Fold always-`selector` utility: `0.000522`.
- Rust policy-gate bandit utility: `-0.003747`.
- The bandit is executable and useful as an ML-agent proof of concept, but it is not the best policy in the current gate.

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

The project has reached a credible proof-of-concept state. The agentic loop is real: public data feeds Rust paper sessions, Python trains small logistic, linear, and contextual-bandit policy models, the models are exported to JSON, and Rust executes those learned decisions in replay.

The result is useful because it is measurable and falsifiable, not because it proves a trading edge. The learned selector leads under the configured evaluation, while the linear and bandit agents are functional multi-action controllers with mixed results. The weakest point remains fill realism.

## Limitations

- Public top-of-book snapshots are limited data.
- Fill behavior is modeled, not exchange-verified.
- The learned models are small and trained on a limited number of quote windows.
- Live paper mode polls public quotes and never places orders.

## Wrap-Up Assessment

This is a reasonable place to wrap the current phase. The next genuinely interesting phase would be more live public-data evaluation and a larger dataset before trying to improve the ML agents.

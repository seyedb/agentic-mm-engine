# Research Harness

Python is used for data collection, model training, evaluation gates, and reports. Rust remains the execution engine for replay, paper sessions, fills, fees, inventory, and PnL accounting. The active scripts use the Python standard library.

Current entrypoints:

- `run_research_pipeline.py`: reproducible offline report pipeline.
- `collect_quote_dataset.py`: collect public top-of-book quote snapshots.
- `policy_evaluation_gate.py`: evaluate Rust paper policies across datasets and fill assumptions.
- `train_policy_selector.py`: train the logistic learned selector.
- `train_linear_policy_agent.py`: train the linear multi-action policy agent.
- `train_contextual_bandit_agent.py`: train the executable LinUCB policy agent.
- `run_paper_live_report.py`: run public-data paper mode and write a Plotly report.
- `analyze_paper_session.py` and `plot_paper_session.py`: inspect a single Rust paper-session CSV.
- `write_project_report.py`: regenerate the concise project report.

`archive/` contains older diagnostics, tuning scripts, and one-off comparison helpers kept for reference.

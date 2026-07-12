# Research Workflow

This is the current workflow for the project. It favors a small number of repeatable commands over keeping every historical experiment around.

## Collect Public Quote Data

For market-making research, prefer top-of-book quote snapshots over candle closes:

```bash
python3 research/collect_quote_dataset.py \
  --pair SOLUSD \
  --samples 360 \
  --interval-seconds 5 \
  --evaluate \
  --window-size 60 \
  --step-size 60
```

This writes a quote CSV and metadata file under `data/quotes/`. With `--evaluate`, it also writes dataset-specific paper-policy evaluations under `target/research/quote_datasets/`.

The repository includes a small checked-in SOLUSD quote dataset bundle for reproducing the current report. Treat new captures as research inputs: review them before committing and keep the checked-in set small.

Use the lighter fetchers when you only need raw replay inputs:

```bash
python3 research/fetch_public_quotes.py --pair SOLUSD --samples 60 --out data/kraken_solusd_quotes.csv
python3 research/fetch_public_events.py --pair SOLUSD --bars 120 --out data/kraken_solusd_events.csv
```

## Run Paper Sessions

Paper sessions replay public data through the Rust engine and record quotes, fills, inventory, PnL, drawdown, policy mode, and policy trigger:

```bash
cargo run -- run configs/runs/kraken_solusd_learned_selector_maker_fee_paper_session.json
```

Analyze or visualize one session:

```bash
python3 research/analyze_paper_session.py target/reports/kraken_solusd_learned_selector_maker_fee_paper_session.csv
python3 research/plot_paper_session.py target/reports/kraken_solusd_learned_selector_maker_fee_paper_session.csv
```

## Evaluate Policies

Run the full offline research pipeline:

```bash
python3 research/run_research_pipeline.py
```

For quick report regeneration from existing policy-gate outputs:

```bash
python3 research/run_research_pipeline.py --skip-policy-gates
```

The main evaluation gate compares static, adaptive, hybrid, selector, and, when a learned model exists, learned-selector policies across collected quote datasets and multiple fill assumptions:

```bash
python3 research/policy_evaluation_gate.py
```

Outputs:

```text
target/research/policy_gate_dataset_summary.csv
target/research/policy_gate_window_results.csv
target/research/policy_gate_policy_summary.csv
target/research/policy_gate_report.md
```

Write a compact live-dataset manifest and policy replay summary from the latest gate outputs:

```bash
python3 research/summarize_live_dataset_evaluation.py
```

Outputs:

```text
target/research/live_dataset_manifest.csv
target/research/live_dataset_evaluation.md
```

Sweep selector weights when tuning the rule-based agentic controller:

```bash
python3 research/sweep_selector_policy.py
```

Train the first learned policy gate after the policy gate has produced window results:

```bash
python3 research/train_policy_selector.py
```

Train the multi-action linear policy agent:

```bash
python3 research/train_linear_policy_agent.py
```

Train the executable contextual-bandit policy agent:

```bash
python3 research/train_contextual_bandit_agent.py
```

Run the offline contextual-bandit research diagnostic:

```bash
python3 research/train_bandit_selector.py
```

Then rerun the gate so Rust loads the Python-trained model and evaluates it as a normal paper policy:

```bash
python3 research/policy_evaluation_gate.py
```

Write the concise project status report:

```bash
python3 research/write_project_report.py
```

This writes [final_report.md](final_report.md).

The evaluation standard is described in [research_protocol.md](research_protocol.md).

## Live Paper Mode

Live paper mode polls public Kraken top-of-book data and never places orders:

```bash
cargo run -- run configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live.json
```

Run a live paper config and produce analysis plus a Plotly report in one step:

```bash
python3 research/run_paper_live_report.py configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live.json \
  --run-id solusd_001
```

Use the longer learned-selector config when checking whether the live-paper behavior is stable beyond a short smoke test:

```bash
python3 research/run_paper_live_report.py configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live_long.json \
  --run-id solusd_long_001
python3 research/compare_paper_live_runs.py
```

See [live_demo.md](live_demo.md) for the latest learned-selector live-paper demonstration.

Optional live-run comparison/calibration helpers:

```bash
python3 research/compare_paper_live_runs.py
python3 research/calibrate_paper_fill_model.py target/reports/paper_live/solusd_001.csv
python3 research/propose_paper_fill_config.py target/research/solusd_001_paper_fill_calibration.csv
python3 research/compare_paper_fill_calibrations.py
python3 research/summarize_paper_live_research.py
```

## Replay And Strategy Baselines

Replay sweeps are still useful for checking the simulator and comparing baseline strategy families:

```bash
cargo run -- run configs/runs/sample_replay_sweep.json
python3 research/run_replay_sweeps.py data/kraken_solusd_*.csv
python3 research/analyze_replay_sweep.py target/reports/kraken_solusd_events_replay_sweep.csv
python3 research/compare_replay_sweeps.py target/reports/*_replay_sweep.csv
python3 research/compare_strategy_sweeps.py
```

# Research Workflow

This is the current public-data replay workflow. It keeps data collection, replay, sweep analysis, and cross-dataset comparison explicit.

## Fetch Public Data

```bash
python3 research/fetch_public_events.py \
  --pair SOLUSD \
  --bars 120 \
  --out data/kraken_solusd_events.csv
```

The fetch script writes replay events as `timestamp_ms,mid_price`. Candle closes are used as a mid-price proxy.

Fetch public top-of-book snapshots when quote-crossing replay behavior is needed:

```bash
python3 research/fetch_public_quotes.py \
  --pair SOLUSD \
  --samples 60 \
  --interval-seconds 5 \
  --out data/kraken_solusd_quotes.csv
```

Use `--since` to create named replay windows:

```bash
python3 research/fetch_public_events.py \
  --pair SOLUSD \
  --interval 1 \
  --bars 120 \
  --since 2026-07-01T12:00:00Z \
  --out data/kraken_solusd_20260701_1200.csv
```

Fetch several windows when comparing replay robustness:

```bash
python3 research/fetch_replay_windows.py \
  --pair SOLUSD \
  --interval 1 \
  --bars 120 \
  --start 2026-07-01T12:00:00Z \
  --windows 3 \
  --step-minutes 120 \
  --out-dir data
```

Kraken OHLC responses are subject to the exchange's public history limits.

## Run A Replay Sweep

Prefer a committed run config for reproducible replay work:

```bash
cargo run -- run configs/runs/sample_replay_sweep.json
```

Run configs keep long parameter lists out of the command line:

```json
{
  "type": "replay_sweep",
  "data": "data/sample_events.csv",
  "seeds": [42, 43, 44, 45, 46],
  "spreads": [0.2, 0.5, 1.0],
  "skews": [0.0, 0.02, 0.05],
  "quantities": [0.05, 0.1, 0.2],
  "fee_rate": 0.001
}
```

The flag-based command is still available for quick experiments:

```bash
cargo run -- replay-sweep data/kraken_solusd_events.csv \
  --seeds 42,43,44,45,46 \
  --spreads 0.2,0.5,1.0 \
  --skews 0.0,0.02,0.05 \
  --quantities 0.05,0.1,0.2 \
  --fee-rate 0.001
```

Replay sweep output:

```text
target/reports/kraken_solusd_events_replay_sweep.csv
```

Run the same sweep over several replay CSVs:

```bash
python3 research/run_replay_sweeps.py data/kraken_solusd_*.csv \
  --seeds 42,43,44,45,46 \
  --spreads 0.2,0.5,1.0 \
  --skews 0.0,0.02,0.05 \
  --quantities 0.05,0.1,0.2 \
  --fee-rate 0.001
```

## Analyze One Replay Sweep

```bash
python3 research/analyze_replay_sweep.py target/reports/kraken_solusd_events_replay_sweep.csv
```

This summarizes the best row and average behavior by spread, quantity, and skew.

## Compare Replay Sweeps

```bash
python3 research/compare_replay_sweeps.py target/reports/*_replay_sweep.csv
```

Comparison outputs:

```text
target/research/replay_sweep_best.csv
target/research/replay_sweep_parameters.csv
```

Use this step to check whether a parameter set is robust across datasets instead of only strong on one replay.

## Compare Strategy Families

Run the shared mixed-regime strategy comparison:

```bash
python3 research/compare_strategy_sweeps.py
```

This runs the mixed-regime fixed-spread, volatility-aware, inventory-risk, rule-based controller, regime-adaptive, and Avellaneda-Stoikov sweeps, then writes:

```text
target/research/strategy_comparison.csv
```

Use `--skip-run` to compare existing report CSVs without rerunning the Rust sweeps.

## Run A Paper Session

Paper sessions replay public market events through a controller and write first-class decision logs:

```bash
cargo run -- run configs/runs/kraken_solusd_paper_session.json
```

The output CSV records the observed market state, controller mode, quote, fills, inventory, PnL, and drawdown at each step.
Use quote CSVs when you want paper fills from public top-of-book data; mid-only replay files can still record decisions, but they cannot produce observed quote fills.

Analyze the session log:

```bash
python3 research/analyze_paper_session.py target/reports/kraken_solusd_paper_session.csv
```

Render an interactive Plotly view of the session:

```bash
python3 research/plot_paper_session.py target/reports/kraken_solusd_paper_session.csv
```

The report is written to:

```text
target/research/kraken_solusd_paper_session.html
```

## Run A Live Paper Session

Live paper sessions poll public Kraken top-of-book data and append one decision row per sample:

```bash
cargo run -- run configs/runs/kraken_solusd_paper_live.json
```

This does not place orders. It only records public quote snapshots, agent quotes, paper fills, inventory, PnL, fees, and drawdown.

Run the live session and produce the analysis plus Plotly report in one step:

```bash
python3 research/run_paper_live_report.py configs/runs/kraken_solusd_paper_live.json \
  --run-id solusd_20260708_001
```

This writes a live CSV, sidecar metadata file, and Plotly HTML report without overwriting earlier runs:

```text
target/reports/paper_live/solusd_20260708_001.csv
target/reports/paper_live/solusd_20260708_001.meta.json
target/research/solusd_20260708_001.html
```

Use `--skip-run` to regenerate the analysis, metadata, and Plotly report from an existing live CSV.

Calibrate the touch-intensity paper fill model against a logged paper session:

```bash
python3 research/calibrate_paper_fill_model.py target/reports/kraken_solusd_paper_live.csv
```

The calibration report ranks parameter grids by likelihood of the logged buy/sell fill labels.

Print the best calibrated `fill_model` block without editing the run config:

```bash
python3 research/propose_paper_fill_config.py target/research/kraken_solusd_paper_live_paper_fill_calibration.csv
```

Compare preserved live paper runs:

```bash
python3 research/compare_paper_live_runs.py
```

The comparison is written to:

```text
target/research/paper_live_runs.csv
```

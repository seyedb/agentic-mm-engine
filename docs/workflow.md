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

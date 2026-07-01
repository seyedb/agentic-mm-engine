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

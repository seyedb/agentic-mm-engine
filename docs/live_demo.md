# Live Paper Demo

This project can run the learned selector on live public top-of-book quotes without placing orders.

## Demo Command

```bash
python3 research/run_paper_live_report.py configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live.json \
  --run-id solusd_learned_demo_20260711_001
```

The command runs the Rust `paper_live` engine, writes a CSV decision log, analyzes the run, and renders a Plotly HTML report.

## Latest Demo

- Run ID: `solusd_learned_demo_20260711_001`
- Pair: `SOLUSD`
- Samples: `60`
- Interval: `2s`
- Policy: `learned_selector`
- Fills: `16`
- Final inventory: `0.0000`
- Final paper PnL: `0.0091`
- Max drawdown: `0.0020`
- Policy mode split: `55` adaptive steps, `5` static steps
- Main triggers: `spread` and `configured`

Generated artifacts:

```text
target/reports/paper_live/solusd_learned_demo_20260711_001.csv
target/reports/paper_live/solusd_learned_demo_20260711_001.meta.json
target/research/solusd_learned_demo_20260711_001.html
```

## Interpretation

This is a live public-data paper demonstration, not a trading result. It shows that the learned policy trained by Python can be exported to JSON, loaded by Rust, and executed against live public quote snapshots end to end.

The short run kept inventory bounded and produced paper fills under the configured fill model. The result should be treated as an operational demo, not evidence of a live edge.

# Configs

The config set is intentionally small. Top-level configs are simulator sweeps; `configs/runs/` contains concrete replay, paper-session, and live-paper run specs.

## Simulator Sweeps

- `baseline_sweep.json`: fixed-spread baseline in a mostly normal-volatility environment.
- `mixed_regime_fixed_spread_sweep.json`: fixed-spread baseline in the mixed-regime environment.
- `mixed_regime_volatility_aware_sweep.json`: volatility-aware baseline in the mixed-regime environment.
- `mixed_regime_inventory_risk_sweep.json`: inventory-risk heuristic in the mixed-regime environment.
- `mixed_regime_rule_based_controller_sweep.json`: simple controller that switches between fixed-spread and risk-managed quoting.
- `mixed_regime_avellaneda_stoikov_sweep.json`: finite-horizon Avellaneda-Stoikov baseline.
- `mixed_regime_adaptive_sweep.json`: main mixed-regime experiment with regime-conditioned quoting.

Run all simulator sweeps with:

```bash
cargo run -- configs/*.json
```

## Run Specs

Sample configs:

```bash
cargo run -- run configs/runs/sample_replay.json
cargo run -- run configs/runs/sample_replay_sweep.json
cargo run -- run configs/runs/sample_paper_session.json
```

Paper-policy configs used by the research loop:

- `configs/runs/kraken_solusd_maker_fee_paper_session.json`
- `configs/runs/kraken_solusd_adaptive_maker_fee_paper_session.json`
- `configs/runs/kraken_solusd_hybrid_maker_fee_paper_session.json`
- `configs/runs/kraken_solusd_selector_maker_fee_paper_session.json`
- `configs/runs/kraken_solusd_learned_selector_maker_fee_paper_session.json`
- `configs/runs/kraken_solusd_linear_agent_maker_fee_paper_session.json`

Live paper configs:

```bash
cargo run -- run configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live.json
cargo run -- run configs/runs/kraken_solusd_learned_selector_maker_fee_paper_live_long.json
```

Use the `_long` config for the documented live-paper demonstration.

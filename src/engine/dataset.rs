use std::fmt::Write;

use crate::engine::simulation::{MarketRegime, SimulationResult};
use crate::market::FillSide;

const STEP_DATASET_HEADER: &str = "experiment,strategy_type,seed,step,mid_price,estimated_volatility,regime,bid,ask,spread,observed_bid,observed_ask,bid_distance_to_observed_ask,ask_distance_to_observed_bid,inventory,cash,pnl,total_fills,buy_fills,sell_fills,fill_quantity,fill_notional,fees,adverse_selection\n";

pub fn step_dataset_header() -> &'static str {
    STEP_DATASET_HEADER
}

pub fn append_step_dataset_rows(
    csv: &mut String,
    experiment: &str,
    strategy_type: &str,
    seed: u64,
    result: &SimulationResult,
) {
    for (step_index, step) in result.steps.iter().enumerate() {
        let mut buy_fills = 0;
        let mut sell_fills = 0;
        let mut fill_quantity = 0.0;
        let mut fill_notional = 0.0;
        let mut fees = 0.0;

        for fill in &step.fills {
            fill_quantity += fill.quantity;
            fill_notional += fill.notional();
            fees += fill.fee;

            match fill.side {
                FillSide::Buy => buy_fills += 1,
                FillSide::Sell => sell_fills += 1,
            }
        }

        let row = vec![
            experiment.to_string(),
            strategy_type.to_string(),
            seed.to_string(),
            step_index.to_string(),
            format_f64(step.mid_price),
            format_f64(step.estimated_volatility),
            regime_name(step.regime).to_string(),
            format_f64(step.quote.bid),
            format_f64(step.quote.ask),
            format_f64(step.quote.spread()),
            optional_f64(step.observed_quote.map(|quote| quote.bid)),
            optional_f64(step.observed_quote.map(|quote| quote.ask)),
            optional_f64(step.observed_quote.map(|quote| quote.ask - step.quote.bid)),
            optional_f64(step.observed_quote.map(|quote| step.quote.ask - quote.bid)),
            format_f64(step.inventory),
            format_f64(step.cash),
            format_f64(step.pnl),
            step.fills.len().to_string(),
            buy_fills.to_string(),
            sell_fills.to_string(),
            format_f64(fill_quantity),
            format_f64(fill_notional),
            format_f64(fees),
            format_f64(step.adverse_selection_move),
        ];

        writeln!(csv, "{}", row.join(",")).expect("writing to a String should not fail");
    }
}

fn regime_name(regime: MarketRegime) -> &'static str {
    match regime {
        MarketRegime::LowVol => "LowVol",
        MarketRegime::NormalVol => "NormalVol",
        MarketRegime::HighVol => "HighVol",
    }
}

fn format_f64(value: f64) -> String {
    format!("{value:.6}")
}

fn optional_f64(value: Option<f64>) -> String {
    value.map(format_f64).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulation::{SimulationConfig, run_simulation};
    use crate::strategy::market_maker::StrategyParams;

    #[test]
    fn step_dataset_exports_one_row_per_step() {
        let strategy = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };
        let result = run_simulation(
            SimulationConfig {
                steps: 3,
                ..SimulationConfig::default()
            },
            &strategy,
        );
        let mut csv = String::from(step_dataset_header());

        append_step_dataset_rows(&mut csv, "test", "fixed_spread", 42, &result);

        assert!(csv.starts_with("experiment,strategy_type,seed,step"));
        assert!(csv.contains("test,fixed_spread,42,0"));
        assert_eq!(csv.lines().count(), 4);
    }

    #[test]
    fn step_dataset_exports_observed_quote_distances() {
        let result = SimulationResult {
            steps: vec![crate::engine::simulation::SimulationStep {
                mid_price: 100.0,
                estimated_volatility: 0.0,
                regime: MarketRegime::NormalVol,
                quote: crate::market::Quote {
                    bid: 99.5,
                    ask: 100.5,
                },
                observed_quote: Some(crate::market::Quote {
                    bid: 99.8,
                    ask: 100.2,
                }),
                fills: Vec::new(),
                adverse_selection_move: 0.0,
                inventory: 0.0,
                cash: 0.0,
                pnl: 0.0,
            }],
        };
        let mut csv = String::from(step_dataset_header());

        append_step_dataset_rows(&mut csv, "test", "fixed_spread", 42, &result);

        assert!(csv.contains(",99.800000,100.200000,0.700000,0.700000,"));
    }
}

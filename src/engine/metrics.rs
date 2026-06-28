use serde::{Deserialize, Serialize};

use crate::engine::simulation::{MarketRegime, SimulationResult};
use crate::market::FillSide;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimulationMetrics {
    pub steps: usize,
    pub final_pnl: f64,
    pub min_pnl: f64,
    pub max_pnl: f64,
    pub max_drawdown: f64,
    pub final_inventory: f64,
    pub max_abs_inventory: f64,
    pub avg_abs_inventory: f64,
    pub total_fills: usize,
    pub buy_fills: usize,
    pub sell_fills: usize,
    pub traded_quantity: f64,
    pub traded_notional: f64,
    pub total_fees: f64,
    pub total_adverse_selection: f64,
    pub low_vol_steps: usize,
    pub normal_vol_steps: usize,
    pub high_vol_steps: usize,
}

impl SimulationMetrics {
    pub fn from_result(result: &SimulationResult) -> Option<Self> {
        let first_step = result.steps.first()?;
        let final_step = result.steps.last()?;

        let mut min_pnl = first_step.pnl;
        let mut max_pnl = first_step.pnl;
        let mut peak_pnl = first_step.pnl;
        let mut max_drawdown = 0.0;
        let mut max_abs_inventory = 0.0;
        let mut sum_abs_inventory = 0.0;
        let mut total_fills = 0;
        let mut buy_fills = 0;
        let mut sell_fills = 0;
        let mut traded_quantity = 0.0;
        let mut traded_notional = 0.0;
        let mut total_fees = 0.0;
        let mut total_adverse_selection = 0.0;
        let mut low_vol_steps = 0;
        let mut normal_vol_steps = 0;
        let mut high_vol_steps = 0;

        for step in &result.steps {
            min_pnl = min_pnl.min(step.pnl);
            max_pnl = max_pnl.max(step.pnl);
            peak_pnl = peak_pnl.max(step.pnl);
            max_drawdown = f64::max(max_drawdown, peak_pnl - step.pnl);

            let abs_inventory = step.inventory.abs();
            max_abs_inventory = f64::max(max_abs_inventory, abs_inventory);
            sum_abs_inventory += abs_inventory;
            total_adverse_selection += step.adverse_selection_move.abs();

            match step.regime {
                MarketRegime::LowVol => low_vol_steps += 1,
                MarketRegime::NormalVol => normal_vol_steps += 1,
                MarketRegime::HighVol => high_vol_steps += 1,
            }

            for fill in &step.fills {
                total_fills += 1;
                traded_quantity += fill.quantity;
                traded_notional += fill.notional();
                total_fees += fill.fee;

                match fill.side {
                    FillSide::Buy => buy_fills += 1,
                    FillSide::Sell => sell_fills += 1,
                }
            }
        }

        Some(Self {
            steps: result.steps.len(),
            final_pnl: final_step.pnl,
            min_pnl,
            max_pnl,
            max_drawdown,
            final_inventory: final_step.inventory,
            max_abs_inventory,
            avg_abs_inventory: sum_abs_inventory / result.steps.len() as f64,
            total_fills,
            buy_fills,
            sell_fills,
            traded_quantity,
            traded_notional,
            total_fees,
            total_adverse_selection,
            low_vol_steps,
            normal_vol_steps,
            high_vol_steps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulation::{SimulationConfig, run_simulation};
    use crate::strategy::market_maker::StrategyParams;

    #[test]
    fn metrics_summarize_simulation_result() {
        let strategy = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };

        let result = run_simulation(
            SimulationConfig {
                steps: 1_000,
                ..SimulationConfig::default()
            },
            &strategy,
        );

        let metrics = SimulationMetrics::from_result(&result).unwrap();

        assert_eq!(metrics.steps, 1_000);
        assert_eq!(metrics.total_fills, metrics.buy_fills + metrics.sell_fills);
        assert!(metrics.max_abs_inventory >= metrics.final_inventory.abs());
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.traded_quantity >= 0.0);
        assert!(metrics.traded_notional >= 0.0);
        assert!(metrics.total_fees >= 0.0);
        assert!(metrics.total_adverse_selection >= 0.0);
        assert_eq!(
            metrics.steps,
            metrics.low_vol_steps + metrics.normal_vol_steps + metrics.high_vol_steps
        );
    }

    #[test]
    fn metrics_return_none_for_empty_result() {
        let result = SimulationResult { steps: Vec::new() };

        assert!(SimulationMetrics::from_result(&result).is_none());
    }
}

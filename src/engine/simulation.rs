use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::engine::state::SystemState;
use crate::market::{Fill, Quote};
use crate::strategy::QuoteStrategy;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub steps: usize,
    pub initial_mid_price: f64,
    pub seed: u64,
    pub price_volatility: f64,
    pub fill_price_noise: f64,
    pub order_quantity: f64,
    pub fee_rate: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            steps: 10_000,
            initial_mid_price: 100.0,
            seed: 42,
            price_volatility: 0.1,
            fill_price_noise: 0.1,
            order_quantity: 1.0,
            fee_rate: 0.001,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStep {
    pub mid_price: f64,
    pub quote: Quote,
    pub fills: Vec<Fill>,
    pub inventory: f64,
    pub cash: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub steps: Vec<SimulationStep>,
}

impl SimulationResult {
    pub fn final_step(&self) -> Option<&SimulationStep> {
        self.steps.last()
    }
}

pub fn run_simulation<S>(config: SimulationConfig, strategy: &S) -> SimulationResult
where
    S: QuoteStrategy,
{
    let mut state = SystemState::new(config.initial_mid_price);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let price_move = Normal::new(0.0, config.price_volatility).unwrap();
    let fill_noise = Normal::new(0.0, config.fill_price_noise).unwrap();

    let mut steps = Vec::with_capacity(config.steps);

    for _ in 0..config.steps {
        state.mid_price += price_move.sample(&mut rng);

        let quote = strategy.quote(&state);
        let market_price = state.mid_price + fill_noise.sample(&mut rng);
        let fills = simulate_fills(quote, market_price, config.order_quantity, config.fee_rate);

        for fill in &fills {
            state.apply_fill(*fill);
        }

        state.mark_to_market();

        steps.push(SimulationStep {
            mid_price: state.mid_price,
            quote,
            fills,
            inventory: state.inventory,
            cash: state.cash,
            pnl: state.pnl,
        });
    }

    SimulationResult { steps }
}

fn simulate_fills(quote: Quote, market_price: f64, quantity: f64, fee_rate: f64) -> Vec<Fill> {
    let mut fills = Vec::with_capacity(2);

    if market_price <= quote.bid {
        let fee = quote.bid * quantity * fee_rate;
        fills.push(Fill::buy(quote.bid, quantity, fee));
    }

    if market_price >= quote.ask {
        let fee = quote.ask * quantity * fee_rate;
        fills.push(Fill::sell(quote.ask, quantity, fee));
    }

    fills
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::market_maker::StrategyParams;

    #[test]
    fn simulation_records_each_step() {
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

        assert_eq!(result.steps.len(), 1_000);
    }

    #[test]
    fn simulation_moves_price() {
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

        let first_price = result.steps.first().unwrap().mid_price;
        let last_price = result.steps.last().unwrap().mid_price;

        assert_ne!(first_price, last_price);
    }
}

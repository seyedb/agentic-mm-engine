use crate::engine::state::SystemState;
use crate::strategy::market_maker::{compute_quotes, StrategyParams};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub struct BacktestResult {
    pub final_pnl: f64,
    pub final_inventory: f64,
    pub final_mid_price: f64,
    pub num_steps: usize,
}

pub fn run_backtest(
    steps: usize,
    initial_price: f64,
    params: StrategyParams,
) -> BacktestResult {
    let mut state = SystemState {
        mid_price: initial_price,
        inventory: 0.0,
        cash: 0.0,
        pnl: 0.0,
    };

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.1).unwrap();

    for _ in 0..steps {
        let price_move = normal.sample(&mut rng);
        state.mid_price += price_move;

        let (bid, ask) = compute_quotes(&state, &params);

        if state.mid_price <= bid {
            state.inventory += 1.0;
            state.cash -= bid;
        }

        if state.mid_price >= ask {
            state.inventory -= 1.0;
            state.cash += ask;
        }

        state.pnl = state.cash + state.inventory * state.mid_price;
    }

    BacktestResult {
        final_pnl: state.pnl,
        final_inventory: state.inventory,
        final_mid_price: state.mid_price,
        num_steps: steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_runs() {
        let params = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };

        let result = run_backtest(1000, 100.0, params);

        assert_eq!(result.num_steps, 1000);
        assert!(result.final_mid_price > 0.0);
        // use remaining fields
        assert!(result.final_pnl.is_finite());
        assert!(result.final_inventory.is_finite());
    }
}
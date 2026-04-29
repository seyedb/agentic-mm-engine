use crate::engine::state::SystemState;
use crate::strategy::market_maker::{compute_quotes, StrategyParams};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub struct BacktestResult {
    pub pnl_path: Vec<f64>,
    pub inventory_path: Vec<f64>,
    pub mid_price_path: Vec<f64>,
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

    let mut pnl_path = Vec::new();
    let mut inventory_path = Vec::new();
    let mut mid_price_path = Vec::new();

    for _ in 0..steps {
        let price_move = normal.sample(&mut rng);
        state.mid_price += price_move;
        // let market_price = state.mid_price + normal.sample(&mut rng);

        let (bid, ask) = compute_quotes(&state, &params);

        if state.mid_price <= bid {
            state.inventory += 1.0;
            state.cash -= bid;
        }

        if state.mid_price >= ask {
            state.inventory -= 1.0;
            state.cash += ask;
        }

        // if market_price <= bid {
        //     state.inventory += 1.0;
        //     state.cash -= bid;
        // }
        //
        // if market_price >= ask {
        //     state.inventory -= 1.0;
        //     state.cash += ask;
        // }


        state.pnl = state.cash + state.inventory * state.mid_price;

        pnl_path.push(state.pnl);
        inventory_path.push(state.inventory);
        mid_price_path.push(state.mid_price);
    }

    BacktestResult {
        pnl_path,
        inventory_path,
        mid_price_path,
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

        assert_eq!(result.pnl_path.len(), 1000);
        assert_eq!(result.inventory_path.len(), 1000);
        assert_eq!(result.mid_price_path.len(), 1000);
    }

    #[test]
    fn test_backtest_price_moves() {
        let params = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };

        let result = run_backtest(1000, 100.0, params);

        let first_price = result.mid_price_path.first().unwrap();
        let last_price = result.mid_price_path.last().unwrap();

        assert_ne!(first_price, last_price);
    }
}
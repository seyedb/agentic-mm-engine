mod engine;
mod strategy;
mod api;

use tokio::task;
use engine::state::SystemState;
use strategy::market_maker::{compute_quotes, StrategyParams};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[tokio::main]
async fn main() {
    // simulation state
    task::spawn(async move {
        let mut state = SystemState {
            mid_price: 100.0,
            inventory: 0.0,
            cash: 0.0,
            pnl: 0.0,
        };

        let mut params = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };

        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.1).unwrap();

        loop {
            // simulate market
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

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    });

    println!("simulation running (no API)");
    tokio::signal::ctrl_c().await.unwrap();
}
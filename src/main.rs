mod engine;
mod strategy;

use engine::state::SystemState;
use rand_distr::{Distribution, Normal};
use rand::rng;
use strategy::market_maker::{compute_quotes, StrategyParams};

fn main() {
    let mut rng = rng();
    // simulate volatility - price noise
    let normal = Normal::new(0.0, 0.1).unwrap();

    let mut state = SystemState {
        mid_price: 100.0,
        inventory: 0.0,
        cash: 0.0,
        pnl: 0.0,
    };

    let params = StrategyParams {
        /* expected behaviour:
           smaller spread: more fills, more risk
           larger spread: fewer fills, more stable
        */
        spread: 0.5,
        /* expected behaviour:
           high: more control over inventory
           too high: stop trading
        */
        skew_coeff: 0.05,
    };

    for t in 0..10_000 {
        // simulate random price move
        let price_change = normal.sample(&mut rng);
        let new_price = state.mid_price + price_change;

        // quoting
        let (bid, ask) = compute_quotes(&state, &params);

        // fills
        if new_price <= bid {
            state.inventory += 1.0;
            state.cash -= bid;
        }

        if new_price >= ask {
            state.inventory -= 1.0;
            state.cash += ask;
        }

        state.mid_price = new_price;
        state.pnl = state.cash + state.inventory * state.mid_price;

        if t % 500 == 0 {
            println!(
                "t={} price={:.2} inv={:.2} pnl={:.2}",
                t, state.mid_price, state.inventory, state.pnl
            );
        }
    }

    println!("final State: {:?}", state);
}
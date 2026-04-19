use rand::rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug)]
struct SystemState {
    mid_price: f64,
    inventory: f64,
    cash: f64,
    pnl: f64,
}

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

    /* expected behaviour:
       smaller spread: more fills, more risk
       larger spread: fewer fills, more stable
    */
    let spread = 0.5;
    /* expected behaviour:
       high: more control over inventory
       too high: stop trading
     */
    let skew_coeff = 0.05;
    for t in 0..10_000 {
        // simulate random price move
        let price_change = normal.sample(&mut rng);
        let new_price = state.mid_price + price_change;

        // quoting
        let skew = state.inventory * skew_coeff;
        let bid = state.mid_price - spread / 2.0 - skew;
        let ask = state.mid_price + spread / 2.0 - skew;

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

        // mark-to-market PnL
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
mod engine;
mod strategy;
mod api;

use tokio::task;
use tokio::sync::mpsc;
use axum::{Router, routing::{get, post}};
use std::net::SocketAddr;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use api::messages::{Command, Query};
use api::handlers::{AppState, get_state, update_params};
use engine::state::SystemState;
// use engine::backtest::run_backtest;
use strategy::market_maker::{compute_quotes, StrategyParams};

#[tokio::main]
async fn main() {
    // let result = run_backtest(
    //     10_000,
    //     100.0,
    //     StrategyParams {
    //         spread: 0.5,
    //         skew_coeff: 0.05,
    //     },
    // );
    //
    // println!("backtest pnl: {}", result.final_pnl);
    // println!("backtest inventory: {}", result.final_inventory);
    // println!("backtest final price: {}", result.final_mid_price);

    // channels
    let (cmd_tx, mut cmd_rx) = mpsc::channel(100);
    let (query_tx, mut query_rx) = mpsc::channel(100);

    // spawn simulation loop
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
            // handle commands
            if let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    Command::UpdateParams(p) => {
                        println!("updated params: {:?}", p);
                        params = p;
                    }
                }
            }

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

            // handle queries
            while let Ok(query) = query_rx.try_recv() {
                match query {
                    Query::GetState(resp_tx) => {
                        let _ = resp_tx.send(state.clone());
                    }
                }
            }

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    });

    // build api state
    let app_state = AppState{
        cmd_tx,
        query_tx,
    };

    // build router
    let app = Router::new()
        .route("/state", get(get_state))
        .route("/action", post(update_params))
        .with_state(app_state);

    // start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("api running at http://{}", addr);

    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app)
        .await
        .unwrap();

}
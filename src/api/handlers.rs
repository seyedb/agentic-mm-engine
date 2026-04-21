use axum::{extract::State, Json};
use tokio::sync::{mpsc, oneshot};

use crate::engine::state::SystemState;
use crate::strategy::market_maker::StrategyParams;
use crate::api::messages::{Command, Query};

#[derive(Clone)]
pub struct AppState {
    pub cmd_tx: mpsc::Sender<Command>,
    pub query_tx: mpsc::Sender<Query>,
}

// GET /state
pub async fn get_state(
    State(app): State<AppState>,
) -> Json<SystemState> {
    let (tx, rx) = oneshot::channel();

    app.query_tx
        .send(Query::GetState(tx))
        .await
        .unwrap();

    let state = rx.await.unwrap();
    Json(state)
}

// POST /action
pub async fn update_params(
    State(app): State<AppState>,
    Json(params): Json<StrategyParams>,
) -> &'static str {
    app.cmd_tx
        .send(Command::UpdateParams(params))
        .await
        .unwrap();

    "ok"
}
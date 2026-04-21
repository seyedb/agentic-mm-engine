use crate::engine::state::SystemState;
use crate::strategy::market_maker::StrategyParams;
use tokio::sync::oneshot;

#[derive(Debug)]
pub enum Command {
    UpdateParams(StrategyParams),
}

#[derive(Debug)]
pub enum Query {
    GetState(oneshot::Sender<SystemState>),
}
use crate::engine::state::SystemState;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyParams {
    pub spread: f64,
    pub skew_coeff: f64,
}
pub fn compute_quotes(state: &SystemState, params: &StrategyParams) -> (f64, f64) {
    let skew = state.inventory * params.skew_coeff;
    let bid = state.mid_price - params.spread / 2.0 - skew;
    let ask = state.mid_price + params.spread / 2.0 - skew;

    (bid, ask)
}
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::state::SystemState;

    #[test]
    fn test_quotes_symmetry() {
        let state = SystemState {
            mid_price: 100.0,
            inventory: 0.0,
            cash: 0.0,
            pnl: 0.0,
        };

        let params = StrategyParams {
            spread: 1.0,
            skew_coeff: 0.0,
        };

        let (bid, ask) = compute_quotes(&state, &params);

        assert_eq!(bid, 99.5);
        assert_eq!(ask, 100.5);
    }

    #[test]
    fn test_skew_effect() {
        let state = SystemState {
            mid_price: 100.0,
            inventory: 10.0,
            cash: 0.0,
            pnl: 0.0,
        };

        let params = StrategyParams {
            spread: 1.0,
            skew_coeff: 0.1,
        };

        let (bid, ask) = compute_quotes(&state, &params);

        // inventory positive -> quotes shift down
        assert!(bid < 99.5);
        assert!(ask < 100.5);
    }
}
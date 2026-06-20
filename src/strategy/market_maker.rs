use crate::engine::state::SystemState;
use crate::market::Quote;
use crate::strategy::QuoteStrategy;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyParams {
    pub spread: f64,
    pub skew_coeff: f64,
}

impl QuoteStrategy for StrategyParams {
    fn quote(&self, state: &SystemState) -> Quote {
        compute_quote(state, self)
    }
}

pub fn compute_quote(state: &SystemState, params: &StrategyParams) -> Quote {
    let skew = state.inventory * params.skew_coeff;

    Quote {
        bid: state.mid_price - params.spread / 2.0 - skew,
        ask: state.mid_price + params.spread / 2.0 - skew,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::state::SystemState;

    #[test]
    fn test_quotes_symmetry() {
        let state = SystemState::new(100.0);

        let params = StrategyParams {
            spread: 1.0,
            skew_coeff: 0.0,
        };

        let quote = compute_quote(&state, &params);

        assert_eq!(quote.bid, 99.5);
        assert_eq!(quote.ask, 100.5);
        assert_eq!(quote.spread(), 1.0);
    }

    #[test]
    fn test_skew_effect() {
        let mut state = SystemState::new(100.0);
        state.inventory = 10.0;

        let params = StrategyParams {
            spread: 1.0,
            skew_coeff: 0.1,
        };

        let quote = compute_quote(&state, &params);

        // inventory positive -> quotes shift down
        assert!(quote.bid < 99.5);
        assert!(quote.ask < 100.5);
    }
}

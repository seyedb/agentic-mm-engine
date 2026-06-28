use serde::{Deserialize, Serialize};

use crate::engine::state::SystemState;
use crate::market::Quote;
use crate::strategy::{QuoteStrategy, StrategyContext};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VolatilityAwareParams {
    pub base_spread: f64,
    pub volatility_coeff: f64,
    pub skew_coeff: f64,
}

impl QuoteStrategy for VolatilityAwareParams {
    fn quote(&self, state: &SystemState, context: &StrategyContext) -> Quote {
        compute_quote(state, context, self)
    }
}

pub fn compute_quote(
    state: &SystemState,
    context: &StrategyContext,
    params: &VolatilityAwareParams,
) -> Quote {
    let effective_spread =
        params.base_spread + params.volatility_coeff * context.estimated_volatility;
    let skew = state.inventory * params.skew_coeff;

    Quote {
        bid: state.mid_price - effective_spread / 2.0 - skew,
        ask: state.mid_price + effective_spread / 2.0 - skew,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_volatility_uses_base_spread() {
        let state = SystemState::new(100.0);
        let context = StrategyContext {
            estimated_volatility: 0.0,
            ..StrategyContext::default()
        };
        let params = VolatilityAwareParams {
            base_spread: 1.0,
            volatility_coeff: 2.0,
            skew_coeff: 0.0,
        };

        let quote = compute_quote(&state, &context, &params);

        assert_eq!(quote.bid, 99.5);
        assert_eq!(quote.ask, 100.5);
        assert_eq!(quote.spread(), 1.0);
    }

    #[test]
    fn higher_volatility_widens_quotes() {
        let state = SystemState::new(100.0);
        let calm_context = StrategyContext {
            estimated_volatility: 0.1,
            ..StrategyContext::default()
        };
        let volatile_context = StrategyContext {
            estimated_volatility: 0.5,
            ..StrategyContext::default()
        };
        let params = VolatilityAwareParams {
            base_spread: 1.0,
            volatility_coeff: 2.0,
            skew_coeff: 0.0,
        };

        let calm_quote = compute_quote(&state, &calm_context, &params);
        let volatile_quote = compute_quote(&state, &volatile_context, &params);

        assert!(volatile_quote.spread() > calm_quote.spread());
        assert!(volatile_quote.bid < calm_quote.bid);
        assert!(volatile_quote.ask > calm_quote.ask);
    }

    #[test]
    fn inventory_skew_still_shifts_quotes() {
        let mut state = SystemState::new(100.0);
        state.inventory = 10.0;
        let context = StrategyContext {
            estimated_volatility: 0.0,
            ..StrategyContext::default()
        };
        let params = VolatilityAwareParams {
            base_spread: 1.0,
            volatility_coeff: 0.0,
            skew_coeff: 0.1,
        };

        let quote = compute_quote(&state, &context, &params);

        assert_eq!(quote.bid, 98.5);
        assert_eq!(quote.ask, 99.5);
    }
}

use serde::{Deserialize, Serialize};

use crate::engine::state::SystemState;
use crate::market::Quote;
use crate::strategy::{QuoteStrategy, StrategyContext};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InventoryRiskParams {
    pub base_spread: f64,
    pub volatility_coeff: f64,
    pub risk_aversion: f64,
}

impl QuoteStrategy for InventoryRiskParams {
    fn quote(&self, state: &SystemState, context: &StrategyContext) -> Quote {
        compute_quote(state, context, self)
    }
}

pub fn compute_quote(
    state: &SystemState,
    context: &StrategyContext,
    params: &InventoryRiskParams,
) -> Quote {
    let variance = context.estimated_volatility * context.estimated_volatility;
    let reservation_price = state.mid_price - params.risk_aversion * state.inventory * variance;
    let effective_spread = (params.base_spread
        + params.volatility_coeff * context.estimated_volatility
        + params.risk_aversion * variance)
        .max(0.0);

    Quote {
        bid: reservation_price - effective_spread / 2.0,
        ask: reservation_price + effective_spread / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_inventory_and_volatility_quotes_around_mid() {
        let state = SystemState::new(100.0);
        let context = StrategyContext {
            estimated_volatility: 0.0,
            ..StrategyContext::default()
        };
        let params = InventoryRiskParams {
            base_spread: 1.0,
            volatility_coeff: 2.0,
            risk_aversion: 4.0,
        };

        let quote = compute_quote(&state, &context, &params);

        assert_eq!(quote.bid, 99.5);
        assert_eq!(quote.ask, 100.5);
        assert_eq!(quote.spread(), 1.0);
    }

    #[test]
    fn positive_inventory_moves_reservation_price_lower() {
        let mut state = SystemState::new(100.0);
        state.inventory = 10.0;
        let context = StrategyContext {
            estimated_volatility: 0.2,
            ..StrategyContext::default()
        };
        let params = InventoryRiskParams {
            base_spread: 1.0,
            volatility_coeff: 0.0,
            risk_aversion: 2.0,
        };

        let quote = compute_quote(&state, &context, &params);
        let center = (quote.bid + quote.ask) / 2.0;

        assert!(center < state.mid_price);
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
        let params = InventoryRiskParams {
            base_spread: 1.0,
            volatility_coeff: 2.0,
            risk_aversion: 4.0,
        };

        let calm_quote = compute_quote(&state, &calm_context, &params);
        let volatile_quote = compute_quote(&state, &volatile_context, &params);

        assert!(volatile_quote.spread() > calm_quote.spread());
    }
}

use serde::{Deserialize, Serialize};

use crate::engine::state::SystemState;
use crate::market::Quote;
use crate::strategy::{QuoteStrategy, StrategyContext};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AvellanedaStoikovParams {
    pub risk_aversion: f64,
    pub liquidity_depth: f64,
    pub horizon: f64,
    pub min_spread: f64,
}

impl QuoteStrategy for AvellanedaStoikovParams {
    fn quote(&self, state: &SystemState, context: &StrategyContext) -> Quote {
        compute_quote(state, context, self)
    }
}

pub fn compute_quote(
    state: &SystemState,
    context: &StrategyContext,
    params: &AvellanedaStoikovParams,
) -> Quote {
    let risk_aversion = params.risk_aversion.max(0.0);
    let liquidity_depth = params.liquidity_depth.max(f64::EPSILON);
    let horizon = params.horizon.max(0.0);
    let variance = context.estimated_volatility * context.estimated_volatility;
    let inventory_risk = risk_aversion * variance * horizon;
    let reservation_price = state.mid_price - state.inventory * inventory_risk;
    let spread =
        (inventory_risk + liquidity_spread(risk_aversion, liquidity_depth)).max(params.min_spread);

    Quote {
        bid: reservation_price - spread / 2.0,
        ask: reservation_price + spread / 2.0,
    }
}

fn liquidity_spread(risk_aversion: f64, liquidity_depth: f64) -> f64 {
    if risk_aversion <= f64::EPSILON {
        2.0 / liquidity_depth
    } else {
        2.0 * (1.0 + risk_aversion / liquidity_depth).ln() / risk_aversion
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_inventory_quotes_around_mid() {
        let state = SystemState::new(100.0);
        let context = StrategyContext {
            estimated_volatility: 0.2,
            ..StrategyContext::default()
        };
        let params = AvellanedaStoikovParams {
            risk_aversion: 0.1,
            liquidity_depth: 20.0,
            horizon: 10.0,
            min_spread: 0.1,
        };

        let quote = compute_quote(&state, &context, &params);
        let center = (quote.bid + quote.ask) / 2.0;

        assert!((center - state.mid_price).abs() < 1e-12);
        assert!(quote.spread() >= params.min_spread);
    }

    #[test]
    fn positive_inventory_lowers_reservation_price() {
        let mut state = SystemState::new(100.0);
        state.inventory = 5.0;
        let context = StrategyContext {
            estimated_volatility: 0.3,
            ..StrategyContext::default()
        };
        let params = AvellanedaStoikovParams {
            risk_aversion: 0.2,
            liquidity_depth: 10.0,
            horizon: 5.0,
            min_spread: 0.1,
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
        let params = AvellanedaStoikovParams {
            risk_aversion: 0.2,
            liquidity_depth: 10.0,
            horizon: 5.0,
            min_spread: 0.1,
        };

        let calm_quote = compute_quote(&state, &calm_context, &params);
        let volatile_quote = compute_quote(&state, &volatile_context, &params);

        assert!(volatile_quote.spread() > calm_quote.spread());
    }
}

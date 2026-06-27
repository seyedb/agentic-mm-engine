pub mod market_maker;
pub mod volatility_aware;

use crate::engine::state::SystemState;
use crate::market::Quote;

#[derive(Debug, Clone, Copy, Default)]
pub struct StrategyContext {
    pub estimated_volatility: f64,
}

pub trait QuoteStrategy {
    fn quote(&self, state: &SystemState, context: &StrategyContext) -> Quote;
}

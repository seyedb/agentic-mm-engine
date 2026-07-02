pub mod avellaneda_stoikov;
pub mod inventory_risk;
pub mod market_maker;
pub mod regime_adaptive;
pub mod volatility_aware;

use crate::engine::simulation::MarketRegime;
use crate::engine::state::SystemState;
use crate::market::Quote;

#[derive(Debug, Clone, Copy, Default)]
pub struct StrategyContext {
    pub estimated_volatility: f64,
    pub regime: MarketRegime,
}

pub trait QuoteStrategy {
    fn quote(&self, state: &SystemState, context: &StrategyContext) -> Quote;
}

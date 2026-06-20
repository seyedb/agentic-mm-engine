pub mod market_maker;

use crate::engine::state::SystemState;
use crate::market::Quote;

pub trait QuoteStrategy {
    fn quote(&self, state: &SystemState) -> Quote;
}

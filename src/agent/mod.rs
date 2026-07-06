use serde::{Deserialize, Serialize};

use crate::engine::simulation::MarketRegime;
use crate::engine::state::SystemState;
use crate::market::Quote;
use crate::strategy::avellaneda_stoikov::AvellanedaStoikovParams;
use crate::strategy::market_maker::StrategyParams;
use crate::strategy::{QuoteStrategy, StrategyContext};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerMode {
    FixedSpread,
    RiskManaged,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentDecision {
    pub mode: ControllerMode,
    pub quote: Quote,
}

pub trait MarketMakingAgent {
    fn decide(&self, state: &SystemState, context: &StrategyContext) -> AgentDecision;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleBasedControllerParams {
    pub fixed_spread: StrategyParams,
    pub risk_managed: AvellanedaStoikovParams,
    pub inventory_limit: f64,
}

impl MarketMakingAgent for RuleBasedControllerParams {
    fn decide(&self, state: &SystemState, context: &StrategyContext) -> AgentDecision {
        if should_use_risk_managed_mode(state, context, self.inventory_limit) {
            AgentDecision {
                mode: ControllerMode::RiskManaged,
                quote: self.risk_managed.quote(state, context),
            }
        } else {
            AgentDecision {
                mode: ControllerMode::FixedSpread,
                quote: self.fixed_spread.quote(state, context),
            }
        }
    }
}

impl QuoteStrategy for RuleBasedControllerParams {
    fn quote(&self, state: &SystemState, context: &StrategyContext) -> Quote {
        self.decide(state, context).quote
    }
}

fn should_use_risk_managed_mode(
    state: &SystemState,
    context: &StrategyContext,
    inventory_limit: f64,
) -> bool {
    context.regime == MarketRegime::HighVol || state.inventory.abs() >= inventory_limit.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> RuleBasedControllerParams {
        RuleBasedControllerParams {
            fixed_spread: StrategyParams {
                spread: 1.0,
                skew_coeff: 0.0,
            },
            risk_managed: AvellanedaStoikovParams {
                risk_aversion: 0.2,
                liquidity_depth: 10.0,
                horizon: 5.0,
                min_spread: 0.1,
            },
            inventory_limit: 3.0,
        }
    }

    #[test]
    fn uses_fixed_spread_in_normal_conditions() {
        let state = SystemState::new(100.0);
        let context = StrategyContext {
            estimated_volatility: 0.1,
            regime: MarketRegime::NormalVol,
        };

        let decision = controller().decide(&state, &context);

        assert_eq!(decision.mode, ControllerMode::FixedSpread);
        assert_eq!(decision.quote.bid, 99.5);
        assert_eq!(decision.quote.ask, 100.5);
    }

    #[test]
    fn switches_to_risk_managed_mode_in_high_volatility() {
        let state = SystemState::new(100.0);
        let context = StrategyContext {
            estimated_volatility: 0.5,
            regime: MarketRegime::HighVol,
        };

        let decision = controller().decide(&state, &context);

        assert_eq!(decision.mode, ControllerMode::RiskManaged);
        assert!(decision.quote.spread() >= 0.1);
    }

    #[test]
    fn switches_to_risk_managed_mode_near_inventory_limit() {
        let mut state = SystemState::new(100.0);
        state.inventory = 3.0;
        let context = StrategyContext {
            estimated_volatility: 0.2,
            regime: MarketRegime::NormalVol,
        };

        let decision = controller().decide(&state, &context);

        assert_eq!(decision.mode, ControllerMode::RiskManaged);
    }
}

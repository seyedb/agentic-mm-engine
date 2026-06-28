use serde::{Deserialize, Serialize};

use crate::engine::simulation::MarketRegime;
use crate::engine::state::SystemState;
use crate::market::Quote;
use crate::strategy::volatility_aware::{VolatilityAwareParams, compute_quote};
use crate::strategy::{QuoteStrategy, StrategyContext};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeAdaptiveVolatilityAwareParams {
    pub low_vol: VolatilityAwareParams,
    pub normal_vol: VolatilityAwareParams,
    pub high_vol: VolatilityAwareParams,
}

impl RegimeAdaptiveVolatilityAwareParams {
    pub fn params_for_regime(&self, regime: MarketRegime) -> &VolatilityAwareParams {
        match regime {
            MarketRegime::LowVol => &self.low_vol,
            MarketRegime::NormalVol => &self.normal_vol,
            MarketRegime::HighVol => &self.high_vol,
        }
    }
}

impl QuoteStrategy for RegimeAdaptiveVolatilityAwareParams {
    fn quote(&self, state: &SystemState, context: &StrategyContext) -> Quote {
        compute_quote(state, context, self.params_for_regime(context.regime))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(base_spread: f64) -> VolatilityAwareParams {
        VolatilityAwareParams {
            base_spread,
            volatility_coeff: 0.0,
            skew_coeff: 0.0,
        }
    }

    #[test]
    fn selects_parameters_from_current_regime() {
        let strategy = RegimeAdaptiveVolatilityAwareParams {
            low_vol: params(0.5),
            normal_vol: params(1.0),
            high_vol: params(2.0),
        };

        assert_eq!(
            strategy.params_for_regime(MarketRegime::LowVol).base_spread,
            0.5
        );
        assert_eq!(
            strategy
                .params_for_regime(MarketRegime::NormalVol)
                .base_spread,
            1.0
        );
        assert_eq!(
            strategy
                .params_for_regime(MarketRegime::HighVol)
                .base_spread,
            2.0
        );
    }

    #[test]
    fn high_vol_regime_uses_wider_quote() {
        let state = SystemState::new(100.0);
        let strategy = RegimeAdaptiveVolatilityAwareParams {
            low_vol: params(0.5),
            normal_vol: params(1.0),
            high_vol: params(2.0),
        };
        let low_context = StrategyContext {
            estimated_volatility: 0.1,
            regime: MarketRegime::LowVol,
        };
        let high_context = StrategyContext {
            estimated_volatility: 0.1,
            regime: MarketRegime::HighVol,
        };

        let low_quote = strategy.quote(&state, &low_context);
        let high_quote = strategy.quote(&state, &high_context);

        assert!(high_quote.spread() > low_quote.spread());
    }
}

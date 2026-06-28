use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::engine::state::SystemState;
use crate::market::{Fill, Quote};
use crate::strategy::{QuoteStrategy, StrategyContext};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub steps: usize,
    pub initial_mid_price: f64,
    pub seed: u64,
    pub price_volatility: f64,
    pub fill_price_noise: f64,
    #[serde(default)]
    pub fill_model: FillModelConfig,
    pub order_quantity: f64,
    pub fee_rate: f64,
    pub adverse_selection_per_fill: f64,
    pub volatility_window: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            steps: 10_000,
            initial_mid_price: 100.0,
            seed: 42,
            price_volatility: 0.1,
            fill_price_noise: 0.1,
            fill_model: FillModelConfig::default(),
            order_quantity: 1.0,
            fee_rate: 0.001,
            adverse_selection_per_fill: 0.02,
            volatility_window: 50,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FillModelConfig {
    #[serde(rename = "crossing_noise")]
    #[default]
    CrossingNoise,
    #[serde(rename = "distance_intensity")]
    DistanceIntensity {
        base_intensity: f64,
        distance_decay: f64,
        volatility_boost: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStep {
    pub mid_price: f64,
    pub estimated_volatility: f64,
    pub quote: Quote,
    pub fills: Vec<Fill>,
    pub adverse_selection_move: f64,
    pub inventory: f64,
    pub cash: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub steps: Vec<SimulationStep>,
}

impl SimulationResult {
    pub fn final_step(&self) -> Option<&SimulationStep> {
        self.steps.last()
    }
}

pub fn run_simulation<S>(config: SimulationConfig, strategy: &S) -> SimulationResult
where
    S: QuoteStrategy,
{
    let mut state = SystemState::new(config.initial_mid_price);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let price_move = Normal::new(0.0, config.price_volatility).unwrap();
    let fill_noise = Normal::new(0.0, config.fill_price_noise).unwrap();
    let mut volatility_estimator = RollingVolatility::new(config.volatility_window);

    let mut steps = Vec::with_capacity(config.steps);

    for _ in 0..config.steps {
        let previous_mid_price = state.mid_price;
        state.mid_price += price_move.sample(&mut rng);
        volatility_estimator.push(state.mid_price - previous_mid_price);
        let context = StrategyContext {
            estimated_volatility: volatility_estimator.estimate(),
        };

        let quote = strategy.quote(&state, &context);
        let fills = simulate_fills(
            quote,
            state.mid_price,
            context.estimated_volatility,
            &config,
            &fill_noise,
            &mut rng,
        );

        for fill in &fills {
            state.apply_fill(*fill);
        }

        let adverse_selection_move =
            apply_adverse_selection(&mut state, &fills, config.adverse_selection_per_fill);

        state.mark_to_market();

        steps.push(SimulationStep {
            mid_price: state.mid_price,
            estimated_volatility: context.estimated_volatility,
            quote,
            fills,
            adverse_selection_move,
            inventory: state.inventory,
            cash: state.cash,
            pnl: state.pnl,
        });
    }

    SimulationResult { steps }
}

struct RollingVolatility {
    window: usize,
    returns: Vec<f64>,
}

impl RollingVolatility {
    fn new(window: usize) -> Self {
        Self {
            window,
            returns: Vec::with_capacity(window.max(1)),
        }
    }

    fn push(&mut self, value: f64) {
        if self.window == 0 {
            return;
        }

        if self.returns.len() == self.window {
            self.returns.remove(0);
        }

        self.returns.push(value);
    }

    fn estimate(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let mean = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance = self
            .returns
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.returns.len() as f64;

        variance.sqrt()
    }
}

fn simulate_fills(
    quote: Quote,
    mid_price: f64,
    estimated_volatility: f64,
    config: &SimulationConfig,
    fill_noise: &Normal<f64>,
    rng: &mut StdRng,
) -> Vec<Fill> {
    match config.fill_model {
        FillModelConfig::CrossingNoise => {
            let market_price = mid_price + fill_noise.sample(rng);
            crossing_noise_fills(quote, market_price, config.order_quantity, config.fee_rate)
        }
        FillModelConfig::DistanceIntensity {
            base_intensity,
            distance_decay,
            volatility_boost,
        } => distance_intensity_fills(
            quote,
            mid_price,
            estimated_volatility,
            DistanceIntensityParams {
                base_intensity,
                distance_decay,
                volatility_boost,
            },
            config.order_quantity,
            config.fee_rate,
            rng,
        ),
    }
}

fn crossing_noise_fills(
    quote: Quote,
    market_price: f64,
    quantity: f64,
    fee_rate: f64,
) -> Vec<Fill> {
    let mut fills = Vec::with_capacity(2);

    if market_price <= quote.bid {
        let fee = quote.bid * quantity * fee_rate;
        fills.push(Fill::buy(quote.bid, quantity, fee));
    }

    if market_price >= quote.ask {
        let fee = quote.ask * quantity * fee_rate;
        fills.push(Fill::sell(quote.ask, quantity, fee));
    }

    fills
}

#[derive(Debug, Clone, Copy)]
struct DistanceIntensityParams {
    base_intensity: f64,
    distance_decay: f64,
    volatility_boost: f64,
}

fn distance_intensity_fills(
    quote: Quote,
    mid_price: f64,
    estimated_volatility: f64,
    params: DistanceIntensityParams,
    quantity: f64,
    fee_rate: f64,
    rng: &mut StdRng,
) -> Vec<Fill> {
    let mut fills = Vec::with_capacity(2);
    let bid_distance = (mid_price - quote.bid).max(0.0);
    let ask_distance = (quote.ask - mid_price).max(0.0);
    let bid_probability = fill_probability(bid_distance, estimated_volatility, params);
    let ask_probability = fill_probability(ask_distance, estimated_volatility, params);

    if rng.random::<f64>() < bid_probability {
        let fee = quote.bid * quantity * fee_rate;
        fills.push(Fill::buy(quote.bid, quantity, fee));
    }

    if rng.random::<f64>() < ask_probability {
        let fee = quote.ask * quantity * fee_rate;
        fills.push(Fill::sell(quote.ask, quantity, fee));
    }

    fills
}

fn fill_probability(
    quote_distance: f64,
    estimated_volatility: f64,
    params: DistanceIntensityParams,
) -> f64 {
    let volatility_multiplier = (1.0 + params.volatility_boost * estimated_volatility).max(0.0);
    let probability = params.base_intensity
        * (-params.distance_decay * quote_distance).exp()
        * volatility_multiplier;

    probability.clamp(0.0, 1.0)
}

fn apply_adverse_selection(
    state: &mut SystemState,
    fills: &[Fill],
    adverse_selection_per_fill: f64,
) -> f64 {
    let mut price_move = 0.0;

    for fill in fills {
        let signed_move = match fill.side {
            crate::market::FillSide::Buy => -adverse_selection_per_fill * fill.quantity,
            crate::market::FillSide::Sell => adverse_selection_per_fill * fill.quantity,
        };

        price_move += signed_move;
    }

    state.mid_price += price_move;
    price_move
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::market_maker::StrategyParams;

    #[test]
    fn simulation_records_each_step() {
        let strategy = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };

        let result = run_simulation(
            SimulationConfig {
                steps: 1_000,
                ..SimulationConfig::default()
            },
            &strategy,
        );

        assert_eq!(result.steps.len(), 1_000);
    }

    #[test]
    fn simulation_moves_price() {
        let strategy = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };

        let result = run_simulation(
            SimulationConfig {
                steps: 1_000,
                ..SimulationConfig::default()
            },
            &strategy,
        );

        let first_price = result.steps.first().unwrap().mid_price;
        let last_price = result.steps.last().unwrap().mid_price;

        assert_ne!(first_price, last_price);
    }

    #[test]
    fn adverse_selection_moves_price_against_fills() {
        let mut state = SystemState::new(100.0);

        let buy_move = apply_adverse_selection(&mut state, &[Fill::buy(100.0, 1.0, 0.0)], 0.02);

        assert!((buy_move + 0.02).abs() < 1e-9);
        assert!((state.mid_price - 99.98).abs() < 1e-9);

        let sell_move = apply_adverse_selection(&mut state, &[Fill::sell(100.0, 2.0, 0.0)], 0.02);

        assert!((sell_move - 0.04).abs() < 1e-9);
        assert!((state.mid_price - 100.02).abs() < 1e-9);
    }

    #[test]
    fn rolling_volatility_estimates_recent_price_moves() {
        let mut estimator = RollingVolatility::new(3);

        estimator.push(1.0);
        estimator.push(-1.0);
        estimator.push(1.0);

        assert!(estimator.estimate() > 0.0);

        estimator.push(0.0);

        assert_eq!(estimator.returns.len(), 3);
    }

    #[test]
    fn simulation_records_estimated_volatility() {
        let strategy = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.05,
        };

        let result = run_simulation(
            SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            &strategy,
        );

        assert!(
            result
                .steps
                .iter()
                .any(|step| step.estimated_volatility > 0.0)
        );
    }

    #[test]
    fn default_fill_model_is_crossing_noise() {
        assert!(matches!(
            SimulationConfig::default().fill_model,
            FillModelConfig::CrossingNoise
        ));
    }

    #[test]
    fn distance_intensity_probability_decays_with_distance() {
        let params = DistanceIntensityParams {
            base_intensity: 0.5,
            distance_decay: 4.0,
            volatility_boost: 0.0,
        };

        let near_probability = fill_probability(0.1, 0.0, params);
        let far_probability = fill_probability(1.0, 0.0, params);

        assert!(near_probability > far_probability);
    }

    #[test]
    fn distance_intensity_probability_increases_with_volatility() {
        let params = DistanceIntensityParams {
            base_intensity: 0.1,
            distance_decay: 2.0,
            volatility_boost: 2.0,
        };

        let calm_probability = fill_probability(0.5, 0.1, params);
        let volatile_probability = fill_probability(0.5, 0.5, params);

        assert!(volatile_probability > calm_probability);
    }

    #[test]
    fn distance_intensity_model_can_generate_fills() {
        let strategy = StrategyParams {
            spread: 0.5,
            skew_coeff: 0.0,
        };

        let result = run_simulation(
            SimulationConfig {
                steps: 10,
                price_volatility: 0.0,
                fill_model: FillModelConfig::DistanceIntensity {
                    base_intensity: 1.0,
                    distance_decay: 0.0,
                    volatility_boost: 0.0,
                },
                ..SimulationConfig::default()
            },
            &strategy,
        );

        assert!(result.steps.iter().all(|step| !step.fills.is_empty()));
    }
}

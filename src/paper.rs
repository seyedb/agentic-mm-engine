use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::agent::{
    AgentObservation, ControllerMode, MarketMakingAgent, PolicyAction, PolicyAgent,
    PolicyAgentDecision, PolicyAgentKind, PolicyReason, RuleBasedControllerParams,
};
use crate::engine::simulation::{MarketRegime, RegimeConfig};
use crate::engine::state::SystemState;
use crate::market::{Fill, MarketEvent, Quote};
use crate::strategy::StrategyContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSessionConfig {
    pub order_quantity: f64,
    pub fee_rate: f64,
    pub fee_spread_multiplier: f64,
    #[serde(default)]
    pub policy: PaperPolicyConfig,
    pub seed: u64,
    #[serde(default)]
    pub fill_model: PaperFillModelConfig,
    pub volatility_window: usize,
    #[serde(default)]
    pub regime: RegimeConfig,
}

impl Default for PaperSessionConfig {
    fn default() -> Self {
        Self {
            order_quantity: 1.0,
            fee_rate: 0.001,
            fee_spread_multiplier: 0.0,
            policy: PaperPolicyConfig::default(),
            seed: 42,
            fill_model: PaperFillModelConfig::default(),
            volatility_window: 50,
            regime: RegimeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
pub enum PaperPolicyConfig {
    #[serde(rename = "static")]
    #[default]
    Static,
    #[serde(rename = "adaptive")]
    Adaptive {
        min_spread: f64,
        max_spread: f64,
        volatility_spread_multiplier: f64,
        inventory_skew_multiplier: f64,
        touch_spread_multiplier: f64,
    },
    #[serde(rename = "hybrid")]
    Hybrid {
        min_spread: f64,
        max_spread: f64,
        volatility_spread_multiplier: f64,
        inventory_skew_multiplier: f64,
        touch_spread_multiplier: f64,
        drawdown_threshold: f64,
        inventory_threshold: f64,
        volatility_threshold: f64,
    },
    #[serde(rename = "selector")]
    Selector {
        min_spread: f64,
        max_spread: f64,
        volatility_spread_multiplier: f64,
        inventory_skew_multiplier: f64,
        touch_spread_multiplier: f64,
        volatility_weight: f64,
        spread_weight: f64,
        inventory_weight: f64,
        drawdown_weight: f64,
        activation_threshold: f64,
    },
    #[serde(rename = "learned_selector")]
    LearnedSelector {
        model_path: String,
        adaptive_policy: PaperAdaptivePolicyConfig,
        selector_policy: PaperSelectorPolicyConfig,
    },
    #[serde(rename = "linear_agent")]
    LinearAgent {
        model_path: String,
        adaptive_policy: PaperAdaptivePolicyConfig,
        selector_policy: PaperSelectorPolicyConfig,
    },
    #[serde(rename = "bandit_agent")]
    BanditAgent {
        model_path: String,
        adaptive_policy: PaperAdaptivePolicyConfig,
        selector_policy: PaperSelectorPolicyConfig,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PaperAdaptivePolicyConfig {
    pub min_spread: f64,
    pub max_spread: f64,
    pub volatility_spread_multiplier: f64,
    pub inventory_skew_multiplier: f64,
    pub touch_spread_multiplier: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PaperSelectorPolicyConfig {
    pub min_spread: f64,
    pub max_spread: f64,
    pub volatility_spread_multiplier: f64,
    pub inventory_skew_multiplier: f64,
    pub touch_spread_multiplier: f64,
    pub volatility_weight: f64,
    pub spread_weight: f64,
    pub inventory_weight: f64,
    pub drawdown_weight: f64,
    pub activation_threshold: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PaperFillModelConfig {
    #[serde(rename = "crossing")]
    #[default]
    Crossing,
    #[serde(rename = "touch_intensity")]
    TouchIntensity {
        base_intensity: f64,
        distance_decay: f64,
        volatility_boost: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSessionRow {
    pub timestamp_ms: u64,
    pub mid_price: f64,
    pub observed_bid: Option<f64>,
    pub observed_ask: Option<f64>,
    pub estimated_volatility: f64,
    pub regime: MarketRegime,
    pub agent_mode: ControllerMode,
    pub policy_agent: PolicyAgentKind,
    pub policy_action: PolicyAction,
    pub policy_score: Option<f64>,
    pub policy_mode: PaperPolicyMode,
    pub policy_trigger: PaperPolicyTrigger,
    pub bid: f64,
    pub ask: f64,
    pub inventory: f64,
    pub cash: f64,
    pub pnl: f64,
    pub drawdown: f64,
    pub fills: usize,
    pub buy_fills: usize,
    pub sell_fills: usize,
    pub fill_quantity: f64,
    pub fill_notional: f64,
    pub fees: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaperPolicyMode {
    Static,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaperPolicyTrigger {
    None,
    Configured,
    Inventory,
    Drawdown,
    Volatility,
    Spread,
    Multiple,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSessionResult {
    pub rows: Vec<PaperSessionRow>,
}

impl PaperSessionResult {
    pub fn final_row(&self) -> Option<&PaperSessionRow> {
        self.rows.last()
    }
}

pub fn run_paper_session(
    events: &[MarketEvent],
    agent: &RuleBasedControllerParams,
    config: PaperSessionConfig,
) -> PaperSessionResult {
    let mut runner = PaperSessionRunner::new(config);
    let mut rows = Vec::with_capacity(events.len());

    for event in events {
        rows.push(runner.step(*event, agent));
    }

    PaperSessionResult { rows }
}

pub struct PaperSessionRunner {
    config: PaperSessionConfig,
    state: Option<SystemState>,
    previous_mid_price: Option<f64>,
    volatility: RollingVolatility,
    learned_model: Option<LearnedPolicyModel>,
    linear_model: Option<LinearPolicyModel>,
    bandit_model: Option<BanditPolicyModel>,
    learned_features: RollingLearnedFeatures,
    peak_pnl: f64,
    rng: StdRng,
}

impl PaperSessionRunner {
    pub fn new(config: PaperSessionConfig) -> Self {
        let learned_model = load_learned_policy_model(&config.policy);
        let linear_model = load_linear_policy_model(&config.policy);
        let bandit_model = load_bandit_policy_model(&config.policy);
        let learned_window = learned_model
            .as_ref()
            .and_then(|model| model.feature_window)
            .or_else(|| linear_model.as_ref().and_then(|model| model.feature_window))
            .or_else(|| bandit_model.as_ref().and_then(|model| model.feature_window))
            .unwrap_or(30);
        Self {
            volatility: RollingVolatility::new(config.volatility_window),
            learned_features: RollingLearnedFeatures::new(learned_window),
            learned_model,
            linear_model,
            bandit_model,
            rng: StdRng::seed_from_u64(config.seed),
            config,
            state: None,
            previous_mid_price: None,
            peak_pnl: 0.0,
        }
    }

    pub fn step(
        &mut self,
        event: MarketEvent,
        agent: &RuleBasedControllerParams,
    ) -> PaperSessionRow {
        let state = self
            .state
            .get_or_insert_with(|| SystemState::new(event.mid_price));
        let mid_move = self
            .previous_mid_price
            .map(|previous_mid_price| (event.mid_price - previous_mid_price).abs())
            .unwrap_or(0.0);
        if let Some(previous_mid_price) = self.previous_mid_price {
            self.volatility.push(event.mid_price - previous_mid_price);
        }
        self.previous_mid_price = Some(event.mid_price);
        state.mid_price = event.mid_price;
        state.mark_to_market();

        let estimated_volatility = self.volatility.estimate();
        let regime = MarketRegime::classify(estimated_volatility, self.config.regime);
        let context = StrategyContext {
            estimated_volatility,
            regime,
        };
        let decision = agent.decide(state, &context);
        let observed_quote = event_quote(event);
        let current_drawdown = (self.peak_pnl - state.pnl).max(0.0);
        self.learned_features.push(LearnedFeatureSample {
            estimated_volatility,
            observed_spread: observed_quote.map(|quote| quote.spread()).unwrap_or(0.0),
            abs_mid_move: mid_move,
            abs_inventory: state.inventory.abs(),
            drawdown: current_drawdown,
        });
        let learned_features = self.learned_features.snapshot();
        let observation = AgentObservation {
            estimated_volatility,
            observed_spread: observed_quote.map(|quote| quote.spread()).unwrap_or(0.0),
            max_observed_spread: learned_features
                .map(|features| features.max_observed_spread)
                .unwrap_or(0.0),
            abs_mid_move: mid_move,
            abs_inventory: state.inventory.abs(),
            drawdown: current_drawdown,
        };
        let policy_decision = apply_paper_policy(
            PaperPolicyInput {
                quote: decision.quote,
                state,
                observed_quote,
                estimated_volatility,
                current_drawdown,
                observation,
            },
            &self.config.policy,
            self.learned_model.as_ref(),
            self.linear_model.as_ref(),
            self.bandit_model.as_ref(),
            learned_features,
        );
        let quote = fee_aware_quote(
            policy_decision.quote,
            state.mid_price,
            self.config.fee_rate,
            self.config.fee_spread_multiplier,
        );
        let fills = observed_quote
            .map(|observed_quote| {
                observed_quote_fills(
                    quote,
                    observed_quote,
                    estimated_volatility,
                    self.config.fill_model,
                    self.config.order_quantity,
                    self.config.fee_rate,
                    &mut self.rng,
                )
            })
            .unwrap_or_default();

        for fill in &fills {
            state.apply_fill(*fill);
        }
        state.mid_price = event.mid_price;
        state.mark_to_market();
        self.peak_pnl = f64::max(self.peak_pnl, state.pnl);
        let drawdown = self.peak_pnl - state.pnl;

        session_row(SessionRowInput {
            event,
            observed_quote,
            estimated_volatility,
            regime,
            agent_mode: decision.mode,
            policy_agent: policy_decision.agent.agent,
            policy_action: policy_decision.agent.action,
            policy_score: policy_decision.agent.score,
            policy_mode: policy_decision.mode,
            policy_trigger: policy_decision.trigger,
            quote,
            state,
            drawdown,
            fills: &fills,
        })
    }
}

fn apply_paper_policy(
    input: PaperPolicyInput<'_>,
    policy: &PaperPolicyConfig,
    learned_model: Option<&LearnedPolicyModel>,
    linear_model: Option<&LinearPolicyModel>,
    bandit_model: Option<&BanditPolicyModel>,
    learned_features: Option<LearnedFeatureSnapshot>,
) -> PaperPolicyDecision {
    match policy.clone() {
        PaperPolicyConfig::Static => PaperPolicyDecision {
            quote: input.quote,
            mode: PaperPolicyMode::Static,
            trigger: PaperPolicyTrigger::None,
            agent: PolicyAgentDecision {
                agent: PolicyAgentKind::Static,
                action: PolicyAction::Static,
                reason: PolicyReason::None,
                score: None,
            },
        },
        PaperPolicyConfig::Adaptive {
            min_spread,
            max_spread,
            volatility_spread_multiplier,
            inventory_skew_multiplier,
            touch_spread_multiplier,
        } => PaperPolicyDecision {
            quote: adaptive_quote(
                input.quote,
                input.state,
                input.observed_quote,
                input.estimated_volatility,
                AdaptivePolicyParams {
                    min_spread,
                    max_spread,
                    volatility_spread_multiplier,
                    inventory_skew_multiplier,
                    touch_spread_multiplier,
                },
            ),
            mode: PaperPolicyMode::Adaptive,
            trigger: PaperPolicyTrigger::Configured,
            agent: PolicyAgentDecision {
                agent: PolicyAgentKind::Adaptive,
                action: PolicyAction::Adaptive,
                reason: PolicyReason::Configured,
                score: None,
            },
        },
        PaperPolicyConfig::Hybrid {
            min_spread,
            max_spread,
            volatility_spread_multiplier,
            inventory_skew_multiplier,
            touch_spread_multiplier,
            drawdown_threshold,
            inventory_threshold,
            volatility_threshold,
        } => {
            let trigger = adaptive_policy_trigger(
                input.state,
                input.estimated_volatility,
                input.current_drawdown,
                drawdown_threshold,
                inventory_threshold,
                volatility_threshold,
            );
            if trigger == PaperPolicyTrigger::None {
                PaperPolicyDecision {
                    quote: input.quote,
                    mode: PaperPolicyMode::Static,
                    trigger,
                    agent: PolicyAgentDecision {
                        agent: PolicyAgentKind::Hybrid,
                        action: PolicyAction::Static,
                        reason: PolicyReason::None,
                        score: None,
                    },
                }
            } else {
                PaperPolicyDecision {
                    quote: adaptive_quote(
                        input.quote,
                        input.state,
                        input.observed_quote,
                        input.estimated_volatility,
                        AdaptivePolicyParams {
                            min_spread,
                            max_spread,
                            volatility_spread_multiplier,
                            inventory_skew_multiplier,
                            touch_spread_multiplier,
                        },
                    ),
                    mode: PaperPolicyMode::Adaptive,
                    trigger,
                    agent: PolicyAgentDecision {
                        agent: PolicyAgentKind::Hybrid,
                        action: PolicyAction::Adaptive,
                        reason: policy_reason_from_trigger(trigger),
                        score: None,
                    },
                }
            }
        }
        PaperPolicyConfig::Selector {
            min_spread,
            max_spread,
            volatility_spread_multiplier,
            inventory_skew_multiplier,
            touch_spread_multiplier,
            volatility_weight,
            spread_weight,
            inventory_weight,
            drawdown_weight,
            activation_threshold,
        } => selector_policy_decision(
            input.quote,
            input.state,
            input.observed_quote,
            input.estimated_volatility,
            input.observation,
            AdaptivePolicyParams {
                min_spread,
                max_spread,
                volatility_spread_multiplier,
                inventory_skew_multiplier,
                touch_spread_multiplier,
            },
            SelectorPolicyParams {
                volatility_weight,
                spread_weight,
                inventory_weight,
                drawdown_weight,
                activation_threshold,
            },
        ),
        PaperPolicyConfig::LearnedSelector {
            model_path: _,
            adaptive_policy,
            selector_policy,
        } => learned_selector_policy_decision(
            input,
            adaptive_policy,
            selector_policy,
            learned_model.expect("learned selector model should be loaded before paper session"),
            learned_features.expect("learned selector features should be available"),
        ),
        PaperPolicyConfig::LinearAgent {
            model_path: _,
            adaptive_policy,
            selector_policy,
        } => linear_agent_policy_decision(
            input,
            adaptive_policy,
            selector_policy,
            linear_model.expect("linear agent model should be loaded before paper session"),
            learned_features.expect("linear agent features should be available"),
        ),
        PaperPolicyConfig::BanditAgent {
            model_path: _,
            adaptive_policy,
            selector_policy,
        } => bandit_agent_policy_decision(
            input,
            adaptive_policy,
            selector_policy,
            bandit_model.expect("bandit agent model should be loaded before paper session"),
            learned_features.expect("bandit agent features should be available"),
        ),
    }
}

#[derive(Debug, Clone, Copy)]
struct PaperPolicyInput<'a> {
    quote: Quote,
    state: &'a SystemState,
    observed_quote: Option<Quote>,
    estimated_volatility: f64,
    current_drawdown: f64,
    observation: AgentObservation,
}

#[derive(Debug, Clone, Copy)]
struct PaperPolicyDecision {
    quote: Quote,
    mode: PaperPolicyMode,
    trigger: PaperPolicyTrigger,
    agent: PolicyAgentDecision,
}

#[derive(Debug, Clone, Copy)]
struct AdaptivePolicyParams {
    min_spread: f64,
    max_spread: f64,
    volatility_spread_multiplier: f64,
    inventory_skew_multiplier: f64,
    touch_spread_multiplier: f64,
}

#[derive(Debug, Clone, Copy)]
struct SelectorPolicyParams {
    volatility_weight: f64,
    spread_weight: f64,
    inventory_weight: f64,
    drawdown_weight: f64,
    activation_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
struct SelectorPolicyAgent {
    params: SelectorPolicyParams,
}

impl PolicyAgent for SelectorPolicyAgent {
    fn decide(&self, observation: AgentObservation) -> PolicyAgentDecision {
        let contributions = [
            (
                PolicyReason::Volatility,
                observation.estimated_volatility * self.params.volatility_weight,
            ),
            (
                PolicyReason::Spread,
                observation.observed_spread * self.params.spread_weight,
            ),
            (
                PolicyReason::Inventory,
                observation.abs_inventory * self.params.inventory_weight,
            ),
            (
                PolicyReason::Drawdown,
                observation.drawdown * self.params.drawdown_weight,
            ),
        ];
        let score = contributions
            .iter()
            .map(|(_reason, contribution)| contribution)
            .sum::<f64>();

        if score < self.params.activation_threshold {
            return PolicyAgentDecision {
                agent: PolicyAgentKind::Selector,
                action: PolicyAction::Static,
                reason: PolicyReason::None,
                score: Some(score),
            };
        }

        PolicyAgentDecision {
            agent: PolicyAgentKind::Selector,
            action: PolicyAction::Adaptive,
            reason: dominant_policy_reason(contributions),
            score: Some(score),
        }
    }
}

struct LearnedSelectorAgent<'a> {
    model: &'a LearnedPolicyModel,
}

impl PolicyAgent for LearnedSelectorAgent<'_> {
    fn decide(&self, observation: AgentObservation) -> PolicyAgentDecision {
        let features = LearnedFeatureSnapshot {
            estimated_volatility: observation.estimated_volatility,
            observed_spread: observation.observed_spread,
            max_observed_spread: observation.max_observed_spread,
            abs_mid_move: observation.abs_mid_move,
            abs_inventory: observation.abs_inventory,
            drawdown: observation.drawdown,
        };
        let score = learned_policy_score(self.model, features);
        let action = if score >= self.model.threshold {
            self.model.action_on.as_str()
        } else {
            self.model.action_off.as_str()
        };

        PolicyAgentDecision {
            agent: PolicyAgentKind::LearnedSelector,
            action: policy_action_from_model_action(action),
            reason: PolicyReason::ModelScore,
            score: Some(score),
        }
    }
}

struct LinearPolicyAgent<'a> {
    model: &'a LinearPolicyModel,
}

impl PolicyAgent for LinearPolicyAgent<'_> {
    fn decide(&self, observation: AgentObservation) -> PolicyAgentDecision {
        let features = LearnedFeatureSnapshot {
            estimated_volatility: observation.estimated_volatility,
            observed_spread: observation.observed_spread,
            max_observed_spread: observation.max_observed_spread,
            abs_mid_move: observation.abs_mid_move,
            abs_inventory: observation.abs_inventory,
            drawdown: observation.drawdown,
        };
        let (action, score) = self.model.best_action(features);
        PolicyAgentDecision {
            agent: PolicyAgentKind::LinearAgent,
            action: policy_action_from_model_action(action),
            reason: PolicyReason::ModelScore,
            score: Some(score),
        }
    }
}

struct BanditPolicyAgent<'a> {
    model: &'a BanditPolicyModel,
}

impl PolicyAgent for BanditPolicyAgent<'_> {
    fn decide(&self, observation: AgentObservation) -> PolicyAgentDecision {
        let features = LearnedFeatureSnapshot {
            estimated_volatility: observation.estimated_volatility,
            observed_spread: observation.observed_spread,
            max_observed_spread: observation.max_observed_spread,
            abs_mid_move: observation.abs_mid_move,
            abs_inventory: observation.abs_inventory,
            drawdown: observation.drawdown,
        };
        let (action, score) = self.model.best_action(features);
        PolicyAgentDecision {
            agent: PolicyAgentKind::BanditAgent,
            action: policy_action_from_model_action(action),
            reason: PolicyReason::ModelScore,
            score: Some(score),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct LearnedPolicyModel {
    action_on: String,
    action_off: String,
    weights: LearnedFeatureWeights,
    #[serde(default)]
    intercept: f64,
    threshold: f64,
    #[serde(default)]
    probability_threshold: Option<f64>,
    feature_scale: LearnedFeatureScale,
    #[serde(default)]
    feature_window: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct LinearPolicyModel {
    actions: Vec<String>,
    weights: BTreeMap<String, LearnedFeatureWeights>,
    intercepts: BTreeMap<String, f64>,
    feature_scale: LearnedFeatureScale,
    #[serde(default)]
    feature_window: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct BanditPolicyModel {
    actions: Vec<String>,
    alpha: f64,
    theta: BTreeMap<String, LearnedFeatureWeights>,
    inverse_covariance: BTreeMap<String, Vec<Vec<f64>>>,
    feature_scale: LearnedFeatureScale,
    #[serde(default)]
    feature_window: Option<usize>,
}

impl BanditPolicyModel {
    fn best_action(&self, features: LearnedFeatureSnapshot) -> (&str, f64) {
        self.actions
            .iter()
            .map(|action| (action.as_str(), bandit_policy_score(self, action, features)))
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(right.0))
            })
            .expect("bandit policy model should have at least one action")
    }
}

impl LinearPolicyModel {
    fn best_action(&self, features: LearnedFeatureSnapshot) -> (&str, f64) {
        self.actions
            .iter()
            .map(|action| (action.as_str(), linear_policy_score(self, action, features)))
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(right.0))
            })
            .expect("linear policy model should have at least one action")
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct LearnedFeatureWeights {
    estimated_volatility: f64,
    observed_spread: f64,
    max_observed_spread: f64,
    abs_mid_move: f64,
    abs_inventory: f64,
    drawdown: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct LearnedFeatureScale {
    estimated_volatility: f64,
    observed_spread: f64,
    max_observed_spread: f64,
    abs_mid_move: f64,
    abs_inventory: f64,
    drawdown: f64,
}

#[derive(Debug, Clone, Copy)]
struct LearnedFeatureSample {
    estimated_volatility: f64,
    observed_spread: f64,
    abs_mid_move: f64,
    abs_inventory: f64,
    drawdown: f64,
}

#[derive(Debug, Clone, Copy)]
struct LearnedFeatureSnapshot {
    estimated_volatility: f64,
    observed_spread: f64,
    max_observed_spread: f64,
    abs_mid_move: f64,
    abs_inventory: f64,
    drawdown: f64,
}

fn selector_policy_decision(
    quote: Quote,
    state: &SystemState,
    observed_quote: Option<Quote>,
    estimated_volatility: f64,
    observation: AgentObservation,
    adaptive_params: AdaptivePolicyParams,
    selector_params: SelectorPolicyParams,
) -> PaperPolicyDecision {
    let agent = SelectorPolicyAgent {
        params: selector_params,
    };
    let decision = agent.decide(observation);

    if decision.action == PolicyAction::Static {
        return PaperPolicyDecision {
            quote,
            mode: PaperPolicyMode::Static,
            trigger: PaperPolicyTrigger::None,
            agent: decision,
        };
    }

    let trigger = paper_trigger_from_reason(decision.reason);
    PaperPolicyDecision {
        quote: adaptive_quote(
            quote,
            state,
            observed_quote,
            estimated_volatility,
            adaptive_params,
        ),
        mode: PaperPolicyMode::Adaptive,
        trigger,
        agent: decision,
    }
}

fn learned_selector_policy_decision(
    input: PaperPolicyInput<'_>,
    adaptive_config: PaperAdaptivePolicyConfig,
    selector_config: PaperSelectorPolicyConfig,
    model: &LearnedPolicyModel,
    features: LearnedFeatureSnapshot,
) -> PaperPolicyDecision {
    let agent = LearnedSelectorAgent { model };
    let agent_decision = agent.decide(AgentObservation {
        estimated_volatility: features.estimated_volatility,
        observed_spread: features.observed_spread,
        max_observed_spread: features.max_observed_spread,
        abs_mid_move: features.abs_mid_move,
        abs_inventory: features.abs_inventory,
        drawdown: features.drawdown,
    });

    match agent_decision.action {
        PolicyAction::Static => PaperPolicyDecision {
            quote: input.quote,
            mode: PaperPolicyMode::Static,
            trigger: PaperPolicyTrigger::None,
            agent: agent_decision,
        },
        PolicyAction::Adaptive => PaperPolicyDecision {
            quote: adaptive_quote(
                input.quote,
                input.state,
                input.observed_quote,
                input.estimated_volatility,
                adaptive_config.into(),
            ),
            mode: PaperPolicyMode::Adaptive,
            trigger: PaperPolicyTrigger::Configured,
            agent: agent_decision,
        },
        PolicyAction::Selector => {
            let mut selector_decision = selector_policy_decision(
                input.quote,
                input.state,
                input.observed_quote,
                input.estimated_volatility,
                input.observation,
                adaptive_config.into(),
                selector_config.into(),
            );
            selector_decision.agent = agent_decision;
            selector_decision
        }
    }
}

fn linear_agent_policy_decision(
    input: PaperPolicyInput<'_>,
    adaptive_config: PaperAdaptivePolicyConfig,
    selector_config: PaperSelectorPolicyConfig,
    model: &LinearPolicyModel,
    features: LearnedFeatureSnapshot,
) -> PaperPolicyDecision {
    let agent = LinearPolicyAgent { model };
    let agent_decision = agent.decide(AgentObservation {
        estimated_volatility: features.estimated_volatility,
        observed_spread: features.observed_spread,
        max_observed_spread: features.max_observed_spread,
        abs_mid_move: features.abs_mid_move,
        abs_inventory: features.abs_inventory,
        drawdown: features.drawdown,
    });

    match agent_decision.action {
        PolicyAction::Static => PaperPolicyDecision {
            quote: input.quote,
            mode: PaperPolicyMode::Static,
            trigger: PaperPolicyTrigger::None,
            agent: agent_decision,
        },
        PolicyAction::Adaptive => PaperPolicyDecision {
            quote: adaptive_quote(
                input.quote,
                input.state,
                input.observed_quote,
                input.estimated_volatility,
                adaptive_config.into(),
            ),
            mode: PaperPolicyMode::Adaptive,
            trigger: PaperPolicyTrigger::Configured,
            agent: agent_decision,
        },
        PolicyAction::Selector => {
            let mut selector_decision = selector_policy_decision(
                input.quote,
                input.state,
                input.observed_quote,
                input.estimated_volatility,
                input.observation,
                adaptive_config.into(),
                selector_config.into(),
            );
            selector_decision.agent = agent_decision;
            selector_decision
        }
    }
}

fn bandit_agent_policy_decision(
    input: PaperPolicyInput<'_>,
    adaptive_config: PaperAdaptivePolicyConfig,
    selector_config: PaperSelectorPolicyConfig,
    model: &BanditPolicyModel,
    features: LearnedFeatureSnapshot,
) -> PaperPolicyDecision {
    let agent = BanditPolicyAgent { model };
    let agent_decision = agent.decide(AgentObservation {
        estimated_volatility: features.estimated_volatility,
        observed_spread: features.observed_spread,
        max_observed_spread: features.max_observed_spread,
        abs_mid_move: features.abs_mid_move,
        abs_inventory: features.abs_inventory,
        drawdown: features.drawdown,
    });

    match agent_decision.action {
        PolicyAction::Static => PaperPolicyDecision {
            quote: input.quote,
            mode: PaperPolicyMode::Static,
            trigger: PaperPolicyTrigger::None,
            agent: agent_decision,
        },
        PolicyAction::Adaptive => PaperPolicyDecision {
            quote: adaptive_quote(
                input.quote,
                input.state,
                input.observed_quote,
                input.estimated_volatility,
                adaptive_config.into(),
            ),
            mode: PaperPolicyMode::Adaptive,
            trigger: PaperPolicyTrigger::Configured,
            agent: agent_decision,
        },
        PolicyAction::Selector => {
            let mut selector_decision = selector_policy_decision(
                input.quote,
                input.state,
                input.observed_quote,
                input.estimated_volatility,
                input.observation,
                adaptive_config.into(),
                selector_config.into(),
            );
            selector_decision.agent = agent_decision;
            selector_decision
        }
    }
}

fn policy_action_from_model_action(action: &str) -> PolicyAction {
    match action {
        "static" => PolicyAction::Static,
        "adaptive" => PolicyAction::Adaptive,
        "selector" => PolicyAction::Selector,
        other => panic!("learned selector model chose unsupported action {other:?}"),
    }
}

fn learned_policy_score(model: &LearnedPolicyModel, features: LearnedFeatureSnapshot) -> f64 {
    model.intercept
        + normalized_feature(
            features.estimated_volatility,
            model.feature_scale.estimated_volatility,
        ) * model.weights.estimated_volatility
        + normalized_feature(
            features.observed_spread,
            model.feature_scale.observed_spread,
        ) * model.weights.observed_spread
        + normalized_feature(
            features.max_observed_spread,
            model.feature_scale.max_observed_spread,
        ) * model.weights.max_observed_spread
        + normalized_feature(features.abs_mid_move, model.feature_scale.abs_mid_move)
            * model.weights.abs_mid_move
        + normalized_feature(features.abs_inventory, model.feature_scale.abs_inventory)
            * model.weights.abs_inventory
        + normalized_feature(features.drawdown, model.feature_scale.drawdown)
            * model.weights.drawdown
}

fn linear_policy_score(
    model: &LinearPolicyModel,
    action: &str,
    features: LearnedFeatureSnapshot,
) -> f64 {
    let weights = model
        .weights
        .get(action)
        .unwrap_or_else(|| panic!("linear agent model missing weights for action {action:?}"));
    model.intercepts.get(action).copied().unwrap_or(0.0)
        + normalized_feature(
            features.estimated_volatility,
            model.feature_scale.estimated_volatility,
        ) * weights.estimated_volatility
        + normalized_feature(
            features.observed_spread,
            model.feature_scale.observed_spread,
        ) * weights.observed_spread
        + normalized_feature(
            features.max_observed_spread,
            model.feature_scale.max_observed_spread,
        ) * weights.max_observed_spread
        + normalized_feature(features.abs_mid_move, model.feature_scale.abs_mid_move)
            * weights.abs_mid_move
        + normalized_feature(features.abs_inventory, model.feature_scale.abs_inventory)
            * weights.abs_inventory
        + normalized_feature(features.drawdown, model.feature_scale.drawdown) * weights.drawdown
}

fn bandit_policy_score(
    model: &BanditPolicyModel,
    action: &str,
    features: LearnedFeatureSnapshot,
) -> f64 {
    let weights = model
        .theta
        .get(action)
        .unwrap_or_else(|| panic!("bandit agent model missing theta for action {action:?}"));
    let vector = normalized_feature_vector(features, model.feature_scale);
    let exploit = weights.estimated_volatility * vector[0]
        + weights.observed_spread * vector[1]
        + weights.max_observed_spread * vector[2]
        + weights.abs_mid_move * vector[3]
        + weights.abs_inventory * vector[4]
        + weights.drawdown * vector[5];
    let inverse = model.inverse_covariance.get(action).unwrap_or_else(|| {
        panic!("bandit agent model missing inverse covariance for action {action:?}")
    });
    let uncertainty = quadratic_form(&vector, inverse).max(0.0).sqrt();
    exploit + model.alpha * uncertainty
}

fn normalized_feature_vector(
    features: LearnedFeatureSnapshot,
    scale: LearnedFeatureScale,
) -> [f64; 6] {
    [
        normalized_feature(features.estimated_volatility, scale.estimated_volatility),
        normalized_feature(features.observed_spread, scale.observed_spread),
        normalized_feature(features.max_observed_spread, scale.max_observed_spread),
        normalized_feature(features.abs_mid_move, scale.abs_mid_move),
        normalized_feature(features.abs_inventory, scale.abs_inventory),
        normalized_feature(features.drawdown, scale.drawdown),
    ]
}

fn quadratic_form(vector: &[f64; 6], matrix: &[Vec<f64>]) -> f64 {
    vector
        .iter()
        .enumerate()
        .map(|(row, left)| {
            left * vector
                .iter()
                .enumerate()
                .map(|(col, right)| matrix[row][col] * right)
                .sum::<f64>()
        })
        .sum()
}

fn normalized_feature(value: f64, scale: f64) -> f64 {
    if scale <= 0.0 { 0.0 } else { value / scale }
}

fn load_learned_policy_model(policy: &PaperPolicyConfig) -> Option<LearnedPolicyModel> {
    let PaperPolicyConfig::LearnedSelector { model_path, .. } = policy else {
        return None;
    };
    let payload = fs::read_to_string(model_path).unwrap_or_else(|error| {
        panic!("failed to read learned selector model {model_path}: {error}")
    });
    let model: LearnedPolicyModel = serde_json::from_str(&payload).unwrap_or_else(|error| {
        panic!("failed to parse learned selector model {model_path}: {error}")
    });
    validate_learned_policy_model(model_path, &model);
    Some(model)
}

fn load_linear_policy_model(policy: &PaperPolicyConfig) -> Option<LinearPolicyModel> {
    let PaperPolicyConfig::LinearAgent { model_path, .. } = policy else {
        return None;
    };
    let payload = fs::read_to_string(model_path)
        .unwrap_or_else(|error| panic!("failed to read linear agent model {model_path}: {error}"));
    let model: LinearPolicyModel = serde_json::from_str(&payload)
        .unwrap_or_else(|error| panic!("failed to parse linear agent model {model_path}: {error}"));
    validate_linear_policy_model(model_path, &model);
    Some(model)
}

fn load_bandit_policy_model(policy: &PaperPolicyConfig) -> Option<BanditPolicyModel> {
    let PaperPolicyConfig::BanditAgent { model_path, .. } = policy else {
        return None;
    };
    let payload = fs::read_to_string(model_path)
        .unwrap_or_else(|error| panic!("failed to read bandit agent model {model_path}: {error}"));
    let model: BanditPolicyModel = serde_json::from_str(&payload)
        .unwrap_or_else(|error| panic!("failed to parse bandit agent model {model_path}: {error}"));
    validate_bandit_policy_model(model_path, &model);
    Some(model)
}

fn validate_learned_policy_model(path: &str, model: &LearnedPolicyModel) {
    for action in [&model.action_on, &model.action_off] {
        match action.as_str() {
            "static" | "adaptive" | "selector" => {}
            other => panic!("learned selector model {path} has unsupported action {other:?}"),
        }
    }
    if !model.threshold.is_finite() {
        panic!("learned selector model {path} has non-finite threshold");
    }
    if !model.intercept.is_finite() {
        panic!("learned selector model {path} has non-finite intercept");
    }
    if let Some(probability_threshold) = model.probability_threshold
        && (!probability_threshold.is_finite() || !(0.0..=1.0).contains(&probability_threshold))
    {
        panic!("learned selector model {path} has invalid probability threshold");
    }
}

fn validate_linear_policy_model(path: &str, model: &LinearPolicyModel) {
    if model.actions.is_empty() {
        panic!("linear agent model {path} has no actions");
    }
    for action in &model.actions {
        match action.as_str() {
            "static" | "adaptive" | "selector" => {}
            other => panic!("linear agent model {path} has unsupported action {other:?}"),
        }
        if !model.weights.contains_key(action) {
            panic!("linear agent model {path} missing weights for action {action:?}");
        }
        if let Some(intercept) = model.intercepts.get(action)
            && !intercept.is_finite()
        {
            panic!("linear agent model {path} has non-finite intercept for action {action:?}");
        }
    }
}

fn validate_bandit_policy_model(path: &str, model: &BanditPolicyModel) {
    if model.actions.is_empty() {
        panic!("bandit agent model {path} has no actions");
    }
    if !model.alpha.is_finite() || model.alpha < 0.0 {
        panic!("bandit agent model {path} has invalid alpha");
    }
    for action in &model.actions {
        match action.as_str() {
            "static" | "adaptive" | "selector" => {}
            other => panic!("bandit agent model {path} has unsupported action {other:?}"),
        }
        if !model.theta.contains_key(action) {
            panic!("bandit agent model {path} missing theta for action {action:?}");
        }
        let Some(inverse) = model.inverse_covariance.get(action) else {
            panic!("bandit agent model {path} missing inverse covariance for action {action:?}");
        };
        if inverse.len() != 6 || inverse.iter().any(|row| row.len() != 6) {
            panic!(
                "bandit agent model {path} has invalid inverse covariance for action {action:?}"
            );
        }
    }
}

fn dominant_policy_reason(contributions: [(PolicyReason, f64); 4]) -> PolicyReason {
    let mut best_reason = PolicyReason::None;
    let mut best_value = 0.0;
    let mut ties = 0;

    for (reason, value) in contributions {
        if value > best_value {
            best_reason = reason;
            best_value = value;
            ties = 1;
        } else if value == best_value && value > 0.0 {
            ties += 1;
        }
    }

    if ties > 1 {
        PolicyReason::Multiple
    } else {
        best_reason
    }
}

fn policy_reason_from_trigger(trigger: PaperPolicyTrigger) -> PolicyReason {
    match trigger {
        PaperPolicyTrigger::None => PolicyReason::None,
        PaperPolicyTrigger::Configured => PolicyReason::Configured,
        PaperPolicyTrigger::Inventory => PolicyReason::Inventory,
        PaperPolicyTrigger::Drawdown => PolicyReason::Drawdown,
        PaperPolicyTrigger::Volatility => PolicyReason::Volatility,
        PaperPolicyTrigger::Spread => PolicyReason::Spread,
        PaperPolicyTrigger::Multiple => PolicyReason::Multiple,
    }
}

fn paper_trigger_from_reason(reason: PolicyReason) -> PaperPolicyTrigger {
    match reason {
        PolicyReason::None | PolicyReason::ModelScore => PaperPolicyTrigger::None,
        PolicyReason::Configured => PaperPolicyTrigger::Configured,
        PolicyReason::Inventory => PaperPolicyTrigger::Inventory,
        PolicyReason::Drawdown => PaperPolicyTrigger::Drawdown,
        PolicyReason::Volatility => PaperPolicyTrigger::Volatility,
        PolicyReason::Spread => PaperPolicyTrigger::Spread,
        PolicyReason::Multiple => PaperPolicyTrigger::Multiple,
    }
}

impl From<PaperAdaptivePolicyConfig> for AdaptivePolicyParams {
    fn from(config: PaperAdaptivePolicyConfig) -> Self {
        Self {
            min_spread: config.min_spread,
            max_spread: config.max_spread,
            volatility_spread_multiplier: config.volatility_spread_multiplier,
            inventory_skew_multiplier: config.inventory_skew_multiplier,
            touch_spread_multiplier: config.touch_spread_multiplier,
        }
    }
}

impl From<PaperSelectorPolicyConfig> for SelectorPolicyParams {
    fn from(config: PaperSelectorPolicyConfig) -> Self {
        Self {
            volatility_weight: config.volatility_weight,
            spread_weight: config.spread_weight,
            inventory_weight: config.inventory_weight,
            drawdown_weight: config.drawdown_weight,
            activation_threshold: config.activation_threshold,
        }
    }
}

fn adaptive_policy_trigger(
    state: &SystemState,
    estimated_volatility: f64,
    current_drawdown: f64,
    drawdown_threshold: f64,
    inventory_threshold: f64,
    volatility_threshold: f64,
) -> PaperPolicyTrigger {
    let inventory = state.inventory.abs() >= inventory_threshold;
    let drawdown = current_drawdown >= drawdown_threshold;
    let volatility = estimated_volatility >= volatility_threshold;
    match (inventory, drawdown, volatility) {
        (false, false, false) => PaperPolicyTrigger::None,
        (true, false, false) => PaperPolicyTrigger::Inventory,
        (false, true, false) => PaperPolicyTrigger::Drawdown,
        (false, false, true) => PaperPolicyTrigger::Volatility,
        _ => PaperPolicyTrigger::Multiple,
    }
}

fn adaptive_quote(
    quote: Quote,
    state: &SystemState,
    observed_quote: Option<Quote>,
    estimated_volatility: f64,
    params: AdaptivePolicyParams,
) -> Quote {
    let mut spread = quote.spread().max(params.min_spread).max(
        observed_quote.map(|quote| quote.spread()).unwrap_or(0.0) * params.touch_spread_multiplier,
    ) + estimated_volatility * params.volatility_spread_multiplier;
    spread = spread.clamp(params.min_spread, params.max_spread);

    let center = (quote.bid + quote.ask) / 2.0 - state.inventory * params.inventory_skew_multiplier;

    Quote {
        bid: center - spread / 2.0,
        ask: center + spread / 2.0,
    }
}

pub fn paper_session_to_csv(result: &PaperSessionResult) -> String {
    let mut csv = String::from(paper_session_csv_header());
    csv.push('\n');

    for row in &result.rows {
        writeln!(csv, "{}", paper_session_row_to_csv(row))
            .expect("writing to a String should not fail");
    }

    csv
}

pub fn paper_session_csv_header() -> &'static str {
    "timestamp_ms,mid_price,observed_bid,observed_ask,estimated_volatility,regime,agent_mode,policy_agent,policy_action,policy_score,policy_mode,policy_trigger,bid,ask,spread,inventory,cash,pnl,drawdown,fills,buy_fills,sell_fills,fill_quantity,fill_notional,fees"
}

pub fn paper_session_row_to_csv(row: &PaperSessionRow) -> String {
    let values = vec![
        row.timestamp_ms.to_string(),
        format_f64(row.mid_price),
        optional_f64(row.observed_bid),
        optional_f64(row.observed_ask),
        format_f64(row.estimated_volatility),
        regime_name(row.regime).to_string(),
        controller_mode_name(&row.agent_mode).to_string(),
        policy_agent_name(row.policy_agent).to_string(),
        policy_action_name(row.policy_action).to_string(),
        optional_f64(row.policy_score),
        policy_mode_name(row.policy_mode).to_string(),
        policy_trigger_name(row.policy_trigger).to_string(),
        format_f64(row.bid),
        format_f64(row.ask),
        format_f64(row.ask - row.bid),
        format_f64(row.inventory),
        format_f64(row.cash),
        format_f64(row.pnl),
        format_f64(row.drawdown),
        row.fills.to_string(),
        row.buy_fills.to_string(),
        row.sell_fills.to_string(),
        format_f64(row.fill_quantity),
        format_f64(row.fill_notional),
        format_f64(row.fees),
    ];
    values.join(",")
}

struct SessionRowInput<'a> {
    event: MarketEvent,
    observed_quote: Option<Quote>,
    estimated_volatility: f64,
    regime: MarketRegime,
    agent_mode: ControllerMode,
    policy_agent: PolicyAgentKind,
    policy_action: PolicyAction,
    policy_score: Option<f64>,
    policy_mode: PaperPolicyMode,
    policy_trigger: PaperPolicyTrigger,
    quote: Quote,
    state: &'a SystemState,
    drawdown: f64,
    fills: &'a [Fill],
}

fn session_row(input: SessionRowInput<'_>) -> PaperSessionRow {
    let mut buy_fills = 0;
    let mut sell_fills = 0;
    let mut fill_quantity = 0.0;
    let mut fill_notional = 0.0;
    let mut fees = 0.0;

    for fill in input.fills {
        fill_quantity += fill.quantity;
        fill_notional += fill.notional();
        fees += fill.fee;

        match fill.side {
            crate::market::FillSide::Buy => buy_fills += 1,
            crate::market::FillSide::Sell => sell_fills += 1,
        }
    }

    PaperSessionRow {
        timestamp_ms: input.event.timestamp_ms,
        mid_price: input.event.mid_price,
        observed_bid: input.observed_quote.map(|quote| quote.bid),
        observed_ask: input.observed_quote.map(|quote| quote.ask),
        estimated_volatility: input.estimated_volatility,
        regime: input.regime,
        agent_mode: input.agent_mode,
        policy_agent: input.policy_agent,
        policy_action: input.policy_action,
        policy_score: input.policy_score,
        policy_mode: input.policy_mode,
        policy_trigger: input.policy_trigger,
        bid: input.quote.bid,
        ask: input.quote.ask,
        inventory: input.state.inventory,
        cash: input.state.cash,
        pnl: input.state.pnl,
        drawdown: input.drawdown,
        fills: input.fills.len(),
        buy_fills,
        sell_fills,
        fill_quantity,
        fill_notional,
        fees,
    }
}

fn event_quote(event: MarketEvent) -> Option<Quote> {
    Some(Quote {
        bid: event.bid?,
        ask: event.ask?,
    })
}

fn fee_aware_quote(quote: Quote, mid_price: f64, fee_rate: f64, multiplier: f64) -> Quote {
    if multiplier <= 0.0 || fee_rate <= 0.0 {
        return quote;
    }

    let break_even_spread = 2.0 * mid_price * fee_rate;
    let minimum_spread = multiplier * break_even_spread;
    if quote.spread() >= minimum_spread {
        return quote;
    }

    let center = (quote.bid + quote.ask) / 2.0;
    Quote {
        bid: center - minimum_spread / 2.0,
        ask: center + minimum_spread / 2.0,
    }
}

fn observed_quote_fills(
    quote: Quote,
    observed_quote: Quote,
    estimated_volatility: f64,
    fill_model: PaperFillModelConfig,
    quantity: f64,
    fee_rate: f64,
    rng: &mut StdRng,
) -> Vec<Fill> {
    match fill_model {
        PaperFillModelConfig::Crossing => {
            observed_quote_crossing_fills(quote, observed_quote, quantity, fee_rate)
        }
        PaperFillModelConfig::TouchIntensity {
            base_intensity,
            distance_decay,
            volatility_boost,
        } => observed_quote_touch_intensity_fills(
            quote,
            observed_quote,
            estimated_volatility,
            TouchIntensityParams {
                base_intensity,
                distance_decay,
                volatility_boost,
            },
            quantity,
            fee_rate,
            rng,
        ),
    }
}

fn observed_quote_crossing_fills(
    quote: Quote,
    observed_quote: Quote,
    quantity: f64,
    fee_rate: f64,
) -> Vec<Fill> {
    let mut fills = Vec::with_capacity(2);

    if quote.bid >= observed_quote.ask {
        let fee = quote.bid * quantity * fee_rate;
        fills.push(Fill::buy(quote.bid, quantity, fee));
    }

    if quote.ask <= observed_quote.bid {
        let fee = quote.ask * quantity * fee_rate;
        fills.push(Fill::sell(quote.ask, quantity, fee));
    }

    fills
}

#[derive(Debug, Clone, Copy)]
struct TouchIntensityParams {
    base_intensity: f64,
    distance_decay: f64,
    volatility_boost: f64,
}

fn observed_quote_touch_intensity_fills(
    quote: Quote,
    observed_quote: Quote,
    estimated_volatility: f64,
    params: TouchIntensityParams,
    quantity: f64,
    fee_rate: f64,
    rng: &mut StdRng,
) -> Vec<Fill> {
    let mut fills = observed_quote_crossing_fills(quote, observed_quote, quantity, fee_rate);

    if quote.bid < observed_quote.ask {
        let bid_distance = (observed_quote.bid - quote.bid).max(0.0);
        if rng.random::<f64>() < touch_fill_probability(bid_distance, estimated_volatility, params)
        {
            let fee = quote.bid * quantity * fee_rate;
            fills.push(Fill::buy(quote.bid, quantity, fee));
        }
    }

    if quote.ask > observed_quote.bid {
        let ask_distance = (quote.ask - observed_quote.ask).max(0.0);
        if rng.random::<f64>() < touch_fill_probability(ask_distance, estimated_volatility, params)
        {
            let fee = quote.ask * quantity * fee_rate;
            fills.push(Fill::sell(quote.ask, quantity, fee));
        }
    }

    fills
}

fn touch_fill_probability(
    quote_distance: f64,
    estimated_volatility: f64,
    params: TouchIntensityParams,
) -> f64 {
    let volatility_multiplier = (1.0 + params.volatility_boost * estimated_volatility).max(0.0);
    let probability = params.base_intensity
        * (-params.distance_decay * quote_distance).exp()
        * volatility_multiplier;

    probability.clamp(0.0, 1.0)
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

struct RollingLearnedFeatures {
    window: usize,
    samples: Vec<LearnedFeatureSample>,
}

impl RollingLearnedFeatures {
    fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            samples: Vec::with_capacity(window.max(1)),
        }
    }

    fn push(&mut self, sample: LearnedFeatureSample) {
        if self.samples.len() == self.window {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    fn snapshot(&self) -> Option<LearnedFeatureSnapshot> {
        if self.samples.is_empty() {
            return None;
        }
        let count = self.samples.len() as f64;
        Some(LearnedFeatureSnapshot {
            estimated_volatility: self
                .samples
                .iter()
                .map(|sample| sample.estimated_volatility)
                .sum::<f64>()
                / count,
            observed_spread: self
                .samples
                .iter()
                .map(|sample| sample.observed_spread)
                .sum::<f64>()
                / count,
            max_observed_spread: self
                .samples
                .iter()
                .map(|sample| sample.observed_spread)
                .fold(0.0, f64::max),
            abs_mid_move: self
                .samples
                .iter()
                .map(|sample| sample.abs_mid_move)
                .sum::<f64>()
                / count,
            abs_inventory: self
                .samples
                .iter()
                .map(|sample| sample.abs_inventory)
                .sum::<f64>()
                / count,
            drawdown: self
                .samples
                .iter()
                .map(|sample| sample.drawdown)
                .sum::<f64>()
                / count,
        })
    }
}

fn regime_name(regime: MarketRegime) -> &'static str {
    match regime {
        MarketRegime::LowVol => "LowVol",
        MarketRegime::NormalVol => "NormalVol",
        MarketRegime::HighVol => "HighVol",
    }
}

fn controller_mode_name(mode: &ControllerMode) -> &'static str {
    match mode {
        ControllerMode::FixedSpread => "FixedSpread",
        ControllerMode::RiskManaged => "RiskManaged",
    }
}

fn policy_mode_name(mode: PaperPolicyMode) -> &'static str {
    match mode {
        PaperPolicyMode::Static => "static",
        PaperPolicyMode::Adaptive => "adaptive",
    }
}

fn policy_agent_name(agent: PolicyAgentKind) -> &'static str {
    match agent {
        PolicyAgentKind::Static => "static",
        PolicyAgentKind::Adaptive => "adaptive",
        PolicyAgentKind::Hybrid => "hybrid",
        PolicyAgentKind::Selector => "selector",
        PolicyAgentKind::LearnedSelector => "learned_selector",
        PolicyAgentKind::LinearAgent => "linear_agent",
        PolicyAgentKind::BanditAgent => "bandit_agent",
    }
}

fn policy_action_name(action: PolicyAction) -> &'static str {
    match action {
        PolicyAction::Static => "static",
        PolicyAction::Adaptive => "adaptive",
        PolicyAction::Selector => "selector",
    }
}

fn policy_trigger_name(trigger: PaperPolicyTrigger) -> &'static str {
    match trigger {
        PaperPolicyTrigger::None => "none",
        PaperPolicyTrigger::Configured => "configured",
        PaperPolicyTrigger::Inventory => "inventory",
        PaperPolicyTrigger::Drawdown => "drawdown",
        PaperPolicyTrigger::Volatility => "volatility",
        PaperPolicyTrigger::Spread => "spread",
        PaperPolicyTrigger::Multiple => "multiple",
    }
}

fn format_f64(value: f64) -> String {
    format!("{value:.6}")
}

fn optional_f64(value: Option<f64>) -> String {
    value.map(format_f64).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::avellaneda_stoikov::AvellanedaStoikovParams;
    use crate::strategy::market_maker::StrategyParams;

    fn agent() -> RuleBasedControllerParams {
        RuleBasedControllerParams {
            fixed_spread: StrategyParams {
                spread: 0.2,
                skew_coeff: 0.0,
            },
            risk_managed: AvellanedaStoikovParams {
                risk_aversion: 0.2,
                liquidity_depth: 4.0,
                horizon: 10.0,
                min_spread: 0.1,
            },
            inventory_limit: 2.0,
        }
    }

    fn paper_policy_input<'a>(
        quote: Quote,
        state: &'a SystemState,
        observed_quote: Option<Quote>,
        estimated_volatility: f64,
        current_drawdown: f64,
    ) -> PaperPolicyInput<'a> {
        PaperPolicyInput {
            quote,
            state,
            observed_quote,
            estimated_volatility,
            current_drawdown,
            observation: AgentObservation {
                estimated_volatility,
                observed_spread: observed_quote.map(|quote| quote.spread()).unwrap_or(0.0),
                max_observed_spread: observed_quote.map(|quote| quote.spread()).unwrap_or(0.0),
                abs_mid_move: 0.0,
                abs_inventory: state.inventory.abs(),
                drawdown: current_drawdown,
            },
        }
    }

    #[test]
    fn paper_session_records_agent_decisions() {
        let events = vec![
            MarketEvent::from_quote(1, 99.95, 100.05),
            MarketEvent::from_quote(2, 100.00, 100.10),
        ];

        let result = run_paper_session(
            &events,
            &agent(),
            PaperSessionConfig {
                order_quantity: 1.0,
                fee_rate: 0.001,
                volatility_window: 10,
                ..PaperSessionConfig::default()
            },
        );

        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0].agent_mode, ControllerMode::FixedSpread);
        assert_eq!(result.rows[0].policy_agent, PolicyAgentKind::Static);
        assert_eq!(result.rows[0].policy_action, PolicyAction::Static);
        assert_eq!(result.rows[0].policy_score, None);
        assert_eq!(result.rows[0].policy_mode, PaperPolicyMode::Static);
        assert_eq!(result.rows[0].policy_trigger, PaperPolicyTrigger::None);
        assert_eq!(result.rows[0].observed_bid, Some(99.95));
        assert_eq!(result.rows[0].observed_ask, Some(100.05));
    }

    #[test]
    fn paper_session_does_not_fill_passive_quotes() {
        let events = vec![MarketEvent::from_quote(1, 100.01, 100.02)];
        let result = run_paper_session(
            &events,
            &agent(),
            PaperSessionConfig {
                order_quantity: 1.0,
                fee_rate: 0.001,
                ..PaperSessionConfig::default()
            },
        );

        assert_eq!(result.rows[0].fills, 0);
        assert_eq!(result.rows[0].buy_fills, 0);
        assert_eq!(result.rows[0].sell_fills, 0);
        assert_eq!(result.rows[0].fees, 0.0);
    }

    #[test]
    fn paper_session_touch_intensity_can_fill_near_touch() {
        let events = vec![MarketEvent::from_quote(1, 99.95, 100.05)];
        let result = run_paper_session(
            &events,
            &agent(),
            PaperSessionConfig {
                order_quantity: 1.0,
                fee_rate: 0.001,
                fill_model: PaperFillModelConfig::TouchIntensity {
                    base_intensity: 1.0,
                    distance_decay: 0.0,
                    volatility_boost: 0.0,
                },
                ..PaperSessionConfig::default()
            },
        );

        assert_eq!(result.rows[0].fills, 2);
        assert_eq!(result.rows[0].buy_fills, 1);
        assert_eq!(result.rows[0].sell_fills, 1);
        assert!(result.rows[0].fees > 0.0);
    }

    #[test]
    fn paper_session_exports_csv() {
        let events = vec![MarketEvent::from_quote(1, 99.95, 100.05)];
        let result = run_paper_session(&events, &agent(), PaperSessionConfig::default());
        let csv = paper_session_to_csv(&result);

        assert!(csv.starts_with("timestamp_ms,mid_price"));
        assert!(csv.contains(",FixedSpread,"));
        assert_eq!(csv.lines().count(), 2);
    }

    #[test]
    fn paper_session_can_enforce_fee_aware_spread_floor() {
        let events = vec![MarketEvent::from_quote(1, 99.95, 100.05)];
        let result = run_paper_session(
            &events,
            &agent(),
            PaperSessionConfig {
                fee_rate: 0.001,
                fee_spread_multiplier: 1.5,
                ..PaperSessionConfig::default()
            },
        );

        let row = &result.rows[0];
        let expected_spread = 2.0 * row.mid_price * 0.001 * 1.5;
        assert!((row.ask - row.bid - expected_spread).abs() < 1e-12);
    }

    #[test]
    fn adaptive_policy_widens_toward_observed_spread() {
        let state = SystemState::new(100.0);
        let decision = apply_paper_policy(
            paper_policy_input(
                Quote {
                    bid: 99.99,
                    ask: 100.01,
                },
                &state,
                Some(Quote {
                    bid: 99.90,
                    ask: 100.10,
                }),
                0.0,
                0.0,
            ),
            &PaperPolicyConfig::Adaptive {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 0.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
            },
            None,
            None,
            None,
            None,
        );

        assert_eq!(decision.mode, PaperPolicyMode::Adaptive);
        assert_eq!(decision.trigger, PaperPolicyTrigger::Configured);
        assert!((decision.quote.spread() - 0.10).abs() < 1e-12);
    }

    #[test]
    fn adaptive_policy_skews_away_from_long_inventory() {
        let mut state = SystemState::new(100.0);
        state.inventory = 2.0;
        let decision = apply_paper_policy(
            paper_policy_input(
                Quote {
                    bid: 99.90,
                    ask: 100.10,
                },
                &state,
                None,
                0.0,
                0.0,
            ),
            &PaperPolicyConfig::Adaptive {
                min_spread: 0.20,
                max_spread: 0.20,
                volatility_spread_multiplier: 0.0,
                inventory_skew_multiplier: 0.10,
                touch_spread_multiplier: 0.0,
            },
            None,
            None,
            None,
            None,
        );

        assert!((decision.quote.bid - 99.70).abs() < 1e-12);
        assert!((decision.quote.ask - 99.90).abs() < 1e-12);
    }

    #[test]
    fn hybrid_policy_keeps_static_quote_when_risk_is_low() {
        let state = SystemState::new(100.0);
        let input = Quote {
            bid: 99.99,
            ask: 100.01,
        };
        let decision = apply_paper_policy(
            paper_policy_input(
                input,
                &state,
                Some(Quote {
                    bid: 99.90,
                    ask: 100.10,
                }),
                0.001,
                0.0,
            ),
            &PaperPolicyConfig::Hybrid {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.10,
                touch_spread_multiplier: 0.5,
                drawdown_threshold: 1.0,
                inventory_threshold: 10.0,
                volatility_threshold: 1.0,
            },
            None,
            None,
            None,
            None,
        );

        assert_eq!(decision.quote, input);
        assert_eq!(decision.mode, PaperPolicyMode::Static);
        assert_eq!(decision.trigger, PaperPolicyTrigger::None);
    }

    #[test]
    fn hybrid_policy_uses_adaptive_quote_when_risk_trigger_fires() {
        let state = SystemState::new(100.0);
        let decision = apply_paper_policy(
            paper_policy_input(
                Quote {
                    bid: 99.99,
                    ask: 100.01,
                },
                &state,
                Some(Quote {
                    bid: 99.90,
                    ask: 100.10,
                }),
                0.02,
                0.0,
            ),
            &PaperPolicyConfig::Hybrid {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
                drawdown_threshold: 1.0,
                inventory_threshold: 10.0,
                volatility_threshold: 0.01,
            },
            None,
            None,
            None,
            None,
        );

        assert_eq!(decision.mode, PaperPolicyMode::Adaptive);
        assert_eq!(decision.trigger, PaperPolicyTrigger::Volatility);
        assert!((decision.quote.spread() - 0.12).abs() < 1e-12);
    }

    #[test]
    fn selector_policy_keeps_static_quote_below_activation_threshold() {
        let state = SystemState::new(100.0);
        let input = Quote {
            bid: 99.99,
            ask: 100.01,
        };
        let decision = apply_paper_policy(
            paper_policy_input(
                input,
                &state,
                Some(Quote {
                    bid: 99.99,
                    ask: 100.01,
                }),
                0.001,
                0.0,
            ),
            &PaperPolicyConfig::Selector {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
                volatility_weight: 1.0,
                spread_weight: 1.0,
                inventory_weight: 1.0,
                drawdown_weight: 1.0,
                activation_threshold: 1.0,
            },
            None,
            None,
            None,
            None,
        );

        assert_eq!(decision.quote, input);
        assert_eq!(decision.mode, PaperPolicyMode::Static);
        assert_eq!(decision.trigger, PaperPolicyTrigger::None);
    }

    #[test]
    fn selector_policy_uses_adaptive_quote_above_activation_threshold() {
        let state = SystemState::new(100.0);
        let decision = apply_paper_policy(
            paper_policy_input(
                Quote {
                    bid: 99.99,
                    ask: 100.01,
                },
                &state,
                Some(Quote {
                    bid: 99.90,
                    ask: 100.10,
                }),
                0.01,
                0.0,
            ),
            &PaperPolicyConfig::Selector {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
                volatility_weight: 20.0,
                spread_weight: 1.0,
                inventory_weight: 0.0,
                drawdown_weight: 0.0,
                activation_threshold: 0.10,
            },
            None,
            None,
            None,
            None,
        );

        assert_eq!(decision.mode, PaperPolicyMode::Adaptive);
        assert_eq!(decision.trigger, PaperPolicyTrigger::Volatility);
        assert_eq!(decision.agent.agent, PolicyAgentKind::Selector);
        assert_eq!(decision.agent.action, PolicyAction::Adaptive);
        assert!(decision.agent.score.unwrap_or_default() >= 0.10);
        assert!((decision.quote.spread() - 0.11).abs() < 1e-12);
    }

    #[test]
    fn learned_selector_uses_model_score_to_choose_selector() {
        let state = SystemState::new(100.0);
        let model = LearnedPolicyModel {
            action_on: "selector".to_string(),
            action_off: "adaptive".to_string(),
            intercept: 0.0,
            threshold: 0.0,
            probability_threshold: Some(0.5),
            feature_window: Some(30),
            weights: LearnedFeatureWeights {
                estimated_volatility: 0.0,
                observed_spread: 0.0,
                max_observed_spread: 0.0,
                abs_mid_move: 0.0,
                abs_inventory: 1.0,
                drawdown: 0.0,
            },
            feature_scale: LearnedFeatureScale {
                estimated_volatility: 1.0,
                observed_spread: 1.0,
                max_observed_spread: 1.0,
                abs_mid_move: 1.0,
                abs_inventory: 1.0,
                drawdown: 1.0,
            },
        };
        let decision = learned_selector_policy_decision(
            paper_policy_input(
                Quote {
                    bid: 99.99,
                    ask: 100.01,
                },
                &state,
                Some(Quote {
                    bid: 99.90,
                    ask: 100.10,
                }),
                0.01,
                0.0,
            ),
            PaperAdaptivePolicyConfig {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
            },
            PaperSelectorPolicyConfig {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
                volatility_weight: 20.0,
                spread_weight: 1.0,
                inventory_weight: 0.0,
                drawdown_weight: 0.0,
                activation_threshold: 0.10,
            },
            &model,
            LearnedFeatureSnapshot {
                estimated_volatility: 0.0,
                observed_spread: 0.0,
                max_observed_spread: 0.0,
                abs_mid_move: 0.0,
                abs_inventory: 1.0,
                drawdown: 0.0,
            },
        );

        assert_eq!(decision.mode, PaperPolicyMode::Adaptive);
        assert_eq!(decision.trigger, PaperPolicyTrigger::Volatility);
        assert_eq!(decision.agent.agent, PolicyAgentKind::LearnedSelector);
        assert_eq!(decision.agent.action, PolicyAction::Selector);
        assert!(decision.agent.score.unwrap_or_default() >= 0.0);
    }

    #[test]
    fn linear_agent_uses_highest_scored_action() {
        let state = SystemState::new(100.0);
        let model = LinearPolicyModel {
            actions: vec![
                "static".to_string(),
                "adaptive".to_string(),
                "selector".to_string(),
            ],
            intercepts: BTreeMap::from([
                ("static".to_string(), 0.0),
                ("adaptive".to_string(), 0.1),
                ("selector".to_string(), 0.2),
            ]),
            weights: BTreeMap::from([
                (
                    "static".to_string(),
                    LearnedFeatureWeights {
                        estimated_volatility: 0.0,
                        observed_spread: 0.0,
                        max_observed_spread: 0.0,
                        abs_mid_move: 0.0,
                        abs_inventory: 0.0,
                        drawdown: 0.0,
                    },
                ),
                (
                    "adaptive".to_string(),
                    LearnedFeatureWeights {
                        estimated_volatility: 0.0,
                        observed_spread: 0.0,
                        max_observed_spread: 0.0,
                        abs_mid_move: 0.0,
                        abs_inventory: 0.0,
                        drawdown: 0.0,
                    },
                ),
                (
                    "selector".to_string(),
                    LearnedFeatureWeights {
                        estimated_volatility: 0.0,
                        observed_spread: 0.0,
                        max_observed_spread: 0.0,
                        abs_mid_move: 0.0,
                        abs_inventory: 1.0,
                        drawdown: 0.0,
                    },
                ),
            ]),
            feature_scale: LearnedFeatureScale {
                estimated_volatility: 1.0,
                observed_spread: 1.0,
                max_observed_spread: 1.0,
                abs_mid_move: 1.0,
                abs_inventory: 1.0,
                drawdown: 1.0,
            },
            feature_window: Some(30),
        };
        let decision = linear_agent_policy_decision(
            paper_policy_input(
                Quote {
                    bid: 99.99,
                    ask: 100.01,
                },
                &state,
                Some(Quote {
                    bid: 99.90,
                    ask: 100.10,
                }),
                0.01,
                0.0,
            ),
            PaperAdaptivePolicyConfig {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
            },
            PaperSelectorPolicyConfig {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
                volatility_weight: 20.0,
                spread_weight: 1.0,
                inventory_weight: 0.0,
                drawdown_weight: 0.0,
                activation_threshold: 0.10,
            },
            &model,
            LearnedFeatureSnapshot {
                estimated_volatility: 0.0,
                observed_spread: 0.0,
                max_observed_spread: 0.0,
                abs_mid_move: 0.0,
                abs_inventory: 1.0,
                drawdown: 0.0,
            },
        );

        assert_eq!(decision.agent.agent, PolicyAgentKind::LinearAgent);
        assert_eq!(decision.agent.action, PolicyAction::Selector);
        assert_eq!(decision.mode, PaperPolicyMode::Adaptive);
        assert_eq!(decision.trigger, PaperPolicyTrigger::Volatility);
        assert!(decision.agent.score.unwrap_or_default() > 1.0);
    }

    #[test]
    fn bandit_agent_uses_ucb_score_to_choose_action() {
        let state = SystemState::new(100.0);
        let identity = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let model = BanditPolicyModel {
            actions: vec![
                "static".to_string(),
                "adaptive".to_string(),
                "selector".to_string(),
            ],
            alpha: 0.0,
            theta: BTreeMap::from([
                (
                    "static".to_string(),
                    LearnedFeatureWeights {
                        estimated_volatility: 0.0,
                        observed_spread: 0.0,
                        max_observed_spread: 0.0,
                        abs_mid_move: 0.0,
                        abs_inventory: 0.0,
                        drawdown: 0.0,
                    },
                ),
                (
                    "adaptive".to_string(),
                    LearnedFeatureWeights {
                        estimated_volatility: 0.0,
                        observed_spread: 0.0,
                        max_observed_spread: 0.0,
                        abs_mid_move: 0.0,
                        abs_inventory: 0.0,
                        drawdown: 0.0,
                    },
                ),
                (
                    "selector".to_string(),
                    LearnedFeatureWeights {
                        estimated_volatility: 0.0,
                        observed_spread: 0.0,
                        max_observed_spread: 0.0,
                        abs_mid_move: 0.0,
                        abs_inventory: 2.0,
                        drawdown: 0.0,
                    },
                ),
            ]),
            inverse_covariance: BTreeMap::from([
                ("static".to_string(), identity.clone()),
                ("adaptive".to_string(), identity.clone()),
                ("selector".to_string(), identity),
            ]),
            feature_scale: LearnedFeatureScale {
                estimated_volatility: 1.0,
                observed_spread: 1.0,
                max_observed_spread: 1.0,
                abs_mid_move: 1.0,
                abs_inventory: 1.0,
                drawdown: 1.0,
            },
            feature_window: Some(30),
        };
        let decision = bandit_agent_policy_decision(
            paper_policy_input(
                Quote {
                    bid: 99.99,
                    ask: 100.01,
                },
                &state,
                Some(Quote {
                    bid: 99.90,
                    ask: 100.10,
                }),
                0.01,
                0.0,
            ),
            PaperAdaptivePolicyConfig {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
            },
            PaperSelectorPolicyConfig {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
                volatility_weight: 20.0,
                spread_weight: 1.0,
                inventory_weight: 0.0,
                drawdown_weight: 0.0,
                activation_threshold: 0.10,
            },
            &model,
            LearnedFeatureSnapshot {
                estimated_volatility: 0.0,
                observed_spread: 0.0,
                max_observed_spread: 0.0,
                abs_mid_move: 0.0,
                abs_inventory: 1.0,
                drawdown: 0.0,
            },
        );

        assert_eq!(decision.agent.agent, PolicyAgentKind::BanditAgent);
        assert_eq!(decision.agent.action, PolicyAction::Selector);
        assert_eq!(decision.mode, PaperPolicyMode::Adaptive);
        assert_eq!(decision.trigger, PaperPolicyTrigger::Volatility);
        assert!(decision.agent.score.unwrap_or_default() > 1.0);
    }

    #[test]
    fn paper_runner_keeps_state_between_steps() {
        let mut runner = PaperSessionRunner::new(PaperSessionConfig {
            order_quantity: 1.0,
            fee_rate: 0.001,
            fill_model: PaperFillModelConfig::TouchIntensity {
                base_intensity: 1.0,
                distance_decay: 0.0,
                volatility_boost: 0.0,
            },
            ..PaperSessionConfig::default()
        });

        let first = runner.step(MarketEvent::from_quote(1, 99.95, 100.05), &agent());
        let second = runner.step(MarketEvent::from_quote(2, 100.00, 100.10), &agent());

        assert_eq!(first.fills, 2);
        assert_eq!(second.fills, 2);
        assert!(second.fees > first.fees);
    }
}

use std::fmt::Write;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::agent::{ControllerMode, MarketMakingAgent, RuleBasedControllerParams};
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PaperPolicyConfig {
    #[serde(rename = "static")]
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
}

impl Default for PaperPolicyConfig {
    fn default() -> Self {
        Self::Static
    }
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
    peak_pnl: f64,
    rng: StdRng,
}

impl PaperSessionRunner {
    pub fn new(config: PaperSessionConfig) -> Self {
        Self {
            volatility: RollingVolatility::new(config.volatility_window),
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
        let policy_decision = apply_paper_policy(
            decision.quote,
            state,
            observed_quote,
            estimated_volatility,
            current_drawdown,
            self.config.policy,
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
    quote: Quote,
    state: &SystemState,
    observed_quote: Option<Quote>,
    estimated_volatility: f64,
    current_drawdown: f64,
    policy: PaperPolicyConfig,
) -> PaperPolicyDecision {
    match policy {
        PaperPolicyConfig::Static => PaperPolicyDecision {
            quote,
            mode: PaperPolicyMode::Static,
            trigger: PaperPolicyTrigger::None,
        },
        PaperPolicyConfig::Adaptive {
            min_spread,
            max_spread,
            volatility_spread_multiplier,
            inventory_skew_multiplier,
            touch_spread_multiplier,
        } => PaperPolicyDecision {
            quote: adaptive_quote(
                quote,
                state,
                observed_quote,
                estimated_volatility,
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
                state,
                estimated_volatility,
                current_drawdown,
                drawdown_threshold,
                inventory_threshold,
                volatility_threshold,
            );
            if trigger == PaperPolicyTrigger::None {
                PaperPolicyDecision {
                    quote,
                    mode: PaperPolicyMode::Static,
                    trigger,
                }
            } else {
                PaperPolicyDecision {
                    quote: adaptive_quote(
                        quote,
                        state,
                        observed_quote,
                        estimated_volatility,
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
            quote,
            state,
            observed_quote,
            estimated_volatility,
            current_drawdown,
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
    }
}

#[derive(Debug, Clone, Copy)]
struct PaperPolicyDecision {
    quote: Quote,
    mode: PaperPolicyMode,
    trigger: PaperPolicyTrigger,
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

fn selector_policy_decision(
    quote: Quote,
    state: &SystemState,
    observed_quote: Option<Quote>,
    estimated_volatility: f64,
    current_drawdown: f64,
    adaptive_params: AdaptivePolicyParams,
    selector_params: SelectorPolicyParams,
) -> PaperPolicyDecision {
    let spread = observed_quote
        .map(|quote| quote.spread())
        .unwrap_or(quote.spread());
    let contributions = [
        (
            PaperPolicyTrigger::Volatility,
            estimated_volatility * selector_params.volatility_weight,
        ),
        (
            PaperPolicyTrigger::Spread,
            spread * selector_params.spread_weight,
        ),
        (
            PaperPolicyTrigger::Inventory,
            state.inventory.abs() * selector_params.inventory_weight,
        ),
        (
            PaperPolicyTrigger::Drawdown,
            current_drawdown * selector_params.drawdown_weight,
        ),
    ];
    let score = contributions
        .iter()
        .map(|(_trigger, contribution)| contribution)
        .sum::<f64>();

    if score < selector_params.activation_threshold {
        return PaperPolicyDecision {
            quote,
            mode: PaperPolicyMode::Static,
            trigger: PaperPolicyTrigger::None,
        };
    }

    let trigger = dominant_selector_trigger(contributions);
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
    }
}

fn dominant_selector_trigger(contributions: [(PaperPolicyTrigger, f64); 4]) -> PaperPolicyTrigger {
    let mut best_trigger = PaperPolicyTrigger::None;
    let mut best_value = 0.0;
    let mut ties = 0;

    for (trigger, value) in contributions {
        if value > best_value {
            best_trigger = trigger;
            best_value = value;
            ties = 1;
        } else if value == best_value && value > 0.0 {
            ties += 1;
        }
    }

    if ties > 1 {
        PaperPolicyTrigger::Multiple
    } else {
        best_trigger
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
    "timestamp_ms,mid_price,observed_bid,observed_ask,estimated_volatility,regime,agent_mode,policy_mode,policy_trigger,bid,ask,spread,inventory,cash,pnl,drawdown,fills,buy_fills,sell_fills,fill_quantity,fill_notional,fees"
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
            PaperPolicyConfig::Adaptive {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 0.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
            },
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
            Quote {
                bid: 99.90,
                ask: 100.10,
            },
            &state,
            None,
            0.0,
            0.0,
            PaperPolicyConfig::Adaptive {
                min_spread: 0.20,
                max_spread: 0.20,
                volatility_spread_multiplier: 0.0,
                inventory_skew_multiplier: 0.10,
                touch_spread_multiplier: 0.0,
            },
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
            input,
            &state,
            Some(Quote {
                bid: 99.90,
                ask: 100.10,
            }),
            0.001,
            0.0,
            PaperPolicyConfig::Hybrid {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.10,
                touch_spread_multiplier: 0.5,
                drawdown_threshold: 1.0,
                inventory_threshold: 10.0,
                volatility_threshold: 1.0,
            },
        );

        assert_eq!(decision.quote, input);
        assert_eq!(decision.mode, PaperPolicyMode::Static);
        assert_eq!(decision.trigger, PaperPolicyTrigger::None);
    }

    #[test]
    fn hybrid_policy_uses_adaptive_quote_when_risk_trigger_fires() {
        let state = SystemState::new(100.0);
        let decision = apply_paper_policy(
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
            PaperPolicyConfig::Hybrid {
                min_spread: 0.02,
                max_spread: 0.20,
                volatility_spread_multiplier: 1.0,
                inventory_skew_multiplier: 0.0,
                touch_spread_multiplier: 0.5,
                drawdown_threshold: 1.0,
                inventory_threshold: 10.0,
                volatility_threshold: 0.01,
            },
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
            input,
            &state,
            Some(Quote {
                bid: 99.99,
                ask: 100.01,
            }),
            0.001,
            0.0,
            PaperPolicyConfig::Selector {
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
        );

        assert_eq!(decision.quote, input);
        assert_eq!(decision.mode, PaperPolicyMode::Static);
        assert_eq!(decision.trigger, PaperPolicyTrigger::None);
    }

    #[test]
    fn selector_policy_uses_adaptive_quote_above_activation_threshold() {
        let state = SystemState::new(100.0);
        let decision = apply_paper_policy(
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
            PaperPolicyConfig::Selector {
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
        );

        assert_eq!(decision.mode, PaperPolicyMode::Adaptive);
        assert_eq!(decision.trigger, PaperPolicyTrigger::Volatility);
        assert!((decision.quote.spread() - 0.11).abs() < 1e-12);
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

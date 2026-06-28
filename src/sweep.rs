use serde::{Deserialize, Serialize};

use std::fmt::Write;

use crate::engine::metrics::SimulationMetrics;
use crate::engine::simulation::{MarketRegime, SimulationConfig};
use crate::experiment::{Experiment, ExperimentReport};
use crate::strategy::QuoteStrategy;
use crate::strategy::inventory_risk::InventoryRiskParams;
use crate::strategy::market_maker::StrategyParams;
use crate::strategy::regime_adaptive::RegimeAdaptiveVolatilityAwareParams;
use crate::strategy::volatility_aware::VolatilityAwareParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    #[serde(default)]
    pub name: Option<String>,
    pub simulation: SimulationConfig,
    #[serde(default)]
    pub seeds: Vec<u64>,
    pub strategy: StrategySweepConfig,
    pub scoring: ScoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StrategySweepConfig {
    #[serde(rename = "fixed_spread")]
    FixedSpread {
        spreads: Vec<f64>,
        skew_coeffs: Vec<f64>,
    },
    #[serde(rename = "volatility_aware")]
    VolatilityAware {
        base_spreads: Vec<f64>,
        volatility_coeffs: Vec<f64>,
        skew_coeffs: Vec<f64>,
    },
    #[serde(rename = "inventory_risk")]
    InventoryRisk {
        base_spreads: Vec<f64>,
        volatility_coeffs: Vec<f64>,
        risk_aversions: Vec<f64>,
    },
    #[serde(rename = "regime_adaptive_volatility_aware")]
    RegimeAdaptiveVolatilityAware {
        low_vol: Vec<VolatilityAwareParams>,
        normal_vol: Vec<VolatilityAwareParams>,
        high_vol: Vec<VolatilityAwareParams>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub name: String,
    pub strategy: SweepStrategyParams,
    pub runs: usize,
    pub metrics: SweepMetrics,
    pub stability: SweepStability,
    pub inactivity_penalty: f64,
    pub score: f64,
    pub stable_score: f64,
    #[serde(skip)]
    seed_metrics: Vec<SimulationMetrics>,
}

impl SweepResult {
    pub fn representative_spread(&self) -> f64 {
        self.strategy
            .spread_for_regime(self.dominant_regime())
            .unwrap_or_else(|| self.strategy.primary_spread())
    }

    pub fn representative_volatility_coeff(&self) -> Option<f64> {
        self.strategy
            .volatility_coeff_for_regime(self.dominant_regime())
            .or_else(|| self.strategy.volatility_coeff())
    }

    pub fn representative_skew_coeff(&self) -> Option<f64> {
        self.strategy
            .skew_coeff_for_regime(self.dominant_regime())
            .or_else(|| self.strategy.skew_coeff())
    }

    fn dominant_regime(&self) -> MarketRegime {
        if self.metrics.high_vol_steps >= self.metrics.normal_vol_steps
            && self.metrics.high_vol_steps >= self.metrics.low_vol_steps
        {
            MarketRegime::HighVol
        } else if self.metrics.normal_vol_steps >= self.metrics.low_vol_steps {
            MarketRegime::NormalVol
        } else {
            MarketRegime::LowVol
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SweepStrategyParams {
    FixedSpread {
        spread: f64,
        skew_coeff: f64,
    },
    VolatilityAware {
        base_spread: f64,
        volatility_coeff: f64,
        skew_coeff: f64,
    },
    InventoryRisk {
        base_spread: f64,
        volatility_coeff: f64,
        risk_aversion: f64,
    },
    RegimeAdaptiveVolatilityAware {
        low_vol: VolatilityAwareParams,
        normal_vol: VolatilityAwareParams,
        high_vol: VolatilityAwareParams,
    },
}

impl SweepStrategyParams {
    pub fn primary_spread(&self) -> f64 {
        match self {
            Self::FixedSpread { spread, .. } => *spread,
            Self::VolatilityAware { base_spread, .. } | Self::InventoryRisk { base_spread, .. } => {
                *base_spread
            }
            Self::RegimeAdaptiveVolatilityAware { normal_vol, .. } => normal_vol.base_spread,
        }
    }

    fn spread_for_regime(&self, regime: MarketRegime) -> Option<f64> {
        self.params_for_regime(regime)
            .map(|params| params.base_spread)
    }

    pub fn skew_coeff(&self) -> Option<f64> {
        match self {
            Self::FixedSpread { skew_coeff, .. } | Self::VolatilityAware { skew_coeff, .. } => {
                Some(*skew_coeff)
            }
            Self::InventoryRisk { .. } => None,
            Self::RegimeAdaptiveVolatilityAware { normal_vol, .. } => Some(normal_vol.skew_coeff),
        }
    }

    fn skew_coeff_for_regime(&self, regime: MarketRegime) -> Option<f64> {
        self.params_for_regime(regime)
            .map(|params| params.skew_coeff)
    }

    pub fn volatility_coeff(&self) -> Option<f64> {
        match self {
            Self::FixedSpread { .. } => None,
            Self::VolatilityAware {
                volatility_coeff, ..
            }
            | Self::InventoryRisk {
                volatility_coeff, ..
            } => Some(*volatility_coeff),
            Self::RegimeAdaptiveVolatilityAware { normal_vol, .. } => {
                Some(normal_vol.volatility_coeff)
            }
        }
    }

    fn volatility_coeff_for_regime(&self, regime: MarketRegime) -> Option<f64> {
        self.params_for_regime(regime)
            .map(|params| params.volatility_coeff)
    }

    pub fn risk_aversion(&self) -> Option<f64> {
        match self {
            Self::InventoryRisk { risk_aversion, .. } => Some(*risk_aversion),
            Self::FixedSpread { .. }
            | Self::VolatilityAware { .. }
            | Self::RegimeAdaptiveVolatilityAware { .. } => None,
        }
    }

    pub fn strategy_type(&self) -> &'static str {
        match self {
            Self::FixedSpread { .. } => "fixed_spread",
            Self::VolatilityAware { .. } => "volatility_aware",
            Self::InventoryRisk { .. } => "inventory_risk",
            Self::RegimeAdaptiveVolatilityAware { .. } => "regime_adaptive_volatility_aware",
        }
    }

    fn params_for_regime(&self, regime: MarketRegime) -> Option<&VolatilityAwareParams> {
        match self {
            Self::RegimeAdaptiveVolatilityAware {
                low_vol,
                normal_vol,
                high_vol,
            } => match regime {
                MarketRegime::LowVol => Some(low_vol),
                MarketRegime::NormalVol => Some(normal_vol),
                MarketRegime::HighVol => Some(high_vol),
            },
            Self::FixedSpread { .. }
            | Self::VolatilityAware { .. }
            | Self::InventoryRisk { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SweepMetrics {
    pub final_pnl: f64,
    pub min_pnl: f64,
    pub max_pnl: f64,
    pub max_drawdown: f64,
    pub final_inventory: f64,
    pub max_abs_inventory: f64,
    pub avg_abs_inventory: f64,
    pub total_fills: f64,
    pub buy_fills: f64,
    pub sell_fills: f64,
    pub traded_quantity: f64,
    pub traded_notional: f64,
    pub total_fees: f64,
    pub total_adverse_selection: f64,
    pub low_vol_steps: f64,
    pub normal_vol_steps: f64,
    pub high_vol_steps: f64,
    pub low_vol_fills: f64,
    pub normal_vol_fills: f64,
    pub high_vol_fills: f64,
    pub low_vol_fees: f64,
    pub normal_vol_fees: f64,
    pub high_vol_fees: f64,
    pub low_vol_adverse_selection: f64,
    pub normal_vol_adverse_selection: f64,
    pub high_vol_adverse_selection: f64,
    pub low_vol_avg_abs_inventory: f64,
    pub normal_vol_avg_abs_inventory: f64,
    pub high_vol_avg_abs_inventory: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SweepStability {
    pub score_std: f64,
    pub final_pnl_std: f64,
    pub max_drawdown_std: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub drawdown_weight: f64,
    pub inventory_weight: f64,
    pub min_fills: usize,
    pub missing_fill_penalty: f64,
    #[serde(default = "default_stability_weight")]
    pub stability_weight: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            drawdown_weight: 2.0,
            inventory_weight: 1.0,
            min_fills: 50,
            missing_fill_penalty: 0.25,
            stability_weight: default_stability_weight(),
        }
    }
}

fn default_stability_weight() -> f64 {
    0.25
}

pub fn run_parameter_sweep(config: SweepConfig) -> Vec<SweepResult> {
    let scoring = config.scoring;
    let seeds = sweep_seeds(&config);
    let mut results = match &config.strategy {
        StrategySweepConfig::FixedSpread {
            spreads,
            skew_coeffs,
        } => run_fixed_spread_sweep(&config.simulation, &seeds, spreads, skew_coeffs),
        StrategySweepConfig::VolatilityAware {
            base_spreads,
            volatility_coeffs,
            skew_coeffs,
        } => run_volatility_aware_sweep(
            &config.simulation,
            &seeds,
            base_spreads,
            volatility_coeffs,
            skew_coeffs,
        ),
        StrategySweepConfig::InventoryRisk {
            base_spreads,
            volatility_coeffs,
            risk_aversions,
        } => run_inventory_risk_sweep(
            &config.simulation,
            &seeds,
            base_spreads,
            volatility_coeffs,
            risk_aversions,
        ),
        StrategySweepConfig::RegimeAdaptiveVolatilityAware {
            low_vol,
            normal_vol,
            high_vol,
        } => run_regime_adaptive_volatility_aware_sweep(
            &config.simulation,
            &seeds,
            low_vol,
            normal_vol,
            high_vol,
        ),
    };

    for result in &mut results {
        result.inactivity_penalty = inactivity_penalty(result, scoring);
        result.score = score_result(result, scoring);
        result.stability = stability_for_metrics(&result.seed_metrics, scoring);
        result.stable_score = stable_score_result(result, scoring);
    }

    results.sort_by(|a, b| b.stable_score.total_cmp(&a.stable_score));
    results
}

pub fn sweep_results_to_csv(results: &[SweepResult]) -> String {
    let mut csv = String::from(
        "rank,experiment,strategy_type,spread,volatility_coeff,risk_aversion,skew,runs,score,score_std,stable_score,inactivity_penalty,avg_final_pnl,final_pnl_std,avg_min_pnl,\
         avg_max_pnl,avg_max_drawdown,avg_final_inventory,avg_max_abs_inventory,\
         max_drawdown_std,avg_abs_inventory,avg_total_fills,avg_buy_fills,avg_sell_fills,avg_traded_quantity,\
         avg_traded_notional,avg_total_fees,avg_total_adverse_selection,avg_low_vol_steps,avg_normal_vol_steps,avg_high_vol_steps,\
         avg_low_vol_fills,avg_normal_vol_fills,avg_high_vol_fills,avg_low_vol_fees,avg_normal_vol_fees,avg_high_vol_fees,\
         avg_low_vol_adverse_selection,avg_normal_vol_adverse_selection,avg_high_vol_adverse_selection,\
         avg_low_vol_abs_inventory,avg_normal_vol_abs_inventory,avg_high_vol_abs_inventory\n",
    );

    for (index, result) in results.iter().enumerate() {
        let metrics = result.metrics;
        let row = vec![
            (index + 1).to_string(),
            result.name.clone(),
            result.strategy.strategy_type().to_string(),
            format_f64(result.representative_spread()),
            optional_f64(result.representative_volatility_coeff()),
            optional_f64(result.strategy.risk_aversion()),
            optional_f64(result.representative_skew_coeff()),
            result.runs.to_string(),
            format_f64(result.score),
            format_f64(result.stability.score_std),
            format_f64(result.stable_score),
            format_f64(result.inactivity_penalty),
            format_f64(metrics.final_pnl),
            format_f64(result.stability.final_pnl_std),
            format_f64(metrics.min_pnl),
            format_f64(metrics.max_pnl),
            format_f64(metrics.max_drawdown),
            format_f64(metrics.final_inventory),
            format_f64(metrics.max_abs_inventory),
            format_f64(result.stability.max_drawdown_std),
            format_f64(metrics.avg_abs_inventory),
            format_f64(metrics.total_fills),
            format_f64(metrics.buy_fills),
            format_f64(metrics.sell_fills),
            format_f64(metrics.traded_quantity),
            format_f64(metrics.traded_notional),
            format_f64(metrics.total_fees),
            format_f64(metrics.total_adverse_selection),
            format_f64(metrics.low_vol_steps),
            format_f64(metrics.normal_vol_steps),
            format_f64(metrics.high_vol_steps),
            format_f64(metrics.low_vol_fills),
            format_f64(metrics.normal_vol_fills),
            format_f64(metrics.high_vol_fills),
            format_f64(metrics.low_vol_fees),
            format_f64(metrics.normal_vol_fees),
            format_f64(metrics.high_vol_fees),
            format_f64(metrics.low_vol_adverse_selection),
            format_f64(metrics.normal_vol_adverse_selection),
            format_f64(metrics.high_vol_adverse_selection),
            format_f64(metrics.low_vol_avg_abs_inventory),
            format_f64(metrics.normal_vol_avg_abs_inventory),
            format_f64(metrics.high_vol_avg_abs_inventory),
        ];

        writeln!(csv, "{}", row.join(",")).expect("writing to a String should not fail");
    }

    csv
}

fn format_f64(value: f64) -> String {
    format!("{value:.6}")
}

fn optional_f64(value: Option<f64>) -> String {
    value.map(format_f64).unwrap_or_default()
}

fn sweep_seeds(config: &SweepConfig) -> Vec<u64> {
    if config.seeds.is_empty() {
        vec![config.simulation.seed]
    } else {
        config.seeds.clone()
    }
}

fn run_seeded_reports(
    simulation: &SimulationConfig,
    seeds: &[u64],
    name: &str,
    strategy: &(impl Clone + QuoteStrategy),
) -> Vec<ExperimentReport> {
    seeds
        .iter()
        .filter_map(|seed| {
            let mut seeded_simulation = simulation.clone();
            seeded_simulation.seed = *seed;

            Experiment::new(name, seeded_simulation, strategy.clone()).run()
        })
        .collect()
}

fn aggregate_reports(
    name: String,
    strategy: SweepStrategyParams,
    reports: &[ExperimentReport],
) -> Option<SweepResult> {
    let runs = reports.len();

    if runs == 0 {
        return None;
    }

    let mut metrics = SweepMetrics {
        final_pnl: 0.0,
        min_pnl: 0.0,
        max_pnl: 0.0,
        max_drawdown: 0.0,
        final_inventory: 0.0,
        max_abs_inventory: 0.0,
        avg_abs_inventory: 0.0,
        total_fills: 0.0,
        buy_fills: 0.0,
        sell_fills: 0.0,
        traded_quantity: 0.0,
        traded_notional: 0.0,
        total_fees: 0.0,
        total_adverse_selection: 0.0,
        low_vol_steps: 0.0,
        normal_vol_steps: 0.0,
        high_vol_steps: 0.0,
        low_vol_fills: 0.0,
        normal_vol_fills: 0.0,
        high_vol_fills: 0.0,
        low_vol_fees: 0.0,
        normal_vol_fees: 0.0,
        high_vol_fees: 0.0,
        low_vol_adverse_selection: 0.0,
        normal_vol_adverse_selection: 0.0,
        high_vol_adverse_selection: 0.0,
        low_vol_avg_abs_inventory: 0.0,
        normal_vol_avg_abs_inventory: 0.0,
        high_vol_avg_abs_inventory: 0.0,
    };

    for report in reports {
        metrics.final_pnl += report.metrics.final_pnl;
        metrics.min_pnl += report.metrics.min_pnl;
        metrics.max_pnl += report.metrics.max_pnl;
        metrics.max_drawdown += report.metrics.max_drawdown;
        metrics.final_inventory += report.metrics.final_inventory;
        metrics.max_abs_inventory += report.metrics.max_abs_inventory;
        metrics.avg_abs_inventory += report.metrics.avg_abs_inventory;
        metrics.total_fills += report.metrics.total_fills as f64;
        metrics.buy_fills += report.metrics.buy_fills as f64;
        metrics.sell_fills += report.metrics.sell_fills as f64;
        metrics.traded_quantity += report.metrics.traded_quantity;
        metrics.traded_notional += report.metrics.traded_notional;
        metrics.total_fees += report.metrics.total_fees;
        metrics.total_adverse_selection += report.metrics.total_adverse_selection;
        metrics.low_vol_steps += report.metrics.low_vol_steps as f64;
        metrics.normal_vol_steps += report.metrics.normal_vol_steps as f64;
        metrics.high_vol_steps += report.metrics.high_vol_steps as f64;
        metrics.low_vol_fills += report.metrics.low_vol_fills as f64;
        metrics.normal_vol_fills += report.metrics.normal_vol_fills as f64;
        metrics.high_vol_fills += report.metrics.high_vol_fills as f64;
        metrics.low_vol_fees += report.metrics.low_vol_fees;
        metrics.normal_vol_fees += report.metrics.normal_vol_fees;
        metrics.high_vol_fees += report.metrics.high_vol_fees;
        metrics.low_vol_adverse_selection += report.metrics.low_vol_adverse_selection;
        metrics.normal_vol_adverse_selection += report.metrics.normal_vol_adverse_selection;
        metrics.high_vol_adverse_selection += report.metrics.high_vol_adverse_selection;
        metrics.low_vol_avg_abs_inventory += report.metrics.low_vol_avg_abs_inventory;
        metrics.normal_vol_avg_abs_inventory += report.metrics.normal_vol_avg_abs_inventory;
        metrics.high_vol_avg_abs_inventory += report.metrics.high_vol_avg_abs_inventory;
    }

    let runs_f64 = runs as f64;
    metrics.final_pnl /= runs_f64;
    metrics.min_pnl /= runs_f64;
    metrics.max_pnl /= runs_f64;
    metrics.max_drawdown /= runs_f64;
    metrics.final_inventory /= runs_f64;
    metrics.max_abs_inventory /= runs_f64;
    metrics.avg_abs_inventory /= runs_f64;
    metrics.total_fills /= runs_f64;
    metrics.buy_fills /= runs_f64;
    metrics.sell_fills /= runs_f64;
    metrics.traded_quantity /= runs_f64;
    metrics.traded_notional /= runs_f64;
    metrics.total_fees /= runs_f64;
    metrics.total_adverse_selection /= runs_f64;
    metrics.low_vol_steps /= runs_f64;
    metrics.normal_vol_steps /= runs_f64;
    metrics.high_vol_steps /= runs_f64;
    metrics.low_vol_fills /= runs_f64;
    metrics.normal_vol_fills /= runs_f64;
    metrics.high_vol_fills /= runs_f64;
    metrics.low_vol_fees /= runs_f64;
    metrics.normal_vol_fees /= runs_f64;
    metrics.high_vol_fees /= runs_f64;
    metrics.low_vol_adverse_selection /= runs_f64;
    metrics.normal_vol_adverse_selection /= runs_f64;
    metrics.high_vol_adverse_selection /= runs_f64;
    metrics.low_vol_avg_abs_inventory /= runs_f64;
    metrics.normal_vol_avg_abs_inventory /= runs_f64;
    metrics.high_vol_avg_abs_inventory /= runs_f64;

    Some(SweepResult {
        name,
        strategy,
        runs,
        metrics,
        stability: SweepStability::default(),
        inactivity_penalty: 0.0,
        score: 0.0,
        stable_score: 0.0,
        seed_metrics: reports.iter().map(|report| report.metrics).collect(),
    })
}

fn run_fixed_spread_sweep(
    simulation: &SimulationConfig,
    seeds: &[u64],
    spreads: &[f64],
    skew_coeffs: &[f64],
) -> Vec<SweepResult> {
    let mut results = Vec::with_capacity(spreads.len() * skew_coeffs.len());

    for spread in spreads {
        for skew_coeff in skew_coeffs {
            let strategy = StrategyParams {
                spread: *spread,
                skew_coeff: *skew_coeff,
            };
            let name = format!("spread_{spread:.2}_skew_{skew_coeff:.2}");
            let reports = run_seeded_reports(simulation, seeds, &name, &strategy);
            let sweep_strategy = SweepStrategyParams::FixedSpread {
                spread: *spread,
                skew_coeff: *skew_coeff,
            };

            if let Some(result) = aggregate_reports(name, sweep_strategy, &reports) {
                results.push(result);
            }
        }
    }

    results
}

fn run_volatility_aware_sweep(
    simulation: &SimulationConfig,
    seeds: &[u64],
    base_spreads: &[f64],
    volatility_coeffs: &[f64],
    skew_coeffs: &[f64],
) -> Vec<SweepResult> {
    let mut results =
        Vec::with_capacity(base_spreads.len() * volatility_coeffs.len() * skew_coeffs.len());

    for base_spread in base_spreads {
        for volatility_coeff in volatility_coeffs {
            for skew_coeff in skew_coeffs {
                let strategy = VolatilityAwareParams {
                    base_spread: *base_spread,
                    volatility_coeff: *volatility_coeff,
                    skew_coeff: *skew_coeff,
                };
                let name =
                    format!("base_{base_spread:.2}_vol_{volatility_coeff:.2}_skew_{skew_coeff:.2}");
                let reports = run_seeded_reports(simulation, seeds, &name, &strategy);
                let sweep_strategy = SweepStrategyParams::VolatilityAware {
                    base_spread: *base_spread,
                    volatility_coeff: *volatility_coeff,
                    skew_coeff: *skew_coeff,
                };

                if let Some(result) = aggregate_reports(name, sweep_strategy, &reports) {
                    results.push(result);
                }
            }
        }
    }

    results
}

fn run_inventory_risk_sweep(
    simulation: &SimulationConfig,
    seeds: &[u64],
    base_spreads: &[f64],
    volatility_coeffs: &[f64],
    risk_aversions: &[f64],
) -> Vec<SweepResult> {
    let mut results =
        Vec::with_capacity(base_spreads.len() * volatility_coeffs.len() * risk_aversions.len());

    for base_spread in base_spreads {
        for volatility_coeff in volatility_coeffs {
            for risk_aversion in risk_aversions {
                let strategy = InventoryRiskParams {
                    base_spread: *base_spread,
                    volatility_coeff: *volatility_coeff,
                    risk_aversion: *risk_aversion,
                };
                let name = format!(
                    "base_{base_spread:.2}_vol_{volatility_coeff:.2}_risk_{risk_aversion:.2}"
                );
                let reports = run_seeded_reports(simulation, seeds, &name, &strategy);
                let sweep_strategy = SweepStrategyParams::InventoryRisk {
                    base_spread: *base_spread,
                    volatility_coeff: *volatility_coeff,
                    risk_aversion: *risk_aversion,
                };

                if let Some(result) = aggregate_reports(name, sweep_strategy, &reports) {
                    results.push(result);
                }
            }
        }
    }

    results
}

fn run_regime_adaptive_volatility_aware_sweep(
    simulation: &SimulationConfig,
    seeds: &[u64],
    low_vol_params: &[VolatilityAwareParams],
    normal_vol_params: &[VolatilityAwareParams],
    high_vol_params: &[VolatilityAwareParams],
) -> Vec<SweepResult> {
    let mut results =
        Vec::with_capacity(low_vol_params.len() * normal_vol_params.len() * high_vol_params.len());

    for low_vol in low_vol_params {
        for normal_vol in normal_vol_params {
            for high_vol in high_vol_params {
                let strategy = RegimeAdaptiveVolatilityAwareParams {
                    low_vol: low_vol.clone(),
                    normal_vol: normal_vol.clone(),
                    high_vol: high_vol.clone(),
                };
                let name = format!(
                    "low_{:.2}_{:.2}_{:.2}_normal_{:.2}_{:.2}_{:.2}_high_{:.2}_{:.2}_{:.2}",
                    low_vol.base_spread,
                    low_vol.volatility_coeff,
                    low_vol.skew_coeff,
                    normal_vol.base_spread,
                    normal_vol.volatility_coeff,
                    normal_vol.skew_coeff,
                    high_vol.base_spread,
                    high_vol.volatility_coeff,
                    high_vol.skew_coeff,
                );
                let reports = run_seeded_reports(simulation, seeds, &name, &strategy);
                let sweep_strategy = SweepStrategyParams::RegimeAdaptiveVolatilityAware {
                    low_vol: low_vol.clone(),
                    normal_vol: normal_vol.clone(),
                    high_vol: high_vol.clone(),
                };

                if let Some(result) = aggregate_reports(name, sweep_strategy, &reports) {
                    results.push(result);
                }
            }
        }
    }

    results
}

fn score_result(result: &SweepResult, scoring: ScoringConfig) -> f64 {
    result.metrics.final_pnl
        - scoring.drawdown_weight * result.metrics.max_drawdown
        - scoring.inventory_weight * result.metrics.max_abs_inventory
        - result.inactivity_penalty
}

fn stable_score_result(result: &SweepResult, scoring: ScoringConfig) -> f64 {
    result.score - scoring.stability_weight * result.stability.score_std
}

fn inactivity_penalty(result: &SweepResult, scoring: ScoringConfig) -> f64 {
    let missing_fills = (scoring.min_fills as f64 - result.metrics.total_fills).max(0.0);

    missing_fills * scoring.missing_fill_penalty
}

fn stability_for_metrics(metrics: &[SimulationMetrics], scoring: ScoringConfig) -> SweepStability {
    let scores: Vec<f64> = metrics
        .iter()
        .map(|metrics| score_metrics(*metrics, scoring))
        .collect();
    let final_pnls: Vec<f64> = metrics.iter().map(|metrics| metrics.final_pnl).collect();
    let max_drawdowns: Vec<f64> = metrics.iter().map(|metrics| metrics.max_drawdown).collect();

    SweepStability {
        score_std: std_dev(&scores),
        final_pnl_std: std_dev(&final_pnls),
        max_drawdown_std: std_dev(&max_drawdowns),
    }
}

fn score_metrics(metrics: SimulationMetrics, scoring: ScoringConfig) -> f64 {
    let missing_fills = (scoring.min_fills as f64 - metrics.total_fills as f64).max(0.0);
    let inactivity_penalty = missing_fills * scoring.missing_fill_penalty;

    metrics.final_pnl
        - scoring.drawdown_weight * metrics.max_drawdown
        - scoring.inventory_weight * metrics.max_abs_inventory
        - inactivity_penalty
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_runs_each_parameter_combination() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![1, 2],
            strategy: StrategySweepConfig::FixedSpread {
                spreads: vec![0.3, 0.5],
                skew_coeffs: vec![0.05, 0.1, 0.2],
            },
            scoring: ScoringConfig::default(),
        });

        assert_eq!(results.len(), 6);
    }

    #[test]
    fn sweep_results_are_sorted_by_stable_score_descending() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![1, 2],
            strategy: StrategySweepConfig::FixedSpread {
                spreads: vec![0.3, 0.5],
                skew_coeffs: vec![0.05, 0.1],
            },
            scoring: ScoringConfig::default(),
        });

        assert!(
            results
                .windows(2)
                .all(|window| window[0].stable_score >= window[1].stable_score)
        );
    }

    #[test]
    fn sweep_results_export_to_csv() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![42],
            strategy: StrategySweepConfig::FixedSpread {
                spreads: vec![0.3],
                skew_coeffs: vec![0.05],
            },
            scoring: ScoringConfig::default(),
        });

        let csv = sweep_results_to_csv(&results);

        assert!(csv.starts_with("rank,experiment,strategy_type,spread"));
        assert!(csv.contains("risk_aversion"));
        assert!(csv.contains("stable_score"));
        assert!(csv.contains("avg_low_vol_steps"));
        assert!(csv.contains("avg_normal_vol_steps"));
        assert!(csv.contains("avg_high_vol_steps"));
        assert!(csv.contains("avg_low_vol_fills"));
        assert!(csv.contains("avg_normal_vol_fills"));
        assert!(csv.contains("avg_high_vol_fills"));
        assert!(csv.contains("avg_low_vol_adverse_selection"));
        assert!(csv.contains("avg_normal_vol_abs_inventory"));
        assert!(csv.contains("spread_0.30_skew_0.05"));
        assert_eq!(csv.lines().count(), 2);
    }

    #[test]
    fn inactivity_penalty_applies_to_low_fill_reports() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![42],
            strategy: StrategySweepConfig::FixedSpread {
                spreads: vec![1.0],
                skew_coeffs: vec![0.05],
            },
            scoring: ScoringConfig {
                min_fills: 50,
                missing_fill_penalty: 0.25,
                ..ScoringConfig::default()
            },
        });

        assert_eq!(results.len(), 1);
        assert!(results[0].metrics.total_fills < 50.0);
        assert!(results[0].inactivity_penalty > 0.0);
        assert!(results[0].score < results[0].metrics.final_pnl);
    }

    #[test]
    fn sweep_aggregates_across_seeds() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![1, 2, 3],
            strategy: StrategySweepConfig::FixedSpread {
                spreads: vec![0.3],
                skew_coeffs: vec![0.05],
            },
            scoring: ScoringConfig::default(),
        });

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].runs, 3);
        assert!(results[0].metrics.total_fills >= 0.0);
    }

    #[test]
    fn sweep_reports_stability_across_seeds() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![1, 2, 3],
            strategy: StrategySweepConfig::FixedSpread {
                spreads: vec![0.3],
                skew_coeffs: vec![0.05],
            },
            scoring: ScoringConfig::default(),
        });

        assert_eq!(results.len(), 1);
        assert!(results[0].stability.final_pnl_std >= 0.0);
        assert!(results[0].stability.max_drawdown_std >= 0.0);
        assert!(results[0].stability.score_std >= 0.0);
        assert!(results[0].stable_score <= results[0].score);
    }

    #[test]
    fn volatility_aware_sweep_runs_each_parameter_combination() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![1],
            strategy: StrategySweepConfig::VolatilityAware {
                base_spreads: vec![0.3, 0.5],
                volatility_coeffs: vec![0.0, 2.0],
                skew_coeffs: vec![0.05],
            },
            scoring: ScoringConfig::default(),
        });

        assert_eq!(results.len(), 4);
        assert!(
            results
                .iter()
                .all(|result| result.strategy.strategy_type() == "volatility_aware")
        );
    }

    #[test]
    fn inventory_risk_sweep_runs_each_parameter_combination() {
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![1],
            strategy: StrategySweepConfig::InventoryRisk {
                base_spreads: vec![0.2, 0.3],
                volatility_coeffs: vec![1.0, 2.0],
                risk_aversions: vec![0.0, 2.0],
            },
            scoring: ScoringConfig::default(),
        });

        assert_eq!(results.len(), 8);
        assert!(
            results
                .iter()
                .all(|result| result.strategy.strategy_type() == "inventory_risk")
        );
        assert!(
            results
                .iter()
                .all(|result| result.strategy.risk_aversion().is_some())
        );
    }

    #[test]
    fn regime_adaptive_sweep_runs_each_parameter_combination() {
        let params = VolatilityAwareParams {
            base_spread: 0.5,
            volatility_coeff: 1.0,
            skew_coeff: 0.05,
        };
        let wider_params = VolatilityAwareParams {
            base_spread: 0.8,
            volatility_coeff: 2.0,
            skew_coeff: 0.1,
        };
        let results = run_parameter_sweep(SweepConfig {
            name: None,
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            seeds: vec![1],
            strategy: StrategySweepConfig::RegimeAdaptiveVolatilityAware {
                low_vol: vec![params.clone(), wider_params.clone()],
                normal_vol: vec![params.clone()],
                high_vol: vec![wider_params],
            },
            scoring: ScoringConfig::default(),
        });

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| {
            result.strategy.strategy_type() == "regime_adaptive_volatility_aware"
        }));
    }
}

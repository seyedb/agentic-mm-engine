use serde::{Deserialize, Serialize};

use std::fmt::Write;

use crate::engine::simulation::SimulationConfig;
use crate::experiment::{Experiment, ExperimentReport};
use crate::strategy::QuoteStrategy;
use crate::strategy::market_maker::StrategyParams;
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub name: String,
    pub strategy: SweepStrategyParams,
    pub runs: usize,
    pub metrics: SweepMetrics,
    pub inactivity_penalty: f64,
    pub score: f64,
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
}

impl SweepStrategyParams {
    pub fn primary_spread(&self) -> f64 {
        match self {
            Self::FixedSpread { spread, .. } => *spread,
            Self::VolatilityAware { base_spread, .. } => *base_spread,
        }
    }

    pub fn skew_coeff(&self) -> f64 {
        match self {
            Self::FixedSpread { skew_coeff, .. } | Self::VolatilityAware { skew_coeff, .. } => {
                *skew_coeff
            }
        }
    }

    pub fn volatility_coeff(&self) -> Option<f64> {
        match self {
            Self::FixedSpread { .. } => None,
            Self::VolatilityAware {
                volatility_coeff, ..
            } => Some(*volatility_coeff),
        }
    }

    pub fn strategy_type(&self) -> &'static str {
        match self {
            Self::FixedSpread { .. } => "fixed_spread",
            Self::VolatilityAware { .. } => "volatility_aware",
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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub drawdown_weight: f64,
    pub inventory_weight: f64,
    pub min_fills: usize,
    pub missing_fill_penalty: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            drawdown_weight: 2.0,
            inventory_weight: 1.0,
            min_fills: 50,
            missing_fill_penalty: 0.25,
        }
    }
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
    };

    for result in &mut results {
        result.inactivity_penalty = inactivity_penalty(result, scoring);
        result.score = score_result(result, scoring);
    }

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results
}

pub fn sweep_results_to_csv(results: &[SweepResult]) -> String {
    let mut csv = String::from(
        "rank,experiment,strategy_type,spread,volatility_coeff,skew,runs,score,inactivity_penalty,avg_final_pnl,avg_min_pnl,\
         avg_max_pnl,avg_max_drawdown,avg_final_inventory,avg_max_abs_inventory,\
         avg_abs_inventory,avg_total_fills,avg_buy_fills,avg_sell_fills,avg_traded_quantity,\
         avg_traded_notional,avg_total_fees,avg_total_adverse_selection\n",
    );

    for (index, result) in results.iter().enumerate() {
        let metrics = result.metrics;

        writeln!(
            csv,
            "{},{},{},{:.6},{},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            index + 1,
            result.name,
            result.strategy.strategy_type(),
            result.strategy.primary_spread(),
            optional_f64(result.strategy.volatility_coeff()),
            result.strategy.skew_coeff(),
            result.runs,
            result.score,
            result.inactivity_penalty,
            metrics.final_pnl,
            metrics.min_pnl,
            metrics.max_pnl,
            metrics.max_drawdown,
            metrics.final_inventory,
            metrics.max_abs_inventory,
            metrics.avg_abs_inventory,
            metrics.total_fills,
            metrics.buy_fills,
            metrics.sell_fills,
            metrics.traded_quantity,
            metrics.traded_notional,
            metrics.total_fees,
            metrics.total_adverse_selection,
        )
        .expect("writing to a String should not fail");
    }

    csv
}

fn optional_f64(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
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
            let mut seeded_simulation = *simulation;
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

    Some(SweepResult {
        name,
        strategy,
        runs,
        metrics,
        inactivity_penalty: 0.0,
        score: 0.0,
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

fn score_result(result: &SweepResult, scoring: ScoringConfig) -> f64 {
    result.metrics.final_pnl
        - scoring.drawdown_weight * result.metrics.max_drawdown
        - scoring.inventory_weight * result.metrics.max_abs_inventory
        - result.inactivity_penalty
}

fn inactivity_penalty(result: &SweepResult, scoring: ScoringConfig) -> f64 {
    let missing_fills = (scoring.min_fills as f64 - result.metrics.total_fills).max(0.0);

    missing_fills * scoring.missing_fill_penalty
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
    fn sweep_results_are_sorted_by_score_descending() {
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
                .all(|window| window[0].score >= window[1].score)
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
}

use serde::{Deserialize, Serialize};

use std::fmt::Write;

use crate::engine::simulation::SimulationConfig;
use crate::experiment::{Experiment, ExperimentReport, run_experiments};
use crate::strategy::market_maker::StrategyParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    pub simulation: SimulationConfig,
    pub spreads: Vec<f64>,
    pub skew_coeffs: Vec<f64>,
    pub scoring: ScoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub report: ExperimentReport,
    pub inactivity_penalty: f64,
    pub score: f64,
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
    let experiments = build_sweep_experiments(config);
    let reports = run_experiments(&experiments);
    let mut results: Vec<SweepResult> = reports
        .into_iter()
        .map(|report| {
            let inactivity_penalty = inactivity_penalty(&report, scoring);
            let score = score_report(&report, scoring, inactivity_penalty);
            SweepResult {
                report,
                inactivity_penalty,
                score,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results
}

pub fn sweep_results_to_csv(results: &[SweepResult]) -> String {
    let mut csv = String::from(
        "rank,experiment,spread,skew,score,inactivity_penalty,final_pnl,min_pnl,max_pnl,max_drawdown,\
         final_inventory,max_abs_inventory,avg_abs_inventory,total_fills,buy_fills,\
         sell_fills,traded_quantity,traded_notional,total_fees,total_adverse_selection\n",
    );

    for (index, result) in results.iter().enumerate() {
        let report = &result.report;
        let metrics = report.metrics;

        writeln!(
            csv,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            index + 1,
            report.name,
            report.strategy.spread,
            report.strategy.skew_coeff,
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

fn build_sweep_experiments(config: SweepConfig) -> Vec<Experiment> {
    let mut experiments = Vec::with_capacity(config.spreads.len() * config.skew_coeffs.len());

    for spread in &config.spreads {
        for skew_coeff in &config.skew_coeffs {
            experiments.push(Experiment::new(
                format!("spread_{spread:.2}_skew_{skew_coeff:.2}"),
                config.simulation,
                StrategyParams {
                    spread: *spread,
                    skew_coeff: *skew_coeff,
                },
            ));
        }
    }

    experiments
}

fn score_report(report: &ExperimentReport, scoring: ScoringConfig, inactivity_penalty: f64) -> f64 {
    report.metrics.final_pnl
        - scoring.drawdown_weight * report.metrics.max_drawdown
        - scoring.inventory_weight * report.metrics.max_abs_inventory
        - inactivity_penalty
}

fn inactivity_penalty(report: &ExperimentReport, scoring: ScoringConfig) -> f64 {
    let missing_fills = scoring.min_fills.saturating_sub(report.metrics.total_fills);

    missing_fills as f64 * scoring.missing_fill_penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_runs_each_parameter_combination() {
        let results = run_parameter_sweep(SweepConfig {
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            spreads: vec![0.3, 0.5],
            skew_coeffs: vec![0.05, 0.1, 0.2],
            scoring: ScoringConfig::default(),
        });

        assert_eq!(results.len(), 6);
    }

    #[test]
    fn sweep_results_are_sorted_by_score_descending() {
        let results = run_parameter_sweep(SweepConfig {
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            spreads: vec![0.3, 0.5],
            skew_coeffs: vec![0.05, 0.1],
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
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            spreads: vec![0.3],
            skew_coeffs: vec![0.05],
            scoring: ScoringConfig::default(),
        });

        let csv = sweep_results_to_csv(&results);

        assert!(csv.starts_with("rank,experiment,spread,skew,score"));
        assert!(csv.contains("spread_0.30_skew_0.05"));
        assert_eq!(csv.lines().count(), 2);
    }

    #[test]
    fn inactivity_penalty_applies_to_low_fill_reports() {
        let results = run_parameter_sweep(SweepConfig {
            simulation: SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            spreads: vec![1.0],
            skew_coeffs: vec![0.05],
            scoring: ScoringConfig {
                min_fills: 50,
                missing_fill_penalty: 0.25,
                ..ScoringConfig::default()
            },
        });

        assert_eq!(results.len(), 1);
        assert!(results[0].report.metrics.total_fills < 50);
        assert!(results[0].inactivity_penalty > 0.0);
        assert!(results[0].score < results[0].report.metrics.final_pnl);
    }
}

use serde::{Deserialize, Serialize};

use crate::engine::simulation::SimulationConfig;
use crate::experiment::{Experiment, ExperimentReport, run_experiments};
use crate::strategy::market_maker::StrategyParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    pub simulation: SimulationConfig,
    pub spreads: Vec<f64>,
    pub skew_coeffs: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub report: ExperimentReport,
    pub score: f64,
}

pub fn run_parameter_sweep(config: SweepConfig) -> Vec<SweepResult> {
    let experiments = build_sweep_experiments(config);
    let reports = run_experiments(&experiments);
    let mut results: Vec<SweepResult> = reports
        .into_iter()
        .map(|report| {
            let score = score_report(&report);
            SweepResult { report, score }
        })
        .collect();

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results
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

fn score_report(report: &ExperimentReport) -> f64 {
    report.metrics.final_pnl - 2.0 * report.metrics.max_drawdown - report.metrics.max_abs_inventory
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
        });

        assert!(
            results
                .windows(2)
                .all(|window| window[0].score >= window[1].score)
        );
    }
}

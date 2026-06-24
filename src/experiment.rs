use serde::{Deserialize, Serialize};

use crate::engine::metrics::SimulationMetrics;
use crate::engine::simulation::{SimulationConfig, run_simulation};
use crate::strategy::market_maker::StrategyParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub name: String,
    pub simulation: SimulationConfig,
    pub strategy: StrategyParams,
}

impl Experiment {
    pub fn new(
        name: impl Into<String>,
        simulation: SimulationConfig,
        strategy: StrategyParams,
    ) -> Self {
        Self {
            name: name.into(),
            simulation,
            strategy,
        }
    }

    pub fn run(&self) -> Option<ExperimentReport> {
        let result = run_simulation(self.simulation, &self.strategy);
        let metrics = SimulationMetrics::from_result(&result)?;

        Some(ExperimentReport {
            name: self.name.clone(),
            simulation: self.simulation,
            strategy: self.strategy.clone(),
            metrics,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub name: String,
    pub simulation: SimulationConfig,
    pub strategy: StrategyParams,
    pub metrics: SimulationMetrics,
}

pub fn run_experiments(experiments: &[Experiment]) -> Vec<ExperimentReport> {
    experiments.iter().filter_map(Experiment::run).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_runs_and_returns_report() {
        let experiment = Experiment::new(
            "baseline",
            SimulationConfig {
                steps: 100,
                ..SimulationConfig::default()
            },
            StrategyParams {
                spread: 0.5,
                skew_coeff: 0.05,
            },
        );

        let report = experiment.run().unwrap();

        assert_eq!(report.name, "baseline");
        assert_eq!(report.metrics.steps, 100);
        assert_eq!(report.strategy.spread, 0.5);
    }

    #[test]
    fn experiment_without_steps_returns_none() {
        let experiment = Experiment::new(
            "empty",
            SimulationConfig {
                steps: 0,
                ..SimulationConfig::default()
            },
            StrategyParams {
                spread: 0.5,
                skew_coeff: 0.05,
            },
        );

        assert!(experiment.run().is_none());
    }
}

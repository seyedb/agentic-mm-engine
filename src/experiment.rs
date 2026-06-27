use serde::{Deserialize, Serialize};

use crate::engine::metrics::SimulationMetrics;
use crate::engine::simulation::{SimulationConfig, run_simulation};
use crate::strategy::QuoteStrategy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment<S> {
    pub name: String,
    pub simulation: SimulationConfig,
    pub strategy: S,
}

impl<S> Experiment<S> {
    pub fn new(name: impl Into<String>, simulation: SimulationConfig, strategy: S) -> Self {
        Self {
            name: name.into(),
            simulation,
            strategy,
        }
    }
}

impl<S> Experiment<S>
where
    S: Clone + QuoteStrategy,
{
    pub fn run(&self) -> Option<ExperimentReport> {
        let result = run_simulation(self.simulation, &self.strategy);
        let metrics = SimulationMetrics::from_result(&result)?;

        Some(ExperimentReport {
            name: self.name.clone(),
            simulation: self.simulation,
            metrics,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub name: String,
    pub simulation: SimulationConfig,
    pub metrics: SimulationMetrics,
}

pub fn run_experiments<S>(experiments: &[Experiment<S>]) -> Vec<ExperimentReport>
where
    S: Clone + QuoteStrategy,
{
    experiments.iter().filter_map(Experiment::run).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::market_maker::StrategyParams;

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

use mm_engine::engine::simulation::SimulationConfig;
use mm_engine::experiment::{Experiment, run_experiments};
use mm_engine::strategy::market_maker::StrategyParams;

fn main() {
    let config = SimulationConfig::default();
    let experiments = vec![
        Experiment::new(
            "baseline",
            config,
            StrategyParams {
                spread: 0.5,
                skew_coeff: 0.05,
            },
        ),
        Experiment::new(
            "wider_spread",
            config,
            StrategyParams {
                spread: 0.8,
                skew_coeff: 0.05,
            },
        ),
        Experiment::new(
            "higher_skew",
            config,
            StrategyParams {
                spread: 0.5,
                skew_coeff: 0.1,
            },
        ),
    ];

    let reports = run_experiments(&experiments);

    println!(
        "{:<16} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "experiment", "pnl", "fills", "max_inv", "avg_inv", "drawdown"
    );

    for report in reports {
        println!(
            "{:<16} {:>8.2} {:>8} {:>8.2} {:>8.2} {:>8.2}",
            report.name,
            report.metrics.final_pnl,
            report.metrics.total_fills,
            report.metrics.max_abs_inventory,
            report.metrics.avg_abs_inventory,
            report.metrics.max_drawdown
        );
    }
}

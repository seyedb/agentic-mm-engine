use mm_engine::engine::simulation::SimulationConfig;
use mm_engine::sweep::{SweepConfig, run_parameter_sweep};

fn main() {
    let results = run_parameter_sweep(SweepConfig {
        simulation: SimulationConfig::default(),
        spreads: vec![0.3, 0.5, 0.8, 1.0],
        skew_coeffs: vec![0.02, 0.05, 0.1, 0.2],
    });

    println!("Top parameter sweep results");
    println!(
        "{:<4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "rank", "spread", "skew", "pnl", "fills", "max_inv", "drawdown", "score"
    );

    for (index, result) in results.iter().take(10).enumerate() {
        println!(
            "{:<4} {:>8.2} {:>8.2} {:>8.2} {:>8} {:>8.2} {:>8.2} {:>8.2}",
            index + 1,
            result.report.strategy.spread,
            result.report.strategy.skew_coeff,
            result.report.metrics.final_pnl,
            result.report.metrics.total_fills,
            result.report.metrics.max_abs_inventory,
            result.report.metrics.max_drawdown,
            result.score,
        );
    }
}

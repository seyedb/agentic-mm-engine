use mm_engine::engine::simulation::SimulationConfig;
use mm_engine::sweep::{ScoringConfig, SweepConfig, run_parameter_sweep, sweep_results_to_csv};
use std::fs;
use std::path::Path;

fn main() {
    let results = run_parameter_sweep(SweepConfig {
        simulation: SimulationConfig::default(),
        spreads: vec![0.3, 0.5, 0.8, 1.0],
        skew_coeffs: vec![0.02, 0.05, 0.1, 0.2],
        scoring: ScoringConfig::default(),
    });

    println!("Top parameter sweep results");
    println!(
        "{:<4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "rank", "spread", "skew", "pnl", "fills", "fees", "adv", "drawdown", "idle", "score"
    );

    for (index, result) in results.iter().take(10).enumerate() {
        println!(
            "{:<4} {:>8.2} {:>8.2} {:>8.2} {:>8} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            index + 1,
            result.report.strategy.spread,
            result.report.strategy.skew_coeff,
            result.report.metrics.final_pnl,
            result.report.metrics.total_fills,
            result.report.metrics.total_fees,
            result.report.metrics.total_adverse_selection,
            result.report.metrics.max_drawdown,
            result.inactivity_penalty,
            result.score,
        );
    }

    let output_dir = Path::new("target").join("reports");
    fs::create_dir_all(&output_dir).expect("failed to create report output directory");

    let csv_path = output_dir.join("sweep_results.csv");
    fs::write(&csv_path, sweep_results_to_csv(&results)).expect("failed to write sweep CSV");

    println!("\nwrote {}", csv_path.display());
}

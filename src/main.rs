use mm_engine::sweep::{SweepConfig, run_parameter_sweep, sweep_results_to_csv};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_PATH: &str = "configs/baseline_sweep.json";

fn main() -> Result<(), Box<dyn Error>> {
    let config_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));

    let config = load_sweep_config(&config_path)?;
    let results = run_parameter_sweep(config);

    println!("Top parameter sweep results");
    println!("config: {}", config_path.display());
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

    Ok(())
}

fn load_sweep_config(path: &Path) -> Result<SweepConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config = serde_json::from_str(&contents)?;

    Ok(config)
}

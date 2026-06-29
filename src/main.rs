use mm_engine::engine::dataset::{append_step_dataset_rows, step_dataset_header};
use mm_engine::engine::metrics::SimulationMetrics;
use mm_engine::engine::simulation::{
    FillModelConfig, SimulationConfig, SimulationResult, run_replay_simulation, run_simulation,
};
use mm_engine::market::{InMemoryMarketData, market_events_from_csv};
use mm_engine::strategy::market_maker::StrategyParams;
use mm_engine::sweep::{SweepConfig, SweepResult, run_parameter_sweep, sweep_results_to_csv};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_PATH: &str = "configs/baseline_sweep.json";
const REGIME_SUMMARY_HEADER: &[&str] = &[
    "regime",
    "strategy_type",
    "best_spread",
    "best_volatility_coeff",
    "best_risk_aversion",
    "best_skew",
    "runs",
    "best_score",
    "best_score_std",
    "best_stable_score",
    "avg_best_pnl",
    "best_pnl_std",
    "avg_fills",
    "avg_max_drawdown",
    "max_drawdown_std",
    "avg_max_abs_inventory",
    "avg_low_vol_steps",
    "avg_normal_vol_steps",
    "avg_high_vol_steps",
    "avg_low_vol_fills",
    "avg_normal_vol_fills",
    "avg_high_vol_fills",
    "avg_low_vol_fees",
    "avg_normal_vol_fees",
    "avg_high_vol_fees",
    "avg_low_vol_adverse_selection",
    "avg_normal_vol_adverse_selection",
    "avg_high_vol_adverse_selection",
    "avg_low_vol_abs_inventory",
    "avg_normal_vol_abs_inventory",
    "avg_high_vol_abs_inventory",
];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "replay") {
        return run_replay_command(&args);
    }

    let output_dir = Path::new("target").join("reports");
    fs::create_dir_all(&output_dir).expect("failed to create report output directory");

    let config_paths = config_paths_from_args(&args);
    let mut summary_rows = Vec::new();

    for config_path in config_paths {
        let config = load_sweep_config(&config_path)?;
        let regime = config_name(&config, &config_path);
        let results = run_parameter_sweep(config.clone());

        print_top_results(&regime, &config_path, &results);

        let csv_path = output_dir.join(format!("{regime}.csv"));
        fs::write(&csv_path, sweep_results_to_csv(&results)).expect("failed to write sweep CSV");

        if let Some(best_result) = results.first() {
            summary_rows.push(regime_summary_row(&regime, best_result));
            let step_dataset_path = output_dir.join(format!("{regime}_best_steps.csv"));
            fs::write(
                &step_dataset_path,
                best_step_dataset_to_csv(&regime, &config, best_result),
            )
            .expect("failed to write best-step dataset CSV");

            println!("wrote {}", step_dataset_path.display());
        }

        println!("\nwrote {}", csv_path.display());
    }

    let summary_path = output_dir.join("regime_summary.csv");
    fs::write(&summary_path, regime_summaries_to_csv(&summary_rows))
        .expect("failed to write regime summary CSV");

    println!("wrote {}", summary_path.display());

    Ok(())
}

fn config_paths_from_args(args: &[String]) -> Vec<PathBuf> {
    if args.is_empty() {
        vec![PathBuf::from(DEFAULT_CONFIG_PATH)]
    } else {
        args.iter().map(PathBuf::from).collect()
    }
}

fn run_replay_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 2 {
        return Err("usage: cargo run -- replay <events.csv>".into());
    }
    let path = &args[1];

    let events = market_events_from_csv(path)?;
    if events.is_empty() {
        return Err("replay CSV contains no events".into());
    }

    let steps = events.len();
    let mut data = InMemoryMarketData::new(events);
    let strategy = StrategyParams {
        spread: 0.5,
        skew_coeff: 0.05,
    };
    let result = run_replay_simulation(replay_config(steps), &mut data, &strategy);

    print_replay_results(path, &result);

    Ok(())
}

fn replay_config(steps: usize) -> SimulationConfig {
    SimulationConfig {
        steps,
        fill_model: FillModelConfig::DistanceIntensity {
            base_intensity: 0.12,
            distance_decay: 4.0,
            volatility_boost: 1.0,
        },
        ..SimulationConfig::default()
    }
}

fn print_replay_results(path: &str, result: &SimulationResult) {
    println!("Replay results");
    println!("data: {path}");

    let Some(metrics) = SimulationMetrics::from_result(result) else {
        println!("steps: 0");
        return;
    };

    println!("steps: {}", metrics.steps);
    println!("fills: {}", metrics.total_fills);
    println!("final_pnl: {:.2}", metrics.final_pnl);
    println!("final_inventory: {:.2}", metrics.final_inventory);
    println!("fees: {:.2}", metrics.total_fees);
    println!("max_drawdown: {:.2}", metrics.max_drawdown);
}

fn load_sweep_config(path: &Path) -> Result<SweepConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config = serde_json::from_str(&contents)?;

    Ok(config)
}

fn print_top_results(regime: &str, config_path: &Path, results: &[SweepResult]) {
    println!("Top parameter sweep results");
    println!("name: {regime}");
    println!("config: {}", config_path.display());
    println!(
        "{:<4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "rank",
        "runs",
        "spread",
        "vol",
        "risk",
        "skew",
        "avg_pnl",
        "avg_fill",
        "avg_fee",
        "avg_adv",
        "avg_dd",
        "pnl_sd",
        "idle",
        "score",
        "score_sd",
        "stable"
    );

    for (index, result) in results.iter().take(10).enumerate() {
        println!(
            "{:<4} {:>5} {:>8.2} {:>8} {:>8} {:>8} {:>8.2} {:>8.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            index + 1,
            result.runs,
            result.representative_spread(),
            optional_f64(result.representative_volatility_coeff()),
            optional_f64(result.strategy.risk_aversion()),
            optional_f64(result.representative_skew_coeff()),
            result.metrics.final_pnl,
            result.metrics.total_fills,
            result.metrics.total_fees,
            result.metrics.total_adverse_selection,
            result.metrics.max_drawdown,
            result.stability.final_pnl_std,
            result.inactivity_penalty,
            result.score,
            result.stability.score_std,
            result.stable_score,
        );
    }
}

fn optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_string())
}

fn best_step_dataset_to_csv(
    regime: &str,
    config: &SweepConfig,
    best_result: &SweepResult,
) -> String {
    let mut csv = String::from(step_dataset_header());

    for seed in sweep_seeds(config) {
        let mut simulation = config.simulation.clone();
        simulation.seed = seed;
        let result = run_simulation(simulation, &best_result.strategy);

        append_step_dataset_rows(
            &mut csv,
            regime,
            best_result.strategy.strategy_type(),
            seed,
            &result,
        );
    }

    csv
}

fn sweep_seeds(config: &SweepConfig) -> Vec<u64> {
    if config.seeds.is_empty() {
        vec![config.simulation.seed]
    } else {
        config.seeds.clone()
    }
}

fn config_name(config: &SweepConfig, config_path: &Path) -> String {
    config
        .name
        .clone()
        .unwrap_or_else(|| file_stem(config_path))
}

fn file_stem(config_path: &Path) -> String {
    config_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("sweep_results")
        .to_string()
}

fn regime_summary_row(regime: &str, result: &SweepResult) -> Vec<String> {
    vec![
        regime.to_string(),
        result.strategy.strategy_type().to_string(),
        format_csv_f64(result.representative_spread()),
        optional_csv_f64(result.representative_volatility_coeff()),
        optional_csv_f64(result.strategy.risk_aversion()),
        optional_csv_f64(result.representative_skew_coeff()),
        result.runs.to_string(),
        format_csv_f64(result.score),
        format_csv_f64(result.stability.score_std),
        format_csv_f64(result.stable_score),
        format_csv_f64(result.metrics.final_pnl),
        format_csv_f64(result.stability.final_pnl_std),
        format_csv_f64(result.metrics.total_fills),
        format_csv_f64(result.metrics.max_drawdown),
        format_csv_f64(result.stability.max_drawdown_std),
        format_csv_f64(result.metrics.max_abs_inventory),
        format_csv_f64(result.metrics.low_vol_steps),
        format_csv_f64(result.metrics.normal_vol_steps),
        format_csv_f64(result.metrics.high_vol_steps),
        format_csv_f64(result.metrics.low_vol_fills),
        format_csv_f64(result.metrics.normal_vol_fills),
        format_csv_f64(result.metrics.high_vol_fills),
        format_csv_f64(result.metrics.low_vol_fees),
        format_csv_f64(result.metrics.normal_vol_fees),
        format_csv_f64(result.metrics.high_vol_fees),
        format_csv_f64(result.metrics.low_vol_adverse_selection),
        format_csv_f64(result.metrics.normal_vol_adverse_selection),
        format_csv_f64(result.metrics.high_vol_adverse_selection),
        format_csv_f64(result.metrics.low_vol_avg_abs_inventory),
        format_csv_f64(result.metrics.normal_vol_avg_abs_inventory),
        format_csv_f64(result.metrics.high_vol_avg_abs_inventory),
    ]
}

fn regime_summaries_to_csv(rows: &[Vec<String>]) -> String {
    let mut csv = REGIME_SUMMARY_HEADER.join(",");
    csv.push('\n');

    for row in rows {
        debug_assert_eq!(REGIME_SUMMARY_HEADER.len(), row.len());
        csv.push_str(&row.join(","));
        csv.push('\n');
    }

    csv
}

fn format_csv_f64(value: f64) -> String {
    format!("{value:.6}")
}

fn optional_csv_f64(value: Option<f64>) -> String {
    value.map(format_csv_f64).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regime_summary_csv_writes_header_and_row() {
        let row = vec!["value".to_string(); REGIME_SUMMARY_HEADER.len()];
        let csv = regime_summaries_to_csv(&[row]);
        let mut lines = csv.lines();

        assert_eq!(
            lines.next().unwrap().split(',').count(),
            REGIME_SUMMARY_HEADER.len()
        );
        assert_eq!(
            lines.next().unwrap().split(',').count(),
            REGIME_SUMMARY_HEADER.len()
        );
        assert!(lines.next().is_none());
    }
}

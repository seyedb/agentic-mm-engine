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
    let options = ReplayOptions::parse(args)?;

    let events = market_events_from_csv(&options.path)?;
    if events.is_empty() {
        return Err("replay CSV contains no events".into());
    }

    let steps = events.len();
    let mut data = InMemoryMarketData::new(events);
    let strategy = StrategyParams {
        spread: options.spread,
        skew_coeff: options.skew,
    };
    let config = replay_config(steps, &options);
    let seed = config.seed;
    let result = run_replay_simulation(config, &mut data, &strategy);
    let output_dir = Path::new("target").join("reports");
    fs::create_dir_all(&output_dir).expect("failed to create report output directory");
    let replay_name = format!("{}_replay", file_stem(Path::new(&options.path)));
    let step_dataset_path = output_dir.join(format!("{replay_name}_steps.csv"));
    fs::write(
        &step_dataset_path,
        replay_step_dataset_to_csv(&replay_name, seed, &result),
    )
    .expect("failed to write replay step dataset CSV");

    print_replay_results(&options, &result);
    println!("wrote {}", step_dataset_path.display());

    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct ReplayOptions {
    path: String,
    spread: f64,
    skew: f64,
    quantity: f64,
    fee_rate: f64,
}

impl ReplayOptions {
    fn parse(args: &[String]) -> Result<Self, String> {
        if args.len() < 2 {
            return Err(replay_usage());
        }

        let mut options = Self {
            path: args[1].clone(),
            spread: 0.5,
            skew: 0.05,
            quantity: SimulationConfig::default().order_quantity,
            fee_rate: SimulationConfig::default().fee_rate,
        };

        let mut index = 2;
        while index < args.len() {
            let flag = args[index].as_str();
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("{flag} requires a value\n\n{}", replay_usage()))?;

            match flag {
                "--spread" => options.spread = parse_positive_f64(flag, value)?,
                "--skew" => options.skew = parse_non_negative_f64(flag, value)?,
                "--quantity" => options.quantity = parse_positive_f64(flag, value)?,
                "--fee-rate" => options.fee_rate = parse_non_negative_f64(flag, value)?,
                _ => {
                    return Err(format!(
                        "unknown replay option: {flag}\n\n{}",
                        replay_usage()
                    ));
                }
            }

            index += 2;
        }

        Ok(options)
    }
}

fn replay_usage() -> String {
    "usage: cargo run -- replay <events.csv> [--spread <value>] [--skew <value>] [--quantity <value>] [--fee-rate <value>]".to_string()
}

fn parse_positive_f64(name: &str, value: &str) -> Result<f64, String> {
    let parsed = parse_finite_f64(name, value)?;
    if parsed <= 0.0 {
        return Err(format!("{name} must be positive"));
    }

    Ok(parsed)
}

fn parse_non_negative_f64(name: &str, value: &str) -> Result<f64, String> {
    let parsed = parse_finite_f64(name, value)?;
    if parsed < 0.0 {
        return Err(format!("{name} must be non-negative"));
    }

    Ok(parsed)
}

fn parse_finite_f64(name: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("{name} must be a number"))?;
    if !parsed.is_finite() {
        return Err(format!("{name} must be finite"));
    }

    Ok(parsed)
}

fn replay_config(steps: usize, options: &ReplayOptions) -> SimulationConfig {
    SimulationConfig {
        steps,
        order_quantity: options.quantity,
        fee_rate: options.fee_rate,
        fill_model: FillModelConfig::DistanceIntensity {
            base_intensity: 0.12,
            distance_decay: 4.0,
            volatility_boost: 1.0,
        },
        ..SimulationConfig::default()
    }
}

fn print_replay_results(options: &ReplayOptions, result: &SimulationResult) {
    println!("Replay results");
    println!("data: {}", options.path);
    println!("spread: {:.4}", options.spread);
    println!("skew: {:.4}", options.skew);
    println!("quantity: {:.4}", options.quantity);
    println!("fee_rate: {:.6}", options.fee_rate);

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

fn replay_step_dataset_to_csv(name: &str, seed: u64, result: &SimulationResult) -> String {
    let mut csv = String::from(step_dataset_header());
    append_step_dataset_rows(&mut csv, name, "fixed_spread", seed, result);
    csv
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

    fn replay_args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn replay_options_use_defaults() {
        let options = ReplayOptions::parse(&replay_args(&["replay", "events.csv"])).unwrap();

        assert_eq!(options.path, "events.csv");
        assert_eq!(options.spread, 0.5);
        assert_eq!(options.skew, 0.05);
        assert_eq!(options.quantity, SimulationConfig::default().order_quantity);
        assert_eq!(options.fee_rate, SimulationConfig::default().fee_rate);
    }

    #[test]
    fn replay_options_parse_numeric_flags() {
        let options = ReplayOptions::parse(&replay_args(&[
            "replay",
            "events.csv",
            "--spread",
            "0.25",
            "--skew",
            "0.02",
            "--quantity",
            "0.10",
            "--fee-rate",
            "0.0005",
        ]))
        .unwrap();

        assert_eq!(options.spread, 0.25);
        assert_eq!(options.skew, 0.02);
        assert_eq!(options.quantity, 0.10);
        assert_eq!(options.fee_rate, 0.0005);
    }

    #[test]
    fn replay_options_reject_unknown_flags() {
        let error = ReplayOptions::parse(&replay_args(&["replay", "events.csv", "--risk", "1.0"]))
            .unwrap_err();

        assert!(error.contains("unknown replay option"));
    }

    #[test]
    fn replay_options_require_positive_quantity() {
        let error =
            ReplayOptions::parse(&replay_args(&["replay", "events.csv", "--quantity", "0.0"]))
                .unwrap_err();

        assert_eq!(error, "--quantity must be positive");
    }

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

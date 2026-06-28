use mm_engine::sweep::{SweepConfig, SweepResult, run_parameter_sweep, sweep_results_to_csv};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_PATH: &str = "configs/baseline_sweep.json";

fn main() -> Result<(), Box<dyn Error>> {
    let output_dir = Path::new("target").join("reports");
    fs::create_dir_all(&output_dir).expect("failed to create report output directory");

    let config_paths = config_paths_from_args();
    let mut summaries = Vec::new();

    for config_path in config_paths {
        let config = load_sweep_config(&config_path)?;
        let regime = config_name(&config, &config_path);
        let results = run_parameter_sweep(config);

        print_top_results(&regime, &config_path, &results);

        let csv_path = output_dir.join(format!("{regime}.csv"));
        fs::write(&csv_path, sweep_results_to_csv(&results)).expect("failed to write sweep CSV");

        if let Some(best_result) = results.first() {
            summaries.push(RegimeSummary::from_best_result(regime, best_result));
        }

        println!("\nwrote {}", csv_path.display());
    }

    let summary_path = output_dir.join("regime_summary.csv");
    fs::write(&summary_path, regime_summaries_to_csv(&summaries))
        .expect("failed to write regime summary CSV");

    println!("wrote {}", summary_path.display());

    Ok(())
}

fn config_paths_from_args() -> Vec<PathBuf> {
    let paths: Vec<PathBuf> = env::args().skip(1).map(PathBuf::from).collect();

    if paths.is_empty() {
        vec![PathBuf::from(DEFAULT_CONFIG_PATH)]
    } else {
        paths
    }
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

struct RegimeSummary {
    regime: String,
    strategy_type: String,
    best_spread: f64,
    best_volatility_coeff: Option<f64>,
    best_risk_aversion: Option<f64>,
    best_skew: Option<f64>,
    runs: usize,
    best_score: f64,
    best_score_std: f64,
    best_stable_score: f64,
    best_pnl: f64,
    best_pnl_std: f64,
    fills: f64,
    max_drawdown: f64,
    max_drawdown_std: f64,
    max_abs_inventory: f64,
    low_vol_steps: f64,
    normal_vol_steps: f64,
    high_vol_steps: f64,
}

impl RegimeSummary {
    fn from_best_result(regime: String, result: &SweepResult) -> Self {
        Self {
            regime,
            strategy_type: result.strategy.strategy_type().to_string(),
            best_spread: result.representative_spread(),
            best_volatility_coeff: result.representative_volatility_coeff(),
            best_risk_aversion: result.strategy.risk_aversion(),
            best_skew: result.representative_skew_coeff(),
            runs: result.runs,
            best_score: result.score,
            best_score_std: result.stability.score_std,
            best_stable_score: result.stable_score,
            best_pnl: result.metrics.final_pnl,
            best_pnl_std: result.stability.final_pnl_std,
            fills: result.metrics.total_fills,
            max_drawdown: result.metrics.max_drawdown,
            max_drawdown_std: result.stability.max_drawdown_std,
            max_abs_inventory: result.metrics.max_abs_inventory,
            low_vol_steps: result.metrics.low_vol_steps,
            normal_vol_steps: result.metrics.normal_vol_steps,
            high_vol_steps: result.metrics.high_vol_steps,
        }
    }
}

fn regime_summaries_to_csv(summaries: &[RegimeSummary]) -> String {
    let mut csv = String::from(
        "regime,strategy_type,best_spread,best_volatility_coeff,best_risk_aversion,best_skew,runs,best_score,best_score_std,best_stable_score,avg_best_pnl,best_pnl_std,avg_fills,avg_max_drawdown,max_drawdown_std,avg_max_abs_inventory,avg_low_vol_steps,avg_normal_vol_steps,avg_high_vol_steps\n",
    );

    for summary in summaries {
        csv.push_str(&format!(
            "{},{},{:.6},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            summary.regime,
            summary.strategy_type,
            summary.best_spread,
            optional_csv_f64(summary.best_volatility_coeff),
            optional_csv_f64(summary.best_risk_aversion),
            optional_csv_f64(summary.best_skew),
            summary.runs,
            summary.best_score,
            summary.best_score_std,
            summary.best_stable_score,
            summary.best_pnl,
            summary.best_pnl_std,
            summary.fills,
            summary.max_drawdown,
            summary.max_drawdown_std,
            summary.max_abs_inventory,
            summary.low_vol_steps,
            summary.normal_vol_steps,
            summary.high_vol_steps,
        ));
    }

    csv
}

fn optional_csv_f64(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

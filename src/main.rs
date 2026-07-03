use mm_engine::engine::dataset::{append_step_dataset_rows, step_dataset_header};
use mm_engine::engine::metrics::SimulationMetrics;
use mm_engine::engine::simulation::{
    FillModelConfig, SimulationConfig, SimulationResult, run_replay_simulation, run_simulation,
};
use mm_engine::market::{InMemoryMarketData, MarketEvent, market_events_from_csv};
use mm_engine::strategy::market_maker::StrategyParams;
use mm_engine::sweep::{SweepConfig, SweepResult, run_parameter_sweep, sweep_results_to_csv};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_PATH: &str = "configs/baseline_sweep.json";
const QUOTE_DISTANCE_SCORE_WEIGHT: f64 = 0.1;
const REGIME_SUMMARY_HEADER: &[&str] = &[
    "regime",
    "strategy_type",
    "best_spread",
    "best_volatility_coeff",
    "best_risk_aversion",
    "best_liquidity_depth",
    "best_horizon",
    "best_inventory_limit",
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
    if args.first().is_some_and(|arg| arg == "replay-sweep") {
        return run_replay_sweep_command(&args);
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

fn run_replay_sweep_command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let options = ReplaySweepOptions::parse(args)?;

    let events = market_events_from_csv(&options.path)?;
    if events.is_empty() {
        return Err("replay CSV contains no events".into());
    }

    let results = run_replay_sweep(events, &options);
    print_top_replay_sweep_results(&options, &results);

    let output_dir = Path::new("target").join("reports");
    fs::create_dir_all(&output_dir).expect("failed to create report output directory");
    let output_path = output_dir.join(format!(
        "{}_replay_sweep.csv",
        file_stem(Path::new(&options.path))
    ));
    fs::write(&output_path, replay_sweep_results_to_csv(&results))
        .expect("failed to write replay sweep CSV");

    println!("wrote {}", output_path.display());

    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct ReplaySweepOptions {
    path: String,
    seeds: Vec<u64>,
    spreads: Vec<f64>,
    skews: Vec<f64>,
    quantities: Vec<f64>,
    fee_rate: f64,
}

impl ReplaySweepOptions {
    fn parse(args: &[String]) -> Result<Self, String> {
        if args.len() < 2 {
            return Err(replay_sweep_usage());
        }

        let mut options = Self::default_for_path(args[1].clone());

        let mut index = 2;
        while index < args.len() {
            let flag = args[index].as_str();
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("{flag} requires a value\n\n{}", replay_sweep_usage()))?;

            match flag {
                "--seeds" => options.seeds = parse_u64_list(flag, value)?,
                "--spreads" => options.spreads = parse_positive_f64_list(flag, value)?,
                "--skews" => options.skews = parse_non_negative_f64_list(flag, value)?,
                "--quantities" => options.quantities = parse_positive_f64_list(flag, value)?,
                "--fee-rate" => options.fee_rate = parse_non_negative_f64(flag, value)?,
                _ => {
                    return Err(format!(
                        "unknown replay-sweep option: {flag}\n\n{}",
                        replay_sweep_usage()
                    ));
                }
            }

            index += 2;
        }

        Ok(options)
    }

    fn default_for_path(path: String) -> Self {
        Self {
            path,
            seeds: vec![SimulationConfig::default().seed],
            spreads: vec![0.2, 0.5, 1.0],
            skews: vec![0.0, 0.02, 0.05],
            quantities: vec![0.05, 0.1, 0.2],
            fee_rate: SimulationConfig::default().fee_rate,
        }
    }

    fn run_count(&self) -> usize {
        self.parameter_count() * self.seeds.len()
    }

    fn parameter_count(&self) -> usize {
        self.spreads.len() * self.skews.len() * self.quantities.len()
    }
}

fn replay_sweep_usage() -> String {
    "usage: cargo run -- replay-sweep <events.csv> [--seeds <a,b,c>] [--spreads <a,b,c>] [--skews <a,b,c>] [--quantities <a,b,c>] [--fee-rate <value>]".to_string()
}

fn parse_u64_list(name: &str, value: &str) -> Result<Vec<u64>, String> {
    let values: Result<Vec<u64>, String> = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            if part.is_empty() {
                return Err(format!("{name} contains an empty value"));
            }

            part.parse::<u64>()
                .map_err(|_| format!("{name} must contain unsigned integers"))
        })
        .collect();
    let values = values?;

    if values.is_empty() {
        return Err(format!("{name} must contain at least one value"));
    }

    Ok(values)
}

fn parse_positive_f64_list(name: &str, value: &str) -> Result<Vec<f64>, String> {
    parse_f64_list(name, value, parse_positive_f64)
}

fn parse_non_negative_f64_list(name: &str, value: &str) -> Result<Vec<f64>, String> {
    parse_f64_list(name, value, parse_non_negative_f64)
}

fn parse_f64_list(
    name: &str,
    value: &str,
    parser: fn(&str, &str) -> Result<f64, String>,
) -> Result<Vec<f64>, String> {
    let values: Result<Vec<f64>, String> = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            if part.is_empty() {
                return Err(format!("{name} contains an empty value"));
            }

            parser(name, part)
        })
        .collect();
    let values = values?;

    if values.is_empty() {
        return Err(format!("{name} must contain at least one value"));
    }

    Ok(values)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReplaySweepResult {
    spread: f64,
    skew: f64,
    quantity: f64,
    fee_rate: f64,
    runs: usize,
    metrics: ReplaySweepMetrics,
    inactivity_penalty: f64,
    score: f64,
    score_std: f64,
    stable_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReplaySweepMetrics {
    final_pnl: f64,
    final_pnl_std: f64,
    min_pnl: f64,
    max_pnl: f64,
    max_drawdown: f64,
    max_drawdown_std: f64,
    final_inventory: f64,
    max_abs_inventory: f64,
    avg_abs_inventory: f64,
    total_fills: f64,
    buy_fills: f64,
    sell_fills: f64,
    traded_quantity: f64,
    traded_notional: f64,
    total_fees: f64,
    total_adverse_selection: f64,
    low_vol_steps: f64,
    normal_vol_steps: f64,
    high_vol_steps: f64,
    observed_quote_steps: f64,
    avg_bid_distance_to_observed_ask: f64,
    avg_ask_distance_to_observed_bid: f64,
    avg_quote_distance: f64,
}

fn run_replay_sweep(
    events: Vec<MarketEvent>,
    options: &ReplaySweepOptions,
) -> Vec<ReplaySweepResult> {
    let mut results = Vec::new();

    for spread in &options.spreads {
        for skew in &options.skews {
            for quantity in &options.quantities {
                let strategy = StrategyParams {
                    spread: *spread,
                    skew_coeff: *skew,
                };
                let mut metrics_by_seed = Vec::new();
                let mut scores_by_seed = Vec::new();
                let mut inactivity_by_seed = Vec::new();
                let mut quote_distance_by_seed = Vec::new();

                for seed in &options.seeds {
                    let replay_options = ReplayOptions {
                        path: "replay_sweep".to_string(),
                        spread: *spread,
                        skew: *skew,
                        quantity: *quantity,
                        fee_rate: options.fee_rate,
                    };
                    let mut config = replay_config(events.len(), &replay_options);
                    config.seed = *seed;
                    let mut data = InMemoryMarketData::new(events.clone());
                    let result = run_replay_simulation(config, &mut data, &strategy);

                    if let Some(metrics) = SimulationMetrics::from_result(&result) {
                        let inactivity_penalty = replay_inactivity_penalty(metrics);
                        let quote_distance = replay_quote_distance(&result);
                        let score = replay_score(metrics, inactivity_penalty, quote_distance);
                        metrics_by_seed.push(metrics);
                        scores_by_seed.push(score);
                        inactivity_by_seed.push(inactivity_penalty);
                        quote_distance_by_seed.push(quote_distance);
                    }
                }

                if !metrics_by_seed.is_empty() {
                    let score = mean(&scores_by_seed);
                    let score_std = std_dev(&scores_by_seed);
                    results.push(ReplaySweepResult {
                        spread: *spread,
                        skew: *skew,
                        quantity: *quantity,
                        fee_rate: options.fee_rate,
                        runs: metrics_by_seed.len(),
                        metrics: aggregate_replay_metrics(
                            &metrics_by_seed,
                            &quote_distance_by_seed,
                        ),
                        inactivity_penalty: mean(&inactivity_by_seed),
                        score,
                        score_std,
                        stable_score: score - 0.25 * score_std,
                    });
                }
            }
        }
    }

    results.sort_by(|a, b| b.stable_score.total_cmp(&a.stable_score));
    results
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReplayQuoteDistance {
    observed_quote_steps: usize,
    avg_bid_distance_to_observed_ask: f64,
    avg_ask_distance_to_observed_bid: f64,
    avg_quote_distance: f64,
}

fn aggregate_replay_metrics(
    metrics: &[SimulationMetrics],
    quote_distances: &[ReplayQuoteDistance],
) -> ReplaySweepMetrics {
    let final_pnls: Vec<f64> = metrics.iter().map(|metric| metric.final_pnl).collect();
    let max_drawdowns: Vec<f64> = metrics.iter().map(|metric| metric.max_drawdown).collect();
    let observed_quote_steps: Vec<f64> = quote_distances
        .iter()
        .map(|distance| distance.observed_quote_steps as f64)
        .collect();
    let bid_distances: Vec<f64> = quote_distances
        .iter()
        .map(|distance| distance.avg_bid_distance_to_observed_ask)
        .collect();
    let ask_distances: Vec<f64> = quote_distances
        .iter()
        .map(|distance| distance.avg_ask_distance_to_observed_bid)
        .collect();
    let quote_distances: Vec<f64> = quote_distances
        .iter()
        .map(|distance| distance.avg_quote_distance)
        .collect();

    ReplaySweepMetrics {
        final_pnl: mean(&final_pnls),
        final_pnl_std: std_dev(&final_pnls),
        min_pnl: mean_metric(metrics, |metric| metric.min_pnl),
        max_pnl: mean_metric(metrics, |metric| metric.max_pnl),
        max_drawdown: mean(&max_drawdowns),
        max_drawdown_std: std_dev(&max_drawdowns),
        final_inventory: mean_metric(metrics, |metric| metric.final_inventory),
        max_abs_inventory: mean_metric(metrics, |metric| metric.max_abs_inventory),
        avg_abs_inventory: mean_metric(metrics, |metric| metric.avg_abs_inventory),
        total_fills: mean_metric(metrics, |metric| metric.total_fills as f64),
        buy_fills: mean_metric(metrics, |metric| metric.buy_fills as f64),
        sell_fills: mean_metric(metrics, |metric| metric.sell_fills as f64),
        traded_quantity: mean_metric(metrics, |metric| metric.traded_quantity),
        traded_notional: mean_metric(metrics, |metric| metric.traded_notional),
        total_fees: mean_metric(metrics, |metric| metric.total_fees),
        total_adverse_selection: mean_metric(metrics, |metric| metric.total_adverse_selection),
        low_vol_steps: mean_metric(metrics, |metric| metric.low_vol_steps as f64),
        normal_vol_steps: mean_metric(metrics, |metric| metric.normal_vol_steps as f64),
        high_vol_steps: mean_metric(metrics, |metric| metric.high_vol_steps as f64),
        observed_quote_steps: mean(&observed_quote_steps),
        avg_bid_distance_to_observed_ask: mean(&bid_distances),
        avg_ask_distance_to_observed_bid: mean(&ask_distances),
        avg_quote_distance: mean(&quote_distances),
    }
}

fn mean_metric(metrics: &[SimulationMetrics], value: fn(SimulationMetrics) -> f64) -> f64 {
    let values: Vec<f64> = metrics.iter().copied().map(value).collect();
    mean(&values)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}

fn replay_score(
    metrics: SimulationMetrics,
    inactivity_penalty: f64,
    quote_distance: ReplayQuoteDistance,
) -> f64 {
    metrics.final_pnl
        - 2.0 * metrics.max_drawdown
        - metrics.max_abs_inventory
        - inactivity_penalty
        - quote_distance_penalty(quote_distance)
}

fn quote_distance_penalty(quote_distance: ReplayQuoteDistance) -> f64 {
    if quote_distance.observed_quote_steps == 0 {
        0.0
    } else {
        QUOTE_DISTANCE_SCORE_WEIGHT * quote_distance.avg_quote_distance.max(0.0)
    }
}

fn replay_quote_distance(result: &SimulationResult) -> ReplayQuoteDistance {
    let mut observed_quote_steps = 0;
    let mut bid_distance = 0.0;
    let mut ask_distance = 0.0;

    for step in &result.steps {
        let Some(observed_quote) = step.observed_quote else {
            continue;
        };

        observed_quote_steps += 1;
        bid_distance += observed_quote.ask - step.quote.bid;
        ask_distance += step.quote.ask - observed_quote.bid;
    }

    if observed_quote_steps == 0 {
        return ReplayQuoteDistance {
            observed_quote_steps,
            avg_bid_distance_to_observed_ask: 0.0,
            avg_ask_distance_to_observed_bid: 0.0,
            avg_quote_distance: 0.0,
        };
    }

    let avg_bid_distance_to_observed_ask = bid_distance / observed_quote_steps as f64;
    let avg_ask_distance_to_observed_bid = ask_distance / observed_quote_steps as f64;

    ReplayQuoteDistance {
        observed_quote_steps,
        avg_bid_distance_to_observed_ask,
        avg_ask_distance_to_observed_bid,
        avg_quote_distance: (avg_bid_distance_to_observed_ask + avg_ask_distance_to_observed_bid)
            / 2.0,
    }
}

fn replay_inactivity_penalty(metrics: SimulationMetrics) -> f64 {
    let min_fills = (metrics.steps / 20).max(1);
    let missing_fills = min_fills.saturating_sub(metrics.total_fills);

    missing_fills as f64 * 0.25
}

fn print_top_replay_sweep_results(options: &ReplaySweepOptions, results: &[ReplaySweepResult]) {
    println!("Replay sweep results");
    println!("data: {}", options.path);
    println!(
        "grid: spreads={} skews={} quantities={} seeds={} parameter_sets={} runs={}",
        options.spreads.len(),
        options.skews.len(),
        options.quantities.len(),
        options.seeds.len(),
        options.parameter_count(),
        options.run_count()
    );
    println!("fee_rate: {:.6}", options.fee_rate);
    println!(
        "{:<4} {:>5} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "rank",
        "runs",
        "spread",
        "skew",
        "qty",
        "fee_rate",
        "avg_pnl",
        "fills",
        "fees",
        "drawdown",
        "q_dist",
        "score_sd",
        "stable"
    );

    for (index, result) in results.iter().take(10).enumerate() {
        let metrics = result.metrics;
        println!(
            "{:<4} {:>5} {:>8.2} {:>8.2} {:>8.2} {:>9.4} {:>8.2} {:>8.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            index + 1,
            result.runs,
            result.spread,
            result.skew,
            result.quantity,
            result.fee_rate,
            metrics.final_pnl,
            metrics.total_fills,
            metrics.total_fees,
            metrics.max_drawdown,
            metrics.avg_quote_distance,
            result.score_std,
            result.stable_score,
        );
    }
}

fn replay_sweep_results_to_csv(results: &[ReplaySweepResult]) -> String {
    let mut csv = String::from(
        "rank,spread,skew,quantity,fee_rate,runs,score,score_std,stable_score,inactivity_penalty,quote_distance_penalty,avg_final_pnl,final_pnl_std,avg_min_pnl,avg_max_pnl,avg_max_drawdown,max_drawdown_std,avg_final_inventory,avg_max_abs_inventory,avg_abs_inventory,avg_total_fills,avg_buy_fills,avg_sell_fills,avg_traded_quantity,avg_traded_notional,avg_total_fees,avg_total_adverse_selection,avg_low_vol_steps,avg_normal_vol_steps,avg_high_vol_steps,avg_observed_quote_steps,avg_bid_distance_to_observed_ask,avg_ask_distance_to_observed_bid,avg_quote_distance\n",
    );

    for (index, result) in results.iter().enumerate() {
        let metrics = result.metrics;
        let row = vec![
            (index + 1).to_string(),
            format_csv_f64(result.spread),
            format_csv_f64(result.skew),
            format_csv_f64(result.quantity),
            format_csv_f64(result.fee_rate),
            result.runs.to_string(),
            format_csv_f64(result.score),
            format_csv_f64(result.score_std),
            format_csv_f64(result.stable_score),
            format_csv_f64(result.inactivity_penalty),
            format_csv_f64(quote_distance_penalty(ReplayQuoteDistance {
                observed_quote_steps: metrics.observed_quote_steps as usize,
                avg_bid_distance_to_observed_ask: metrics.avg_bid_distance_to_observed_ask,
                avg_ask_distance_to_observed_bid: metrics.avg_ask_distance_to_observed_bid,
                avg_quote_distance: metrics.avg_quote_distance,
            })),
            format_csv_f64(metrics.final_pnl),
            format_csv_f64(metrics.final_pnl_std),
            format_csv_f64(metrics.min_pnl),
            format_csv_f64(metrics.max_pnl),
            format_csv_f64(metrics.max_drawdown),
            format_csv_f64(metrics.max_drawdown_std),
            format_csv_f64(metrics.final_inventory),
            format_csv_f64(metrics.max_abs_inventory),
            format_csv_f64(metrics.avg_abs_inventory),
            format_csv_f64(metrics.total_fills),
            format_csv_f64(metrics.buy_fills),
            format_csv_f64(metrics.sell_fills),
            format_csv_f64(metrics.traded_quantity),
            format_csv_f64(metrics.traded_notional),
            format_csv_f64(metrics.total_fees),
            format_csv_f64(metrics.total_adverse_selection),
            format_csv_f64(metrics.low_vol_steps),
            format_csv_f64(metrics.normal_vol_steps),
            format_csv_f64(metrics.high_vol_steps),
            format_csv_f64(metrics.observed_quote_steps),
            format_csv_f64(metrics.avg_bid_distance_to_observed_ask),
            format_csv_f64(metrics.avg_ask_distance_to_observed_bid),
            format_csv_f64(metrics.avg_quote_distance),
        ];

        csv.push_str(&row.join(","));
        csv.push('\n');
    }

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
        "{:<4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "rank",
        "runs",
        "spread",
        "vol",
        "risk",
        "depth",
        "horizon",
        "inv_lim",
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
            "{:<4} {:>5} {:>8.2} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8.2} {:>8.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            index + 1,
            result.runs,
            result.representative_spread(),
            optional_f64(result.representative_volatility_coeff()),
            optional_f64(result.strategy.risk_aversion()),
            optional_f64(result.strategy.liquidity_depth()),
            optional_f64(result.strategy.horizon()),
            optional_f64(result.strategy.inventory_limit()),
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
        optional_csv_f64(result.strategy.liquidity_depth()),
        optional_csv_f64(result.strategy.horizon()),
        optional_csv_f64(result.strategy.inventory_limit()),
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
    fn replay_sweep_options_use_defaults() {
        let options =
            ReplaySweepOptions::parse(&replay_args(&["replay-sweep", "events.csv"])).unwrap();

        assert_eq!(options.path, "events.csv");
        assert_eq!(options.spreads, vec![0.2, 0.5, 1.0]);
        assert_eq!(options.skews, vec![0.0, 0.02, 0.05]);
        assert_eq!(options.quantities, vec![0.05, 0.1, 0.2]);
        assert_eq!(options.seeds, vec![SimulationConfig::default().seed]);
        assert_eq!(options.fee_rate, SimulationConfig::default().fee_rate);
        assert_eq!(options.run_count(), 27);
    }

    #[test]
    fn replay_sweep_options_parse_custom_grid() {
        let options = ReplaySweepOptions::parse(&replay_args(&[
            "replay-sweep",
            "events.csv",
            "--seeds",
            "42,43",
            "--spreads",
            "0.25,0.50",
            "--skews",
            "0.00,0.03",
            "--quantities",
            "0.10,0.20",
            "--fee-rate",
            "0.0005",
        ]))
        .unwrap();

        assert_eq!(options.seeds, vec![42, 43]);
        assert_eq!(options.spreads, vec![0.25, 0.50]);
        assert_eq!(options.skews, vec![0.00, 0.03]);
        assert_eq!(options.quantities, vec![0.10, 0.20]);
        assert_eq!(options.fee_rate, 0.0005);
        assert_eq!(options.parameter_count(), 8);
        assert_eq!(options.run_count(), 16);
    }

    #[test]
    fn replay_sweep_options_reject_invalid_seeds() {
        let error = ReplaySweepOptions::parse(&replay_args(&[
            "replay-sweep",
            "events.csv",
            "--seeds",
            "42,nope",
        ]))
        .unwrap_err();

        assert_eq!(error, "--seeds must contain unsigned integers");
    }

    #[test]
    fn replay_sweep_options_reject_invalid_grid_values() {
        let error = ReplaySweepOptions::parse(&replay_args(&[
            "replay-sweep",
            "events.csv",
            "--spreads",
            "0.2,0.0",
        ]))
        .unwrap_err();

        assert_eq!(error, "--spreads must be positive");
    }

    #[test]
    fn replay_sweep_options_reject_empty_grid_values() {
        let error = ReplaySweepOptions::parse(&replay_args(&[
            "replay-sweep",
            "events.csv",
            "--quantities",
            "0.1,",
        ]))
        .unwrap_err();

        assert_eq!(error, "--quantities contains an empty value");
    }

    #[test]
    fn replay_sweep_runs_fixed_grid() {
        let events = vec![
            mm_engine::market::MarketEvent::from_mid(1, 100.0),
            mm_engine::market::MarketEvent::from_mid(2, 100.1),
            mm_engine::market::MarketEvent::from_mid(3, 100.0),
            mm_engine::market::MarketEvent::from_mid(4, 100.2),
        ];
        let options = ReplaySweepOptions::default_for_path("events.csv".to_string());

        let results = run_replay_sweep(events, &options);

        assert_eq!(results.len(), 27);
        assert!(
            results
                .windows(2)
                .all(|window| window[0].stable_score >= window[1].stable_score)
        );
    }

    #[test]
    fn replay_sweep_aggregates_multiple_seeds() {
        let events = vec![
            mm_engine::market::MarketEvent::from_mid(1, 100.0),
            mm_engine::market::MarketEvent::from_mid(2, 100.1),
            mm_engine::market::MarketEvent::from_mid(3, 100.0),
            mm_engine::market::MarketEvent::from_mid(4, 100.2),
        ];
        let mut options = ReplaySweepOptions::default_for_path("events.csv".to_string());
        options.seeds = vec![42, 43, 44];
        options.spreads = vec![0.5];
        options.skews = vec![0.0];
        options.quantities = vec![0.1];

        let results = run_replay_sweep(events, &options);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].runs, 3);
        assert!(results[0].score_std >= 0.0);
        assert!(results[0].metrics.final_pnl_std >= 0.0);
    }

    #[test]
    fn replay_sweep_penalizes_observed_quote_distance() {
        let events = vec![mm_engine::market::MarketEvent {
            timestamp_ms: 1,
            mid_price: 100.0,
            bid: Some(99.99),
            ask: Some(100.01),
        }];
        let mut options = ReplaySweepOptions::default_for_path("events.csv".to_string());
        options.seeds = vec![42];
        options.spreads = vec![0.5, 1.0];
        options.skews = vec![0.0];
        options.quantities = vec![0.1];

        let results = run_replay_sweep(events, &options);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].spread, 0.5);
        assert!(results[0].metrics.avg_quote_distance < results[1].metrics.avg_quote_distance);
        assert!(results[0].stable_score > results[1].stable_score);
    }

    #[test]
    fn replay_sweep_csv_writes_ranked_rows() {
        let events = vec![
            mm_engine::market::MarketEvent::from_mid(1, 100.0),
            mm_engine::market::MarketEvent::from_mid(2, 100.1),
        ];
        let options = ReplaySweepOptions::default_for_path("events.csv".to_string());
        let results = run_replay_sweep(events, &options);
        let csv = replay_sweep_results_to_csv(&results);
        let mut lines = csv.lines();
        let header_width = lines.next().unwrap().split(',').count();
        let first_row: Vec<&str> = lines.next().unwrap().split(',').collect();

        assert_eq!(first_row[0], "1");
        assert_eq!(first_row.len(), header_width);
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

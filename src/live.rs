use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::agent::RuleBasedControllerParams;
use crate::engine::simulation::RegimeConfig;
use crate::market::MarketEvent;
use crate::paper::{
    PaperFillModelConfig, PaperSessionConfig, PaperSessionResult, PaperSessionRunner,
    paper_session_csv_header, paper_session_row_to_csv,
};

#[derive(Debug, Clone, PartialEq)]
pub struct PaperLiveConfig {
    pub pair: String,
    pub output: Option<String>,
    pub controller: RuleBasedControllerParams,
    pub quantity: f64,
    pub fee_rate: f64,
    pub seed: u64,
    pub fill_model: PaperFillModelConfig,
    pub volatility_window: usize,
    pub regime: RegimeConfig,
    pub samples: usize,
    pub interval_seconds: f64,
    pub timeout_seconds: f64,
}

pub fn run_kraken_paper_live(config: PaperLiveConfig) -> Result<PathBuf, Box<dyn Error>> {
    let output_path = config
        .output
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new("target").join("reports").join(format!(
                "kraken_{}_paper_live.csv",
                config.pair.to_ascii_lowercase()
            ))
        });

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("failed to create paper live output directory");
    }

    let file = fs::File::create(&output_path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", paper_session_csv_header())?;

    let mut runner = PaperSessionRunner::new(PaperSessionConfig {
        order_quantity: config.quantity,
        fee_rate: config.fee_rate,
        seed: config.seed,
        fill_model: config.fill_model,
        volatility_window: config.volatility_window,
        regime: config.regime,
    });
    let mut rows = Vec::with_capacity(config.samples);

    println!("Paper live session");
    println!("pair: {}", config.pair);
    println!("samples: {}", config.samples);
    println!("interval_seconds: {:.3}", config.interval_seconds);
    println!("output: {}", output_path.display());

    for sample_index in 0..config.samples {
        let event = fetch_kraken_top_of_book(&config.pair, config.timeout_seconds)?;
        let row = runner.step(event, &config.controller);
        writeln!(writer, "{}", paper_session_row_to_csv(&row))?;
        writer.flush()?;

        println!(
            "sample={}/{} bid={:.6} ask={:.6} fills={} inventory={:.4} pnl={:.4}",
            sample_index + 1,
            config.samples,
            row.observed_bid.unwrap_or_default(),
            row.observed_ask.unwrap_or_default(),
            row.fills,
            row.inventory,
            row.pnl
        );

        rows.push(row);

        if sample_index + 1 < config.samples && config.interval_seconds > 0.0 {
            thread::sleep(Duration::from_secs_f64(config.interval_seconds));
        }
    }

    let result = PaperSessionResult { rows };
    print_paper_live_results(&config, &result);
    println!("wrote {}", output_path.display());

    Ok(output_path)
}

fn fetch_kraken_top_of_book(
    pair: &str,
    timeout_seconds: f64,
) -> Result<MarketEvent, Box<dyn Error>> {
    let url = format!(
        "https://api.kraken.com/0/public/Depth?pair={}&count=1",
        pair.to_ascii_uppercase()
    );
    let response = ureq::get(&url)
        .timeout(Duration::from_secs_f64(timeout_seconds))
        .call()?;
    let payload: serde_json::Value = response.into_json()?;

    if let Some(errors) = payload.get("error").and_then(|value| value.as_array()) {
        if !errors.is_empty() {
            let message = errors
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("Kraken API error: {message}").into());
        }
    }

    let result = payload
        .get("result")
        .and_then(|value| value.as_object())
        .ok_or("Kraken response missing result object")?;
    let book = result
        .values()
        .find_map(|value| value.as_object())
        .ok_or("Kraken response contains no order book")?;
    let bids = book
        .get("bids")
        .and_then(|value| value.as_array())
        .filter(|values| !values.is_empty())
        .ok_or("Kraken response contains no bids")?;
    let asks = book
        .get("asks")
        .and_then(|value| value.as_array())
        .filter(|values| !values.is_empty())
        .ok_or("Kraken response contains no asks")?;
    let bid = parse_kraken_price(&bids[0], "bid")?;
    let ask = parse_kraken_price(&asks[0], "ask")?;
    let timestamp_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;

    Ok(MarketEvent::from_quote(timestamp_ms, bid, ask))
}

fn parse_kraken_price(level: &serde_json::Value, side: &str) -> Result<f64, Box<dyn Error>> {
    let price = level
        .as_array()
        .and_then(|values| values.first())
        .ok_or_else(|| format!("invalid {side} level: {level}"))?;

    if let Some(value) = price.as_f64() {
        return Ok(value);
    }

    let value = price
        .as_str()
        .ok_or_else(|| format!("invalid {side} price: {price}"))?;

    value
        .parse::<f64>()
        .map_err(|error| format!("invalid {side} price: {value}: {error}").into())
}

fn print_paper_live_results(config: &PaperLiveConfig, result: &PaperSessionResult) {
    println!("Paper live results");
    println!("pair: {}", config.pair);
    println!("quantity: {:.4}", config.quantity);
    println!("fee_rate: {:.6}", config.fee_rate);
    println!("seed: {}", config.seed);
    println!("fill_model: {:?}", config.fill_model);
    println!("steps: {}", result.rows.len());

    let Some(final_row) = result.final_row() else {
        return;
    };

    let total_fills: usize = result.rows.iter().map(|row| row.fills).sum();
    let total_fees: f64 = result.rows.iter().map(|row| row.fees).sum();
    let max_drawdown = result
        .rows
        .iter()
        .map(|row| row.drawdown)
        .fold(0.0, f64::max);

    println!("fills: {}", total_fills);
    println!("final_pnl: {:.2}", final_row.pnl);
    println!("final_inventory: {:.2}", final_row.inventory);
    println!("fees: {:.2}", total_fees);
    println!("max_drawdown: {:.2}", max_drawdown);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kraken_string_price() {
        let price = serde_json::json!(["81.42", "1.5", "123"]);

        assert_eq!(parse_kraken_price(&price, "bid").unwrap(), 81.42);
    }
}

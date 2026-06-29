use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarketEvent {
    pub timestamp_ms: u64,
    pub mid_price: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
}

impl MarketEvent {
    pub fn from_mid(timestamp_ms: u64, mid_price: f64) -> Self {
        Self {
            timestamp_ms,
            mid_price,
            bid: None,
            ask: None,
        }
    }

    pub fn from_quote(timestamp_ms: u64, bid: f64, ask: f64) -> Self {
        Self {
            timestamp_ms,
            mid_price: (bid + ask) / 2.0,
            bid: Some(bid),
            ask: Some(ask),
        }
    }
}

pub trait MarketDataSource {
    fn next_event(&mut self) -> Option<MarketEvent>;
}

#[derive(Debug, Clone)]
pub struct InMemoryMarketData {
    events: Vec<MarketEvent>,
    next_index: usize,
}

impl InMemoryMarketData {
    pub fn new(events: Vec<MarketEvent>) -> Self {
        Self {
            events,
            next_index: 0,
        }
    }

    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self> {
        market_events_from_csv(path).map(Self::new)
    }
}

impl MarketDataSource for InMemoryMarketData {
    fn next_event(&mut self) -> Option<MarketEvent> {
        if self.next_index >= self.events.len() {
            return None;
        }

        let event = self.events[self.next_index];
        self.next_index += 1;
        Some(event)
    }
}

pub fn market_events_from_csv(path: impl AsRef<Path>) -> Result<Vec<MarketEvent>> {
    let contents = fs::read_to_string(path)?;
    let mut lines = contents.lines();
    let header = lines
        .next()
        .ok_or_else(|| invalid_data("market data CSV is empty"))?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();

    let timestamp_index = required_column(&columns, "timestamp_ms")?;
    let mid_index = required_column(&columns, "mid_price")?;
    let bid_index = optional_column(&columns, "bid");
    let ask_index = optional_column(&columns, "ask");
    let mut events = Vec::new();

    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let row_number = line_index + 2;
        let values: Vec<&str> = line.split(',').map(str::trim).collect();
        let timestamp_ms = parse_required_u64(&values, timestamp_index, row_number)?;
        let mid_price = parse_required_f64(&values, mid_index, row_number)?;
        let bid = parse_optional_f64(&values, bid_index, row_number)?;
        let ask = parse_optional_f64(&values, ask_index, row_number)?;

        events.push(MarketEvent {
            timestamp_ms,
            mid_price,
            bid,
            ask,
        });
    }

    Ok(events)
}

fn required_column(columns: &[&str], name: &str) -> Result<usize> {
    optional_column(columns, name).ok_or_else(|| invalid_data(format!("missing column '{name}'")))
}

fn optional_column(columns: &[&str], name: &str) -> Option<usize> {
    columns.iter().position(|column| *column == name)
}

fn parse_required_u64(values: &[&str], index: usize, row_number: usize) -> Result<u64> {
    values
        .get(index)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid_data(format!("missing timestamp_ms at row {row_number}")))?
        .parse()
        .map_err(|_| invalid_data(format!("invalid timestamp_ms at row {row_number}")))
}

fn parse_required_f64(values: &[&str], index: usize, row_number: usize) -> Result<f64> {
    values
        .get(index)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid_data(format!("missing mid_price at row {row_number}")))?
        .parse()
        .map_err(|_| invalid_data(format!("invalid mid_price at row {row_number}")))
}

fn parse_optional_f64(
    values: &[&str],
    index: Option<usize>,
    row_number: usize,
) -> Result<Option<f64>> {
    let Some(index) = index else {
        return Ok(None);
    };
    let Some(value) = values.get(index).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };

    value
        .parse()
        .map(Some)
        .map_err(|_| invalid_data(format!("invalid optional price at row {row_number}")))
}

fn invalid_data(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quote {
    pub bid: f64,
    pub ask: f64,
}

impl Quote {
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    pub side: FillSide,
    pub price: f64,
    pub quantity: f64,
    pub fee: f64,
}

impl Fill {
    pub fn buy(price: f64, quantity: f64, fee: f64) -> Self {
        Self {
            side: FillSide::Buy,
            price,
            quantity,
            fee,
        }
    }

    pub fn sell(price: f64, quantity: f64, fee: f64) -> Self {
        Self {
            side: FillSide::Sell,
            price,
            quantity,
            fee,
        }
    }

    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn quoted_market_event_sets_mid_price() {
        let event = MarketEvent::from_quote(10, 99.0, 101.0);

        assert_eq!(event.timestamp_ms, 10);
        assert_eq!(event.mid_price, 100.0);
        assert_eq!(event.bid, Some(99.0));
        assert_eq!(event.ask, Some(101.0));
    }

    #[test]
    fn in_memory_market_data_returns_events_in_order() {
        let mut source = InMemoryMarketData::new(vec![
            MarketEvent::from_mid(1, 100.0),
            MarketEvent::from_mid(2, 101.0),
        ]);

        assert_eq!(source.next_event().unwrap().mid_price, 100.0);
        assert_eq!(source.next_event().unwrap().mid_price, 101.0);
        assert_eq!(source.next_event(), None);
    }

    #[test]
    fn market_events_load_from_csv() {
        let path = temp_csv_path("market_events_load_from_csv");
        fs::write(
            &path,
            "timestamp_ms,mid_price,bid,ask\n1,100.0,99.9,100.1\n2,101.0,,\n",
        )
        .unwrap();

        let events = market_events_from_csv(&path).unwrap();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].timestamp_ms, 1);
        assert_eq!(events[0].mid_price, 100.0);
        assert_eq!(events[0].bid, Some(99.9));
        assert_eq!(events[0].ask, Some(100.1));
        assert_eq!(events[1].bid, None);
        assert_eq!(events[1].ask, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn market_events_require_mid_price_column() {
        let path = temp_csv_path("market_events_require_mid_price_column");
        fs::write(&path, "timestamp_ms,bid,ask\n1,99.9,100.1\n").unwrap();

        let error = market_events_from_csv(&path).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData);

        fs::remove_file(path).unwrap();
    }

    fn temp_csv_path(test_name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mm_engine_{test_name}_{nanos}.csv"))
    }
}

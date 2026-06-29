use serde::{Deserialize, Serialize};

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
}

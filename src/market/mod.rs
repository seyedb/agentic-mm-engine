use serde::{Deserialize, Serialize};

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

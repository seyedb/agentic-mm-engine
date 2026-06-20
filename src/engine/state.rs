use serde::{Deserialize, Serialize};

use crate::market::{Fill, FillSide};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub mid_price: f64,
    pub inventory: f64,
    pub cash: f64,
    pub pnl: f64,
}

impl SystemState {
    pub fn new(mid_price: f64) -> Self {
        Self {
            mid_price,
            inventory: 0.0,
            cash: 0.0,
            pnl: 0.0,
        }
    }

    pub fn apply_fill(&mut self, fill: Fill) {
        match fill.side {
            FillSide::Buy => {
                self.inventory += fill.quantity;
                self.cash -= fill.price * fill.quantity;
            }
            FillSide::Sell => {
                self.inventory -= fill.quantity;
                self.cash += fill.price * fill.quantity;
            }
        }

        self.mark_to_market();
    }

    pub fn mark_to_market(&mut self) {
        self.pnl = self.cash + self.inventory * self.mid_price;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_pnl_update() {
        let mut state = SystemState::new(100.0);

        state.apply_fill(Fill::buy(100.0, 1.0));

        assert_eq!(state.inventory, 1.0);
        assert_eq!(state.cash, -100.0);
    }
}

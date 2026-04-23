use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub mid_price: f64,
    pub inventory: f64,
    pub cash: f64,
    pub pnl: f64,
}

impl SystemState {
    pub fn apply_trade(&mut self, price: f64, is_buy: bool) {
        if is_buy {
            self.inventory += 1.0;
            self.cash -= price;
        } else {
            self.inventory -= 1.0;
            self.cash += price;
        }

        self.pnl = self.cash + self.inventory * self.mid_price;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_pnl_update() {
        let mut state = SystemState {
            mid_price: 100.0,
            inventory: 0.0,
            cash: 0.0,
            pnl: 0.0,
        };

        state.apply_trade(100.0, true);

        assert_eq!(state.inventory, 1.0);
        assert_eq!(state.cash, -100.0);
    }
}

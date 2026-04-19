#[derive(Debug, Clone)]
pub struct SystemState {
    pub mid_price: f64,
    pub inventory: f64,
    pub cash: f64,
    pub pnl: f64,
}
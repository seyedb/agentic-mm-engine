use mm_engine::engine::metrics::SimulationMetrics;
use mm_engine::engine::simulation::{SimulationConfig, run_simulation};
use mm_engine::strategy::market_maker::StrategyParams;

fn main() {
    let config = SimulationConfig::default();
    let strategy = StrategyParams {
        spread: 0.5,
        skew_coeff: 0.05,
    };

    let result = run_simulation(config, &strategy);
    let final_step = result
        .final_step()
        .expect("simulation config should contain at least one step");

    println!(
        "final mid: {:.4}, pnl: {:.4}, inventory: {:.4}, cash: {:.4}",
        final_step.mid_price, final_step.pnl, final_step.inventory, final_step.cash
    );

    let metrics = SimulationMetrics::from_result(&result)
        .expect("simulation config should contain at least one step");

    println!(
        "fills: {} (buy: {}, sell: {}), traded qty: {:.4}, traded notional: {:.4}",
        metrics.total_fills,
        metrics.buy_fills,
        metrics.sell_fills,
        metrics.traded_quantity,
        metrics.traded_notional
    );
    println!(
        "risk: max inventory: {:.4}, avg abs inventory: {:.4}, max drawdown: {:.4}",
        metrics.max_abs_inventory, metrics.avg_abs_inventory, metrics.max_drawdown
    );
}

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
}

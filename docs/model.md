# Model

This project is an experimental simulator, not a production trading system.

## Market Process

- The mid-price follows a seeded random walk.
- Quotes are generated around the current mid-price.
- A noisy market price is sampled around the mid-price to determine fills.
- The simulator tracks a rolling estimate of recent mid-price volatility.

## Strategy

The current baseline strategy is an inventory-skewed market maker:

- A fixed spread sets the base distance between bid and ask.
- Positive inventory shifts both quotes lower.
- Negative inventory shifts both quotes higher.

This makes the strategy less eager to buy when inventory is high and more eager to buy when inventory is low.

Strategies receive a context object that includes estimated volatility. The current inventory-skew strategy does not use it yet, but this prepares the engine for volatility-aware quoting.

The project also includes a volatility-aware strategy. It widens the effective spread as estimated volatility rises:

```text
effective_spread = base_spread + volatility_coeff * estimated_volatility
```

## Fills

A quote is filled when the noisy market price crosses it:

- `market_price <= bid` creates a buy fill.
- `market_price >= ask` creates a sell fill.

Each fill includes:

- side
- price
- quantity
- fee

## Fees

Fees are charged as a fraction of traded notional:

```text
fee = price * quantity * fee_rate
```

Fees reduce cash on both buys and sells.

## Adverse Selection

After a fill, the mid-price moves slightly against the market maker:

- buy fill -> mid-price moves down
- sell fill -> mid-price moves up

This is a simple way to model fill toxicity: getting filled is not always a free spread capture.

## PnL

PnL is marked to market:

```text
pnl = cash + inventory * mid_price
```

# 🏝️ Team Theta Drip – IMC Prosperity 3 Write-up

> **Global Rank: 10**  
> **Algorithmic Rank: 12**  
> **Manual Rank: 19**  
> **India Rank: 3**

## 🌊 About the Challenge

**IMC Prosperity 3** was a 15-day algorithmic trading competition with over **12,000 teams** competing globally. Set in a fictional archipelago, each team represented an island aiming to accumulate wealth in **SeaShells**, the in-game currency. Participants were challenged to develop Python-based trading algorithms across evolving market scenarios — from simple spread products to options analogs — with new products and complexities introduced each round.

Alongside algorithmic submissions, each round included a **manual trading challenge**, requiring thinking and strategic judgment. Success demanded a blend of quantitative modeling, adversarial strategy design, and adaptability to shifting market dynamics (and of course a messed up sleep schedule).

## 📊 Our Results

| Round | Cumulative P&L (Total) | Manual P&L | Algo P&L | Round P&L | Overall Rank |
|:-----:|:----------------------:|:----------:|:--------:|:---------:|:------------:|
| 1     | 88,040                | 44,340     | 43,700   | 88,040    | 451          |
| 2     | 208,611               | 84,290     | 124,321  | 120,571   | 121          |
| 3     | 494,063               | 138,050    | 356,013  | 285,452   | 42           |
| 4     | 781,658               | 217,262    | 564,396  | 287,595   | 15           |
| 5     | 1,184,311             | 368,340    | 815,971  | 402,653   | 10           |

## Round 1️⃣

Round 1 introduced the first three products: **Rainforest Resin**, **Kelp**, and **Squid Ink**. Resin remained stable around a known historical price. Kelp moved up and down without a clear trend, while Squid Ink was highly volatile.

### 🌳 Rainforest Resin

We treated Resin as a static-value product centred at 10,000. The strategy combined tight market making with controlled position management. We placed passive bids and asks slightly inside inefficient book levels, while selectively taking favourable quotes when prices deviated from the fair value to capture mispricing. We used additional passive orders outside the spread to gently rebalance positions to minimise exposure.

### 🪸 Kelp

Kelp required a more adaptive strategy. We first plotted volume frequency distributions to identify the most reliable quoted sizes on each side of the book. These filtered quotes were then used to compute a more platform-aligned fair price. Around this price, we deployed a mix of passive market making and aggressive liquidity-taking if spreads widened. We also used conservative passive orders to manage position buildup.

### 🦑 Squid Ink

For Squid Ink, we implemented a directional signal based on top-of-book pressure. We computed a running pressure score by analysing changes in the best bid and ask levels and their volumes. Based on the signal (buy/sell/hold), we placed limit orders with self-adjusting price offsets that adapted based on recent fill volumes.

## Round 2️⃣

This round introduced two composite products — **Picnic Basket 1** and **Picnic Basket 2** — which bundled together **Croissants**, **Jams**, and **Djembes** in fixed proportions. These components were also made individually tradable.

### 🦑 Squid Ink

We had to rework our strategy after a negative P&L in Round 1, focusing on better signal generation and execution logic. We transitioned from a heuristic pressure-based strategy to a logistic regression model trained during runtime. The model used a 7-feature vector — including Z-score, price momentum, order book imbalance, MACD, and pressure signals — to generate probabilistic buy/sell decisions. Orders were sized and priced based on signal confidence, and the model was retrained every 50 timestamps using recent price history.

### 🍓 Jams

Jams exhibited mean-reverting behaviour with frequent short-term overextensions. We implemented a simple statistical strategy using the Z-score of mid-prices over a rolling window, with trend confirmation based on recent averages. Long positions were entered when Z-scores dropped sharply in an uptrend, while short positions were taken in overbought conditions with negative momentum. Position sizing was adjusted based on current exposure to control risk.

### 🥐 Croissants & 🪘 Djembes

We chose not to trade Croissants or Djembes individually. Both products were highly volatile, making them unreliable and poor candidates for market making or directional execution.

### 🧺 Picnic Basket 1

We ran a directional index arbitrage strategy on Basket 1 by computing its synthetic value in real time using the mid-prices of its components: `Synthetic Price = 6 × Croissants + 3 × Jams + 1 × Djembe`

We compared this synthetic price to the quoted market price of Basket 1. If the spread between them exceeded a pre-defined threshold, we placed directional trades on the basket — buying when undervalued and selling when overvalued. We deliberately avoided executing on the individual legs to reduce slippage and latency risks.

### 🧺 Picnic Basket 2

We initially applied the same arbitrage logic to Basket 2, using a synthetic price computed as: `Synthetic Price = 4 × Croissants + 2 × Jams`

However, this model showed high drawdowns in backtests due to inconsistent pricing and unstable correlation between Basket 2 and its components. The smaller size of the basket and tighter spreads amplified noise, and execution risk was significantly higher. Given these challenges and limited time to tune the model further, we dropped trading Basket 2 in this round.

## Round 3️⃣

In Round 3, a new product — **Volcanic Rock** — was introduced, along with five associated **Volcanic Rock Vouchers**, each acting as a European-style call option with an expiration of 7 in-game days. 

### 🦑 Squid Ink

Squid Ink again gave a negative P&L, and we decided to come up with an altogether different strategy. This mean reverting strategy using z-score based signals. When the current price deviated significantly from the EMA, we entered trades in the direction of expected mean reversion, limiting our position to 6 for limited exposure to the market.

### 🧺 Picnic Basket 1 & 2

We made some minor changes in the last round strategy by using volume-weighted mids for better estimation of mid prices, and also tuned the thresholds for better performance.

### 🪨 Volcanic Rock

We ran a dynamic market-making strategy that adjusted quoting aggressiveness based on both position and recent execution pressure.
Spread placement and size were adjusted using a position-sensitive logic, quoting tighter when flat and backing off when near limits.

### 🎟️ Volcanic Rock Vouchers

We selected two vouchers (`10250` and `10000` strike) for active trading based on their moneyness and relative liquidity. For each, we computed a theoretical value using the Black-Scholes model, taking into account:
- Current Volcanic Rock mid-price (spot),
- Strike price,
- Time to expiry (adjusted daily),
- Estimated volatility (inferred from Rock’s price history).

We then back-calculated the implied volatility from the voucher’s market price and compared it against a rolling average of historical implied vols. If the deviation exceeded a threshold, we traded directionally — buying underpriced volatility and selling overpriced premium.

## Round 4️⃣

The luxury product **Magnificent Macarons** was introduced. Macarons could only be traded via **conversion requests** through Pristine Cuisine, subject to transport fees, tariffs, and a per-timestamp storage cost on long positions. The product's pricing was influenced by multiple external signals like sunlight, sugar price, and tariffs.

### 🦑 Squid Ink

Despite multiple rounds of tuning and complete rewrites of the logic, Squid Ink continued to underperform, once again closing the round with negative P&L. After significant sunk time across three rounds, we decided not to trade the product and allocate our engineering effort to other products.

### 🧺 Picnic Basket 1

We made targeted improvements to our Basket 1 strategy this round. After analysing noise in our spread signals, we made three key changes:
1. Instead of using the raw synthetic price, we applied an EMA to smooth it and reduce short-term noise.
2. We added an exit threshold to our Z-score logic, allowing us to gradually exit when the spread converged to the mean, rather than waiting for a full reversal.
3. We improved order placement by splitting orders across multiple price levels. Analysis showed that baskets often got filled near the optimal level, so we placed layered orders with varying sizes based on confidence in fill probability.

These changes led to more stable performance and improved fill consistency.

### 🧺 Picnic Basket 2

We extended the same improvements to Basket 2. EMA smoothing, Z-score thresholds (entry + exit), and layered fills were applied similarly. The thresholds were tuned using our own backtester and some other optimisations.

> *(We plan to release cleaned-up versions of our Jupyter notebooks used in tuning once we have time to format and comment them.)*

### 🪨 Volcanic Rock & 🎟️ Vouchers

After Round 3, we did a deeper analysis on Rock and the `9500` and `9750` vouchers, which we hadn’t actively traded before.

- We confirmed that Volcanic Rock exhibited mean-reverting behaviour, so we implemented a Z-score-based mean reversion strategy with conservative sizing and exit logic.
- For `9500` and `9750` vouchers, thanks to the hint released about the volatility smile, we realised that they were (and would remain) deeply ITM and strongly co-moved with Rock. We applied the same mean-reverting model used for Rock, with threshold tuning to adjust for the different option premiums.
- `10000` voucher remained ATM, so we continued with the implied volatility Z-score arbitrage strategy from Round 3.
- `10250` voucher had moved slightly OTM, but we anticipated it would hover around the ATM region. So we reused the `10000` strategy, betting on premium fluctuations and short-term mispricings.

This combined approach allowed us to trade both volatility and direction depending on each voucher’s moneyness and market structure.

### 🍬 Magnificent Macarons

We explored several strategies for Macarons, including regression-based fair value estimation using external inputs (sunlight and sugar prices). However, due to:
- Complex and unstable input dependencies,
- Strict conversion limits (10 units),
- Continuous storage cost on long positions,  
We could not design a strategy we trusted under simulation.

With end-semester exams approaching for all of us, we chose to skip trading Macarons and focus on improving proven models instead.

## Round 5️⃣

No new products were introduced in the final round, but a key feature was unlocked: the exchange began exposing the **counterparty identity** for each executed trade. This enabled detailed adversarial analysis and reactive strategies based on flow from known participants.

By this point, we were **ranked 15 globally**, so our goal was to preserve our standing by avoiding any negative P&L, especially from Squid Ink and Macarons, which we had previously skipped. With limited time and increasing risk, we shifted our focus toward signal-driven trading, prioritising safe and high-confidence setups.

### 🎟️ Volcanic Rock Vouchers

We refined our voucher selection based on updated volatility analysis:

- `10250` voucher showed persistent signs of staying deeply OTM, so we fully removed it from our portfolio.
- `10000` voucher appeared to be drifting towards OTM as well, but our analysis suggested a high-confidence profit window of 15–20k Seashells as it approached the ATM region. We kept it active using the same IV-based Z-score strategy from earlier rounds.
- For `9500` and `9750`, we continued using the mean-reversion strategy synced with the Volcanic Rock movement.

### 🍬 Magnificent Macarons

We chose to include a very conservative arbitrage strategy in this round.
We only executed a single-leg conversion strategy:
- When the local market bid exceeded Pristine Cuisine's adjusted ask (including transport + import fees), we sold Macarons locally and converted them from Pristine Cuisine.
- This trade had zero directional exposure and was triggered only when the arbitrage edge was obvious.

We disabled the reverse leg (buy local, sell to Pristine) due to high export tariffs, which rarely made the flow profitable. This strategy acted as a safe, low-frequency alpha component that aligned with our risk-averse approach in this round.

### 🧠 Signal-Based Trading (Using Bot Names)

After implementing our core voucher and Macaron changes, we shifted focus to exploring the newly introduced bot identities. We performed an exhaustive analysis of all bots, plotting the trades each bot made across different products and analysing bot-pair interactions to identify patterns in flow and reaction.

Shoutout to Aniruddhh and Shresth for leading this research, helping us uncover that Olivia’s trades showed strong, consistent short-term predictiveness, especially on:
- **Squid Ink**
- **Croissants**
- **Kelp**

The signal was clear, and much like in Prosperity 2, we realised that many teams likely picked up on it. To gain an edge, we knew that execution quality had to be the differentiator.

We initially spent time experimenting with dynamic level selection using fill probability modelling. While that approach didn’t improve profits much, it helped us discover that many assets consistently filled at multiple levels near the top of the book.

So, when a signal was detected from Olivia’s trade:
- We went all-in on the predicted direction, placing layered limit orders across multiple levels instead of just hitting the best bid/ask.
- We distributed volume across levels proportional to confidence and historical fill behaviour.

This aggressive and breadth-aware execution strategy significantly boosted our realised profits, turning a widely known signal into a differentiated alpha source through tactical precision.

## 🙌 Acknowledgments

We'd like to thank **IMC Trading** for organising this event and fostering a global community around algorithmic trading. Also, shoutout to the creators of open-source writeups from **Prosperity 1 and 2** — your insights helped shape ours.

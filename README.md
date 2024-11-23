# Algo Trading Strategy leveraging ARIMA-X

<h2> Introduction:</h2>

> I wanted to build an algo trading strategy to beat the SPY Returns and CAGR.
>
> However, the challenge that I set my self was to only trade SPY - essentially finding a way to time my buy and sell signals to beat the market.
> 
> This absolutely did not work out flawlessly. I had built 6 different strategies that massively underperformed before I found the one that worked.
> 
> My goal here is not to glorify Quantitative Trading and pretend like every strategy built - works.
> 
> More often than not, they fail.
>
> Here's a snapshot of some of my failed strategies. I try to use Mean Reversion using Gold Futures (Red line) and a simple MACD strategy (Blue line) and compare that against Cumulative Returns of the SPY for the timeframe listed on the axes.

<img width="827" alt="Screenshot 2024-11-20 at 1 12 14 PM" src="https://github.com/user-attachments/assets/45a3ab74-6f0d-48fc-9d4e-57243098a10e">


<h2>Rationale:</h2>

> I wanted to leverage the inverse relationship between commodities and equities, hoping to build signals that primariy look at the Adjusted Closing prices of commodities and then, based on the market movement there, buy/sell SPY.

> At first, I tried gold - did not work. This was because Gold is not nearly volatile enough to be able to build any strategy on it. The same goes with Gold Futures, Silver and most other commodities.

> However, after a bit of research, I found out that the Energy Futures (yfinance ticker - 'ES=F') are pretty well inversely correlated and also extremely volatile.

> I also used the VIX Index (yfinance ticker - ^VIX) as it captures a lot of downwards volatility that SPY does not seem to capture - effectively predicitng market downturns before it significantly hampers SPY.

<h2> Methodology: </h2>

> I build an ARIMAX, Machine Learning model that uses ^VIX and ES=F as the independent features. I also use lagged values of ^VIX to help train my model.

> The ARIMAX captures the trend of the SPY, which i then overlay onto the SPY Actual Cumulative Returns and use that relationship to build Buy/Sell Signals.

<h3> ARIMAX Predicted Mean (Trend - red line) vs SPY Cumulative Returns (Blue - dashed line): </h3>
<img width="1004" alt="Screenshot 2024-11-23 at 10 53 12 AM" src="https://github.com/user-attachments/assets/835a486e-7c0a-47a2-921b-c615f5c7e6ba">

> After simulating the strategy with a starting base capital of $100,000, we get the following results:
<h3> Performance Metrics: </h3>

<img width="341" alt="Screenshot 2024-11-23 at 10 59 45 AM" src="https://github.com/user-attachments/assets/29bcfc97-3b93-405b-8dfc-17f3f63c214e">

<h3> SPY Portfolio Value vs Strategy Portfolio Value: </h3>
<img width="633" alt="Screenshot 2024-11-23 at 10 59 59 AM" src="https://github.com/user-attachments/assets/9770f602-304d-4a6a-a303-81605aa3bb32">


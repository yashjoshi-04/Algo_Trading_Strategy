# Algo Trading Strategy leveraging ARIMAX

## Introduction:

> I wanted to build an algo trading strategy to beat the SPY Returns and CAGR.
>
> However, the challenge that I set my self was to only trade SPY - essentially finding a way to time my buy and sell signals to beat the market.
> 
> This absolutely did not work out flawlessly. I had built 6 different strategies that massively underperformed before I found the one that worked.
> 
> My goal here is not to glorify Quantitative Trading and pretend like every strategy built - works.
> 
> More often than not, they fail.


## Rationale:

> I wanted to leverage the inverse relationship between commodities and equities, hoping to build signals that primariy look at the Adjusted Closing prices of commodities and then, based on the market movement there, buy/sell SPY.

>> At first, I tried gold - did not work. This was because Gold is not nearly volatile enough to be able to build any strategy on it. The same goes with Gold Futures, Silver and most other commodities.

>>> However, after a bit of research, I found out that the Energy Futures (yfinance ticker - 'ES=F') are pretty well inversely correlated and also extremely volatile.

>>> I also used the VIX Index (yfinance ticker - ^VIX) as it captures a lot of downwards volatility that SPY does not seem to capture - effectively predicitng market downturns before it significantly hampers SPY.

## Methodology:

> I build an ARIMAX, Machine Learning model that uses ^VIX and ES=F as the independent features. I also use lagged values of ^VIX to help train my model.

>> The ARIMAX captures the trend of the SPY, which i then overlay onto the SPY Actual Cumulative Returns and use that relationship to build Buy/Sell Signals.

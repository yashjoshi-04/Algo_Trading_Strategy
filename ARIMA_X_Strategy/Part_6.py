
# Backtesting the trading strategy, comparing it to our SPY Baseline and calculating performance metrics

import pandas as pd
import numpy as np
import plotly.graph_objects as go

df = pd.DataFrame({"Cumulative Log Returns": actual_cum_returns, "Buy Signal": Buy_Signal, "Unwind Signal": Unwind_Signal})

def simulate_trading_strategy(df, initial_capital=100_000, commission=5):
    """
    Simulates trading strategy based on Buy and Unwind signals
    """
    df['Daily Log Returns'] = df['Cumulative Log Returns'].diff() 
    df['Daily Simple Returns'] = np.exp(df['Daily Log Returns']) - 1  
    df['Cumulative Simple Returns'] = np.exp(df['Cumulative Log Returns']) - 1
    df['Portfolio Value'] = initial_capital
    df['Position'] = 0 
    portfolio_value = initial_capital
    in_position = False
    
    for i in range(len(df)):
        if df['Buy Signal'].iloc[i] and not in_position:
            portfolio_value -= commission
            in_position = True
            df.at[df.index[i], 'Position'] = 1
            
        elif df['Unwind Signal'].iloc[i] and in_position:
            if i > 0: 
                portfolio_value = portfolio_value * (1 + df['Daily Simple Returns'].iloc[i]) - commission
            in_position = False
            df.at[df.index[i], 'Position'] = 0
            
        elif in_position and i > 0:
            portfolio_value = portfolio_value * (1 + df['Daily Simple Returns'].iloc[i])
            
        df.at[df.index[i], 'Portfolio Value'] = portfolio_value
    
    df['Strategy Returns'] = df['Portfolio Value'].pct_change()
    spy_final_value = portfolio['Portfolio_Value'].iloc[-1]
    metrics = calculate_metrics(df, initial_capital, spy_final_value)  
    return df, metrics

def calculate_metrics(df, initial_capital, spy_final_value):
    """
    Calculate key performance metrics
    """

    years = (df.index[-1] - df.index[0]).days / 365.25
    final_value = df['Portfolio Value'].iloc[-1]
    strategy_return = (final_value - initial_capital) / initial_capital
    strategy_cagr = (final_value / initial_capital) ** (1/years) - 1
    spy_cagr = (spy_final_value / initial_capital) ** (1/years) - 1
    rf_rate = 0.13
    strategy_returns = df['Strategy Returns'].dropna()
    excess_returns = strategy_returns - rf_rate/252  # Daily risk-free rate
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(strategy_returns)

    spy_return = (spy_final_value - initial_capital) / initial_capital
    outperformance = (strategy_return - spy_return) * 100
    rolling_max = df['Portfolio Value'].cummax()
    drawdowns = (df['Portfolio Value'] - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'Final Portfolio Value': final_value,
        'Total Return': strategy_return,
        'CAGR': strategy_cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'SPY CAGR': spy_cagr,
        'Outperformance vs SPY': outperformance
    }

def plot_results(df):
    """
    Plot strategy performance vs SPY
    """
    fig = go.Figure()
    
    # SPY baseline portfolio value trace:
    
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Portfolio_Value'], mode='lines', name='Portfolio Value', line=dict(color='blue')))
    
    # Strategy portfolio value:
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Portfolio Value'], mode='lines', name='Strategy Performance', line=dict(color='green')))    
    return fig

df, metrics = simulate_trading_strategy(df)

print("\nPerformance Metrics:")
print("-" * 50)
for metric, value in metrics.items():
    if metric in ['CAGR', 'Total Return', 'Out-Performance vs SPY', 'SPY CAGR']:
        print(f"{metric}: {value*100:.2f}%")
    elif metric == 'Final Portfolio Value':
        print(f"{metric}: ${value:,.2f}")
    else:
        print(f"{metric}: {value:.3f}")


fig = plot_results(df)
fig.show()

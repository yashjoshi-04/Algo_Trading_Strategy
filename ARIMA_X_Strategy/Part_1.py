# Building the Baseline SPY trading strategy as a comparison for our trading strategy.
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd

initial_capital = 100000
commission_fee = 5
start_date = dt.datetime(2020, 11, 1)
end_date = dt.datetime(2024, 10, 31)

spy = yf.download('SPY', start=start_date, end=end_date)

portfolio = pd.DataFrame(index=spy.index)
portfolio['SPY_Price'] = spy['Adj Close']
initial_price = portfolio['SPY_Price'].iloc[0]
shares = (initial_capital - commission_fee) // initial_price
cash_left = initial_capital - (shares * initial_price) - commission_fee
portfolio['Portfolio_Value'] = shares * portfolio['SPY_Price'] + cash_left
portfolio['Daily_Returns'] = portfolio['Portfolio_Value'].pct_change()
years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
final_value = portfolio['Portfolio_Value'].iloc[-1]
cagr = (final_value / initial_capital) ** (1/years) - 1
total_return = (final_value - initial_capital) / initial_capital
rf_rate = 0.13 
excess_returns = portfolio['Daily_Returns'].dropna() - rf_rate/252
sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(portfolio['Daily_Returns'].dropna())
rolling_max = portfolio['Portfolio_Value'].cummax()
drawdowns = (portfolio['Portfolio_Value'] - rolling_max) / rolling_max
max_drawdown = drawdowns.min()

print("\nPerformance Metrics:")
print("-" * 50)
print(f"Final Portfolio Value: ${final_value:,.2f}")
print(f"Total Return: {total_return*100:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Portfolio_Value'], mode='lines', name='Portfolio Value', line=dict(color='blue')))
fig.update_layout(title='Long-Only SPY Portfolio Value with Metrics',xaxis_title='Date',yaxis_title='Portfolio Value ($)',template='plotly_white',width=1200,height=600,showlegend=True,yaxis=dict(tickformat='$,.0f'))
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.show()
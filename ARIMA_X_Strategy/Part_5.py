# Comparing Actual Cumulative Returns to Predicted Cumulative Returns

import numpy as np
import plotly.graph_objects as go
import pandas as pd


Predicted_Tuned.index = Y_test.index

actual_cum_returns = np.exp(Y_test).cumprod()
predicted_cum_returns = np.exp(Predicted_Tuned).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(x=actual_cum_returns.index, y=actual_cum_returns, mode='lines', name='Actual Y - SPY cum returns', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=predicted_cum_returns.index, y=predicted_cum_returns, mode='lines', name='SPY predicted returns', line=dict(color='red')))
fig.update_layout(title="Cumulative Returns: Actual vs Predicted", xaxis_title="Date", yaxis_title="Cumulative Returns", template='plotly_white', width=1200, height=800, legend=dict(x=0, y=1.0))
fig.show()



# Building the Buy and Unwind Signals, using a simple Mean-Reversion Strategy

# Buy Signal

n = 0.8
Buy_Signal = actual_cum_returns < (predicted_cum_returns - n * predicted_cum_returns.std())
Buy_Signal.value_counts()

# We buy at the very beginning of the prediction period as well.

Buy_Signal.iloc[0] = True 
Buy_Signal

# Unwind Signal

n = 1.5
N = 1.5
Unwind_Signal = (actual_cum_returns > (predicted_cum_returns + N * predicted_cum_returns.std())) | \
                (actual_cum_returns < (predicted_cum_returns - n * predicted_cum_returns.std()))
print(Unwind_Signal.value_counts())


Predicted_Tuned.index = Y_test.index

actual_cum_returns = np.exp(Y_test).cumprod()
predicted_cum_returns = np.exp(Predicted_Tuned).cumprod()

fig = go.Figure()

# (SPY cumulative returns) trace:
fig.add_trace(go.Scatter(x=actual_cum_returns.index, y=actual_cum_returns, mode='lines', name='Actual Y - SPY Cumulative Returns', line=dict(color='blue', dash='dash')))

# (SPY predicted cumulative returns) trace
fig.add_trace(go.Scatter(x=predicted_cum_returns.index, y=predicted_cum_returns, mode='lines', name='SPY Predicted Cumulative Returns', line=dict(color='red')))

# Buy Signal
fig.add_trace(go.Scatter(x=actual_cum_returns[Buy_Signal].index, y=actual_cum_returns[Buy_Signal], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))

# Unwind Signal
fig.add_trace(go.Scatter(x=actual_cum_returns[Unwind_Signal].index, y=actual_cum_returns[Unwind_Signal], mode='markers', name='Unwind Signal', marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title="Cumulative Returns: Actual vs Predicted", xaxis_title="Date", yaxis_title="Cumulative Returns", template='plotly_white', width=1200, height=800, legend=dict(x=0, y=1.0))  
fig.show()





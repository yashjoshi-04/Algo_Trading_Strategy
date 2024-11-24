# Building the Baseline SPY trading strategy as a comparison for our trading strategy.
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib as plt

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


# --------------------------------------------------------------- #

# Importing and Wrangling data

stock_ticker = ['SPY']
index_ticker = ['^VIX', 'SPY']
bond_ticker = ['ES=F']

start = dt.datetime(2005, 1, 1)
end = dt.datetime(2024, 10, 31)

stock_data = yf.download(stock_ticker, start = start, end = end)
index_data = yf.download(index_ticker, start = start, end = end)
bond_data = yf.download(bond_ticker, start = start, end = end)

# Trying to compute weekly returns

return_period = 5

# Outcome Variable (Y):

Y = (np.log(index_data.loc[ : , ("Adj Close", "SPY")]).diff(return_period).shift(-return_period))
Y.name = (Y.name[-1] + "_pred")

# Independent Variables (X):

X1 = (np.log(index_data.loc[ : , ("Adj Close", ("SPY", "^VIX"))]).diff(return_period))
X1.columns = X1.columns.droplevel()

X2 = (np.log(bond_data.loc[ : , ("Adj Close", "ES=F")]).diff(return_period))

if isinstance(X2, pd.Series):
    X2 = X2.to_frame()
    X2.columns = ["ESF"]

X3 = (pd.concat([np.log(index_data.loc[ : , ("Adj Close", "^VIX")]).diff(i) for i in [return_period, return_period * 3, return_period * 6, return_period * 12]], axis = 1).dropna())
X3.columns = ["VIX_DT", "VIX_3DT", "VIX_6DT", "VIX_12DT"]

# Combining all the independent variables

X = pd.concat([X1, X2, X3], axis = 1).dropna()

# Combining the dependent and independent variables

data = pd.concat([Y, X], axis = 1).dropna().iloc[ : : return_period, :]

Y = data.loc[ : , Y.name]
X = data.loc[ : , X.columns]


# --------------------------------------------------------------- #

# Exploratory Data Analysis

data.hist(bins = 35, sharex = False, sharey = False, figsize =[16, 16])
plt.show()

data.plot(kind = "density", subplots = True, layout = (4, 4), sharex = False, legend = False, figsize = [16, 16])
plt.show()

correlation = data.corr()
plt.figure(figsize =[16, 16])
plt.title("Correlation Matrix")

sns.heatmap(correlation, vmax = 1, square = True, cmap = "viridis", annot = True)


# Running Ten-Fold Cross Validation, some Evaluation Metrics and initializing Training and Testing periods.

# cross-validation and assessment
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc

# feature Selection
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Custom Train-Test Split

validation_size = 0.20
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = (X[0:train_size], X[train_size:len(X)])
Y_train, Y_test = (Y[0:train_size], Y[train_size:len(X)])

# Initializing Models

models = []

models.append(("LR", LinearRegression()))
models.append(("LASSO", Lasso()))
models.append(("EN", ElasticNet()))
models.append(("CART", DecisionTreeRegressor()))
models.append(("KNN", KNeighborsRegressor()))
models.append(("SVR", SVR()))
models.append(("RFR", RandomForestRegressor()))
models.append(("ETR", ExtraTreesRegressor()))
models.append(("GBR", GradientBoostingRegressor()))
models.append(("ABR", AdaBoostRegressor()))

num_folds = 10
seed = 241001
scoring = "neg_mean_squared_error"

names = []
kfold_results = []
train_results = []
test_results = []

for name, model in models:

    names.append(name)
    kfold = KFold(n_splits = num_folds, random_state = seed, shuffle = True)
    cv_results = -1 * cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    kfold_results.append(cv_results)
    res = model.fit(X_train, Y_train) 

# Evaluating Model on Training Set:

    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)

# Evaluating Model on Testing Set:          

    test_result = mean_squared_error(res.predict(X_test), Y_test)
    test_results.append(test_result)
    
# Printing the Results:

    message = "%s: %f (%f) %f %f" % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
    print(message)

# Comparing the algorithms using the results of the K-Fold Cross Validation

fig = plt.figure(figsize = [16, 8])
fig.suptitle("Algorithm Comparison: Results of K-Fold Cross Validation")
ax = fig.add_subplot(111)
plt.boxplot(kfold_results)
ax.set_xticklabels(names)
plt.show()



# --------------------------------------------------------------- #

# Comparing Algorithms and Building the ARIMA - X Model

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib as plt

X_train_ARIMA = X_train.loc[:, ['^VIX', 'ESF', 'VIX_DT', 'VIX_3DT', 'VIX_6DT', 'VIX_12DT']]
X_test_ARIMA = X_test.loc[:, ['^VIX', 'ESF', 'VIX_DT', 'VIX_3DT', 'VIX_6DT', 'VIX_12DT']]

train_len = len(X_train_ARIMA)
test_len = len(X_test_ARIMA)
total_len = len(X)

modelARIMA = stats.ARIMA(endog = Y_train, exog = X_train_ARIMA, order = [1, 0, 0])
model_fit = modelARIMA.fit()

error_training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
predicted = model_fit.predict(start = train_len - 1, end = total_len - 1, exog = X_test_ARIMA)[1: ]
error_testing_ARIMA = mean_squared_error(Y_test, predicted)

test_results.append(error_testing_ARIMA)
train_results.append(error_training_ARIMA)
names.append("ARIMA")

# Comparing the performance of multiple Algorithms
fig = go.Figure()
fig.add_trace(go.Bar(x=names, y=train_results, name='Errors in Training Set', marker=dict(color='red'), offsetgroup=0))
fig.add_trace(go.Bar(x=names, y=test_results, name='Errors in Testing Set', marker=dict(color='blue'), offsetgroup=1))
fig.update_layout(
    title='Comparing the Performance of Various Algorithms on the Training vs. Testing Data',
    xaxis=dict(title='Models'),
    yaxis=dict(title='Mean Squared Error (MSE)'),
    barmode='group',
    legend=dict(title='Error Type'))
fig.show()

# Hyperparameter Tuning

def assess_ARIMA_model(arima_order):
    
    modelARIMA = stats.ARIMA(endog = Y_train, exog = X_train_ARIMA, order = arima_order)  
    model_fit = modelARIMA.fit()
    error = mean_squared_error(Y_train, model_fit.fittedvalues)
    return error

def assess_models(p_values, d_values, q_values):
    
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = assess_ARIMA_model(order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print("ARIMA%s MSE = %.7f" % (order, mse))
                    
                except:
                    continue
    print("Best ARIMA%s MSE = %.7f" % (best_cfg, best_score))

p_values = [0, 1, 2]
d_values = range(0, 2)
q_values = range(0, 2)

# Picks the best ARIMA model order:

assess_models(p_values, d_values, q_values)

ARIMA_Tuned = stats.ARIMA(endog = Y_train, exog = X_train_ARIMA, order = [1, 0, 1]) # Set the best ARIMA model order
ARIMA_Fit_Tuned = ARIMA_Tuned.fit()

Predicted_Tuned = model_fit.predict(start = train_len - 1, end = total_len - 1, exog = X_test_ARIMA)[1: ]
print(mean_squared_error(Y_test, Predicted_Tuned))




# --------------------------------------------------------------- #

# Comparing Actual Cumulative Returns to Predicted Cumulative Returns

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




# --------------------------------------------------------------- #

# Back-testing the trading strategy, comparing it to our SPY Baseline and calculating performance metrics

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








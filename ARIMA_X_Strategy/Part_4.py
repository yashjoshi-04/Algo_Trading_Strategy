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
    print("Best ARIMA%s MSE = %.7f" % (best_cfg, best_score)
          )

p_values = [0, 1, 2]
d_values = range(0, 2)
q_values = range(0, 2)

# Picks the best ARIMA model order:

assess_models(p_values, d_values, q_values)

ARIMA_Tuned = stats.ARIMA(endog = Y_train, exog = X_train_ARIMA, order = [1, 0, 1]) # Set the best ARIMA model order
ARIMA_Fit_Tuned = ARIMA_Tuned.fit()

Predicted_Tuned = model_fit.predict(start = train_len - 1, end = total_len - 1, exog = X_test_ARIMA)[1: ]
print(mean_squared_error(Y_test, Predicted_Tuned))

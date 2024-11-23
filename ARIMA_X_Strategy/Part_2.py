# Importing and Wrangling data

import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


stock_ticker = ['SPY']
index_ticker = ['^VIX', 'SPY']
bond_ticker = ['ES=F']

start = dt.datetime(2005, 1, 1)
end = dt.datetime(2024, 10, 31)

stock_data = yf.download(stock_ticker, start = start, end = end)
index_data = yf.download(index_ticker, start = start, end = end)
bond_data = yf.download(bond_ticker, start = start, end = end)



# Compute weekly returns

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



# Exploratory Data Analysis

data.hist(bins = 35, sharex = False, sharey = False, figsize =[16, 16])
plt.show()

data.plot(kind = "density", subplots = True, layout = (4, 4), sharex = False, legend = False, figsize = [16, 16])
plt.show()

correlation = data.corr()
plt.figure(figsize =[16, 16])
plt.title("Correlation Matrix")

sns.heatmap(correlation, vmax = 1, square = True, cmap = "viridis", annot = True)



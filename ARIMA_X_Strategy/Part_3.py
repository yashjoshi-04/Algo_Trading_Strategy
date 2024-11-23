# Running Ten-Fold Cross Validation, some Evaluation Metrics and initializing Training and Testing periods.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib as plt


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
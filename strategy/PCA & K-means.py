from pack import *
from Jquant import *

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime as dt, timedelta as td
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

start_date = dt(2005, 1, 1)
end_date = dt(2020, 5, 9)
df = create_lagged_series('spy', start_date, end_date)
df.dropna(inplace=True)


# Use the prior n-days of returns as predictor 
# values, with direction as the response
X = df[['Lag1', 'Lag2', 'Lag5', 'Lag10', 'Lag20', 'Lag50']]
y = df['Direction']

# The test data is split into two parts
start_test = dt(2016, 1, 1)

# Create training and test sets
X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

X_train = X_train.values
X_test = X_test.values
pca = PCA(n_components=3, whiten=True)
X_transformed = pca.fit_transform(X_train)

X_train = X_transformed

model = KMeans(n_clusters=3)

model.fit(X_train)

X_test = pca.fit_transform(X_test)
pred = pd.DataFrame(model.predict(X_test))
pred.index = y_test.index

temp = pd.DataFrame(df.loc[y_test.index]['Today'])
K = pred == 0
K.columns = temp.columns
L = pred == 1
L.columns = temp.columns
M = pred == 2
M.columns = temp.columns


temp[K].fillna(0)
ReturnCumulative(temp[K])
temp[L].fillna(0)
ReturnCumulative(temp[L])
temp[M].fillna(0)
ReturnCumulative(temp[M])

# Select Cum ret > 0
inv = K | M
ReturnCumulative(temp[inv])

PerformanceAnalysis(temp[inv])
plot_annual_returns(temp[inv])
plot_monthly_returns_heatmap(temp[inv])
ReturnStats(temp[inv])

spy = fdr.DataReader('spy', start_test, end_date)
spy = spy['Close'].pct_change(1)
spy = spy.fillna(0)
spy = pd.DataFrame(spy)
ReturnCumulative(spy)
##############################################################################
PerformanceAnalysis(temp[inv], spy)

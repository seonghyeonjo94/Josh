# https://www.pythonforfinance.net/2019/07/22/time-series-decomposition-prediction-in-python/

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt
import seaborn as sns

# Set figure width to 12 and height to 9
plt.rcParams['figure.figsize'] = [12, 9]

# Load data and Data preprocessing
df = pd.read_csv(r'C:\Users\USER\Downloads\EURUSD.csv',sep='\t', index_col='Date')
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df = df.resample('W').last()
series = df['Price']

# First the HP filter decomposition
cycle, trend = sm.tsa.filters.hpfilter(series, 50)
fig, ax = plt.subplots(3,1)
ax[0].plot(series)
ax[0].set_title('Price')
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[2].plot(cycle)
ax[2].set_title('Cycle')
plt.show()

# And next the SYL decomposition
result = STL(series).fit()
chart = result.plot()
plt.show()

### Forecast model - persistence model
# that works by assigning the last observed value as the prediction for the following value
predictions = series.shift(1).dropna()
test_score = np.sqrt(mean_squared_error(series[int(len(series) * 0.7)+1:], predictions.iloc[int(len(series) * 0.7):]))
print('Test RMSE: %.5f' % test_score)
plt.plot(series.iloc[-25:], label='Price')
plt.plot(predictions[-25:], color='red', label='Prediction')
plt.legend()
plt.show()

# create a scatter plot
fig, ax = plt.subplots()
ax = sns.regplot(series.iloc[-int(len(series) * 0.3):].pct_change(),
            predictions.iloc[-int(len(series) * 0.3):].pct_change(), )
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('EURUSD Observed vs Predicted Values')
ax.grid(True, which='both')
ax.axhline(y=0, color='#888888')
ax.axvline(x=0, color='#888888')
sns.despine(ax=ax, offset=0)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()
mae = round(abs(series.iloc[-int(len(series) * 0.3):].pct_change() - predictions.iloc[-int(len(series) * 0.3):].pct_change()).mean(),4)
print(f'The MAE is {mae}')

# calculate the Hit rate
price_pred = pd.concat([series.iloc[-int(len(series) * 0.3):].pct_change(), predictions.iloc[-int(len(series) * 0.3):].pct_change()], axis=1)
price_pred.dropna(inplace=True)
price_pred.columns = ['Price', 'preds']
price_pred['hit'] = np.where(np.sign(price_pred['Price']) == np.sign(price_pred['preds']), 1, 0)
print(f"Hit rate: {round((price_pred['hit'].sum() / price_pred['hit'].count()) * 100,2)}%")

### Forecast model - Autoregressive model
historic = series.iloc[:int(len(series) * 0.7)].to_list()
test = series.iloc[int(len(series) * 0.7):]
predictions = []
for i in range(len(test)):
    model = AR(historic)
    model_fit = model.fit()
    pred = model_fit.predict(start=len(historic), end=len(historic), dynamic=False)
    predictions.append(pred[0])
    historic.append(test[i])
    
predictions = pd.Series(predictions, index=test.index)
    
test_score = np.sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.5f' % test_score)

# plot results
plt.plot(test.iloc[-25:], label='Prices')
plt.plot(predictions.iloc[-25:], color='red', label='Prediction')
plt.legend()
plt.show()

# plot a scatter chart
fig, ax = plt.subplots()
ax = sns.regplot(series.iloc[-int(len(series) * 0.3):].pct_change(),
            predictions.iloc[-int(len(series) * 0.3):].pct_change(), )
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('EURUSD Observed vs Predicted Values')
ax.grid(True, which='both')
ax.axhline(y=0, color='#888888')
ax.axvline(x=0, color='#888888')
sns.despine(ax=ax, offset=0)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()
mae = round(abs(test.pct_change() - predictions.pct_change()).mean(),10)
print(f'The MAE is {mae}')

# calculate the Hit rate
price_pred = pd.concat([test.pct_change(), predictions.pct_change()], axis=1)
price_pred.dropna(inplace=True)
price_pred.columns = ['Price', 'preds']
price_pred['hit'] = np.where(np.sign(price_pred['Price']) == np.sign(price_pred['preds']), 1, 0)
print(f"Hit rate: {round((price_pred['hit'].sum() / price_pred['hit'].count()) * 100,2)}%")

### Time Series Decomposition for prediction
# decompose the EURUSD series using the Hodrick-Prescott Filter and store each component
cycle, trend = sm.tsa.filters.hpfilter(series, 50)
fig, ax = plt.subplots(3,1)
ax[0].plot(series)
ax[0].set_title('Price')
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[2].plot(cycle)
ax[2].set_title('Cycle')
plt.show()

# run AR model with decomposed series as input
component_dict = {'cycle': cycle, 'trend': trend}
prediction_results = []
for component in ['trend', 'cycle']:
    historic = component_dict[component].iloc[:int(len(series) * 0.7)].to_list()
    test = component_dict[component].iloc[int(len(series) * 0.7):]
    predictions = []
    for i in range(len(test)):
        model = AR(historic)
        model_fit = model.fit()
        pred = model_fit.predict(start=len(historic), end=len(historic), dynamic=False)
        predictions.append(pred[0])
        historic.append(test[i])
    predictions = pd.Series(predictions, index=test.index, name=component)
    prediction_results.append(predictions)
    test_score = np.sqrt(mean_squared_error(test, predictions))
    print(f'Test for {component} MSE: {test_score}')
    # plot results
    plt.plot(test.iloc[:], label='Observed '+component)
    plt.plot(predictions.iloc[:], color='red', label='Predicted '+component)
    plt.legend()
    plt.show()
    
# recompose the data in an additive manner and calculate the RMSE
recomposed_preds = pd.concat(prediction_results,axis=1).sum(axis=1)
recomposed_preds.name = 'recomposed_preds'
plt.plot(series.iloc[int(len(series) * 0.7):], label='Observed')
plt.plot(recomposed_preds, color='red', label='Predicted')
plt.legend()
plt.show()
test_score = np.sqrt(mean_squared_error(series.iloc[int(len(series) * 0.7):], recomposed_preds))
print(f'RMSE: {test_score}')

# plot a scatter chart and calculate the MAE
fig, ax = plt.subplots()
ax = sns.regplot(series.iloc[-int(len(series) * 0.3):].pct_change(),
            recomposed_preds.iloc[-int(len(series) * 0.3):].pct_change())
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('EURUSD Observed vs Predicted Values')
ax.grid(True, which='both')
ax.axhline(y=0, color='#888888')
ax.axvline(x=0, color='#888888')
sns.despine(ax=ax, offset=0)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()
mae = round(abs(series.iloc[-int(len(series) * 0.3):].pct_change() - 
                recomposed_preds.iloc[-int(len(series) * 0.3):].pct_change()).mean(),10)
print(f'The MAE is {mae}')

# calculate the Hit rate
price_pred = pd.concat([series.iloc[-int(len(series) * 0.3):].pct_change(),recomposed_preds.iloc[-int(len(series) * 0.3):].pct_change()], axis=1)
price_pred.dropna(inplace=True)
price_pred['hit'] = np.where(np.sign(price_pred['Price']) == np.sign(price_pred['recomposed_preds']), 1, 0)
print(f"Hit rate: {round((price_pred['hit'].sum() / price_pred['hit'].count()) * 100,2)}%")

### Time Series Decomposition for prediction
# decompose the EURUSD series using the STL Method
# it's not a biased method when decomposing a time series
result = STL(series).fit()
result.plot()
plt.show()

# an example print out the first 5 rows of the seasonal series below
result.seasonal.head()

# run AR model with decomposed series as input
component_dict = {'seasonal': result.seasonal, 'trend': result.trend, 'residual': result.resid}
prediction_results = []
for component in ['seasonal', 'trend', 'residual']:
    historic = component_dict[component].iloc[:int(len(series) * 0.7)].to_list()
    test = component_dict[component].iloc[int(len(series) * 0.7):]
    predictions = []
    for i in range(len(test)):
        model = AR(historic)
        model_fit = model.fit()
        pred = model_fit.predict(start=len(historic), end=len(historic), dynamic=False)
        predictions.append(pred[0])
        historic.append(test[i])
    predictions = pd.Series(predictions, index=test.index, name=component)
    prediction_results.append(predictions)
    test_score = np.sqrt(mean_squared_error(test, predictions))
    print(f'Test for {component} MSE: {test_score}')
    # plot results
    plt.plot(test.iloc[:], label='Observed '+component)
    plt.plot(predictions.iloc[:], color='red', label='Predicted '+component)
    plt.legend()
    plt.show()
    
# recompose the data in an additive manner and calculate the RMSE
recomposed_preds = pd.concat(prediction_results,axis=1).sum(axis=1)
plt.plot(series.iloc[int(len(series) * 0.7):], label='Observed')
plt.plot(recomposed_preds, color='red', label='Predicted')
plt.legend()
plt.show()
test_score = np.sqrt(mean_squared_error(series.iloc[int(len(series) * 0.7):], recomposed_preds))
print(f'RMSE: {test_score}')

# plot a scatter chart and calculate the MAE
fig, ax = plt.subplots()
ax = sns.regplot(series.iloc[-int(len(series) * 0.3):].pct_change(),
            recomposed_preds.iloc[-int(len(series) * 0.3):].pct_change(), )
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('EURUSD Observed vs Predicted Values')
ax.grid(True, which='both')
ax.axhline(y=0, color='#888888')
ax.axvline(x=0, color='#888888')
sns.despine(ax=ax, offset=0)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()
mae = round(abs(series.iloc[-int(len(series) * 0.3):].pct_change() - 
                recomposed_preds.iloc[-int(len(series) * 0.3):].pct_change()).mean(),10)
print(f'The MAE is {mae}')

# calculate the Hit rate
price_pred = pd.concat([series.iloc[-int(len(series) * 0.3):].pct_change(), 
                recomposed_preds.iloc[-int(len(series) * 0.3):].pct_change()], axis=1)
price_pred.dropna(inplace=True)
price_pred.columns = ['Price', 'preds']
price_pred['hit'] = np.where(np.sign(price_pred['Price']) == np.sign(price_pred['preds']), 1, 0)
print(f"Hit rate: {round((price_pred['hit'].sum() / price_pred['hit'].count()) * 100,2)}%")
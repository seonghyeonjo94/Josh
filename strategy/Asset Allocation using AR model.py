import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
"""
SPY Equity Index
TLT 20+ Year Treasury Bond
TIP TIPS
IYR US Real Estate
EEM MSCI Emerging Markets
EFA MSCI EAFE
DBC Commodity
GLD Gold
"""
"""
VT 글로벌주식
TLT 장기국채
VCLT 장기회사채
EMLC 이머징 국가 채권
LTPZ 물가연동채(15+)
IAU 금
DBC 원자재
"""

start = datetime.datetime(2010, 7, 23)
end = datetime.datetime.today()

tickers = ['SPY', 'TLT', 'VCLT', 'EMLC', 'LTPZ', 'IAU', 'DBC']
all_data = {}
for ticker in tickers:
    all_data[ticker] = web.DataReader(ticker, 'yahoo' ,start=start, end=end)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices_monthly = prices.resample('M').last()

predictions = {}
for ticker in tickers:
    series = prices_monthly[ticker]
    result = STL(series).fit()
    components = {'seasonal': result.seasonal, 'trend': result.trend, 'residual': result.resid}
    pred_results = []
    for component in ['seasonal', 'trend', 'residual']:
        historic = components[component].iloc[:int(len(series) * 0.5)].to_list()
        test = components[component].iloc[int(len(series) * 0.5):]
        preds = []
        for i in range(len(test)):
            model = AR(historic)
            model_fit = model.fit()
            pred = model_fit.predict(start=len(historic), end=len(historic), dynamic=False)
            preds.append(pred[0])
            historic.append(test[i])
        preds = pd.Series(preds, index=test.index, name=component)
        pred_results.append(preds)
    predictions[ticker] = pd.DataFrame(pd.concat(pred_results,axis=1).sum(axis=1), columns=[ticker])
df = pd.DataFrame({tic: data[tic] for tic, data in predictions.items()})  

temp = prices_monthly.iloc[int(len(series) * 0.5)-23:]

wts = pd.DataFrame()
for i in range(len(temp)-23):
    data = temp.iloc[i:23+i]
    data = data.append(df.iloc[i])
    data = data.pct_change()
    
#    yearly_mean = (1 + data).prod() ** (12 / len(data)) - 1
    yearly_mean = data.iloc[-1] * 12
    yearly_vol = np.sqrt(12) * data.std()
    covs = data.cov() * 12
    
    # Simulation    
    numAssets = len(tickers)
    numPortfolio = 10000
    sim = pd.DataFrame()
    
    for j in range(numPortfolio):
        wt = np.random.rand(numAssets)
        wt = wt / np.sum(wt)
        expected_return = wt @ yearly_mean
        expected_vol = (wt.T @ covs @ wt)
        expected_sharpe = expected_return / expected_vol
        sim[j] = list([wt, expected_return, expected_vol, expected_sharpe])
        sim.index = ['wt', 'Return', 'Volatility', 'Sharpe']   
    target_maxsharpe = sim.loc['Sharpe'].max()
    target_maxsharpe_index = (sim.loc['Sharpe'] == target_maxsharpe).argmax()
    
    wts = wts.append(pd.DataFrame(sim.loc['wt'].iloc[target_maxsharpe_index].reshape(1,-1), columns=tickers))
    print(i)

wts.index = df.index
rets_monthly = prices_monthly.pct_change()
rets = rets_monthly.iloc[int(len(series) * 0.5):]

fee = 0.003
result = ReturnPortfolio(rets, wts)

portfolio_ret = result['ret']
turnover = pd.DataFrame((result['eop_weights'].shift(1) - result['bop_weights']).abs().sum(axis = 1))
portfolio_ret_net = portfolio_ret - (turnover * fee)     

PerformanceAnalysis(portfolio_ret_net)
plot_monthly_returns_heatmap(portfolio_ret_net)
plot_annual_returns(portfolio_ret_net)
ReturnStats(portfolio_ret_net)

plt.style.use("seaborn")
ax = wts.plot(label = 'Weights', kind='area', alpha=0.7)
ax.set_title('Portfolio Weights')
ax.legend(loc='best')

# All Weather Portfolio
tickers = ['VT', 'TLT', 'VCLT', 'EMLC', 'LTPZ', 'IAU', 'DBC']
start = '2013-5-1'
all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker, start)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices_monthly = prices.resample('M').last()
prices_monthly = prices_monthly.loc['2016-12':]

wts = np.zeros((prices_monthly.shape[0], prices_monthly.shape[1]))

for i in range(len(prices_monthly)):
    wts[i] = [0.35, 0.2, 0.075, 0.075, 0.2, 0.05, 0.05]

wts = pd.DataFrame(wts, columns=prices_monthly.columns, index=prices_monthly.index)
rets_monthly = prices_monthly.pct_change()

fee = 0.003
result = ReturnPortfolio(rets_monthly, wts)

portfolio_ret = result['ret']
turnover = pd.DataFrame((result['eop_weights'].shift(1) - result['bop_weights']).abs().sum(axis = 1))
portfolio_ret_net = portfolio_ret - (turnover * fee)     

PerformanceAnalysis(portfolio_ret_net)
plot_monthly_returns_heatmap(portfolio_ret_net)
plot_annual_returns(portfolio_ret_net)
ReturnStats(portfolio_ret_net)
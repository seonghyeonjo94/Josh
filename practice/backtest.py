from pack import *
from Jquant import *

#--- Download Raw Data ---#

tickers = ['SPY', 'IEV', 'EWJ', 'EEM', 'TLO', 'IEF', 'IYR', 'RWX', 'GLD', 'DBC']
start = '2007-12-30'
all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker, start)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.fillna(method = 'ffill')
rets = prices.pct_change(1)


#--- Basic Option ---#

fee = 0.0030
lookback = 12
num = 5


#--- Find Endpoints of Month ---#

s = pd.Series(np.arange(prices.shape[0]), index=prices.index)
ep = s.resample("M").max()


#--- Create Weight Matrix using 12M Momentum ---#

wts = list()

for i in range(lookback, len(ep)) :
    ## prices.index[ep[i]]       check the calendar
    cumret = prices.iloc[ep[i]] / prices.iloc[ep[i-12]] - 1
    K = rankdata(-cumret) <= num
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / num
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts.append(wt)
    
wts = pd.concat(wts)


#--- Calculate Portfolio Return & Turnover ---#
    
result = ReturnPortfolio(rets, wts)

portfolio_ret = result['ret']
turnover = pd.DataFrame((result['eop_weights'].shift(1) - result['bop_weights']).abs().sum(axis = 1))
portfolio_ret_net = portfolio_ret - (turnover * fee)     


#--- Calculate Cumulative Return ---#

port_cumret = ReturnCumulative(portfolio_ret_net)


#--- Calculate Drawdown ---#
    
port_dd = drawdown(portfolio_ret_net)


#--- Graph: Portfolio Return and Drawdown ---#

fig, axes = plt.subplots(2, 1)
port_cumret.plot(ax = axes[0], legend = None)
port_dd.plot(ax = axes[1], legend = None)


#--- Daily Return Frequency To Yearly Return Frequency ---#

yr_ret = apply_yearly(portfolio_ret_net)
yr_ret.plot(kind = 'bar')

#--- Calculate Portfolio Stats ---#

ReturnStats(portfolio_ret_net)
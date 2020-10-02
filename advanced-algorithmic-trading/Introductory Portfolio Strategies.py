from pack import *
from Jquant import *

# Ticker #1     #2     #3 
# SPY    60.0%  25.0%  12.5%
# IJS    0.0%   5.0%   12.5%
# EFA    0.0%   20.0%  12.5%
# EEM    0.0%   5.0%   12.5%
# AGG    40.0%  20.0%  12.5%
# JNK    0.0%   5.0%   12.5%
# DJP    0.0%   10.0%  12.5%
# RWR    0.0%   10.0%  12.5%

#--- Download Raw Data ---#

tickers = ['SPY', 'IJS', 'EFA', 'EEM', 'AGG', 'JNK', 'DJP', 'RWR']
start = '2003-07-01'
end = '2016-12-31'
all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker, start, end)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.fillna(method = 'ffill')
rets = prices.pct_change(1)

########################################################################
##################### strategy #1 equty_bond_60_40 #####################
########################################################################

start_date = datetime.datetime(2003, 9, 30)
end_date = datetime.datetime(2016, 10, 12)

df = rets[['SPY', 'AGG']]
df = df.loc[start_date : end_date]

#--- Basic Option ---#

fee = 0.0030
lookback = 1

#--- Find Endpoints of Month ---#

s = pd.Series(np.arange(df.shape[0]), index=df.index)
ep = s.resample("M").max()

#--- Create Weight Matrix ---#

wts = list()

for i in range(lookback, len(ep)) :
    ## df.index[ep[i]]       check the calendar
    
    wt = np.repeat(0.00, df.shape[1], axis = 0)
    
    wt = pd.DataFrame(data = wt.reshape(1,df.shape[1]),
                      index = [df.index[ep[i]]],
                      columns = df.columns)
    wt[wt.columns[0]] = 0.6
    wt[wt.columns[1]] = 0.4
    wts.append(wt)
    
wts = pd.concat(wts)

#--- Calculate Portfolio Return & Turnover ---#
    
result = ReturnPortfolio(df, wts)

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

#--- Calculate Portfolio Stats

ReturnStats(portfolio_ret_net)

########################################################################
#################### strategy #2 strategic weight ######################
########################################################################

start_date = datetime.datetime(2007, 12, 6)
end_date = datetime.datetime(2016, 10, 12)

df = rets
df = df.loc[start_date : end_date]

#--- Basic Option ---#

fee = 0.0030
lookback = 1

#--- Find Endpoints of Month ---#

s = pd.Series(np.arange(df.shape[0]), index=df.index)
ep = s.resample("M").max()

#--- Create Weight Matrix ---#

wts = list()

for i in range(lookback, len(ep)) :
    ## df.index[ep[i]]       check the calendar
    
    wt = np.repeat(0.00, df.shape[1], axis = 0)
    
    wt = pd.DataFrame(data = wt.reshape(1,df.shape[1]),
                      index = [df.index[ep[i]]],
                      columns = df.columns)
    wt[wt.columns[0]] = 0.25
    wt[wt.columns[1]] = 0.05
    wt[wt.columns[2]] = 0.2
    wt[wt.columns[3]] = 0.05
    wt[wt.columns[4]] = 0.2
    wt[wt.columns[5]] = 0.05
    wt[wt.columns[6]] = 0.1
    wt[wt.columns[7]] = 0.1
    wts.append(wt)
    
wts = pd.concat(wts)

#--- Calculate Portfolio Return & Turnover ---#
    
result = ReturnPortfolio(df, wts)

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

#--- Calculate Portfolio Stats

ReturnStats(portfolio_ret_net)

########################################################################
###################### strategy #3 equal weight ########################
########################################################################

start_date = datetime.datetime(2007, 12, 6)
end_date = datetime.datetime(2016, 10, 12)

df = rets
df = df.loc[start_date : end_date]

#--- Basic Option ---#

fee = 0.0030
lookback = 1

#--- Find Endpoints of Month ---#

s = pd.Series(np.arange(df.shape[0]), index=df.index)
ep = s.resample("M").max()

#--- Create Weight Matrix ---#

wts = list()

for i in range(lookback, len(ep)) :
    ## df.index[ep[i]]       check the calendar

    wt = np.repeat(1/ len(tickers), df.shape[1], axis = 0)
    
    wt = pd.DataFrame(data = wt.reshape(1,df.shape[1]),
                      index = [df.index[ep[i]]],
                      columns = df.columns)

    wts.append(wt)
    
wts = pd.concat(wts)

#--- Calculate Portfolio Return & Turnover ---#
    
result = ReturnPortfolio(df, wts)

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

#--- Calculate Portfolio Stats

ReturnStats(portfolio_ret_net)
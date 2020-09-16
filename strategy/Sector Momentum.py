from pack import *
from Jquant import *
"""
XLY(4) Consumer Discretionary
XLP(4) Consumer Staples
XLE Energy
XLF Financials
XLV HealthCare
XLI Industrials
XLB Materials
XLK Technology
XLU Utilities
TLT(5) 20+ Year Treasury Bond
IYR US Real Estate
EEM MSCI Emerging Markets
EFA MSCI EAFE
"""

# Download ETF prices

start_date = '2014-01-01'
end_date = '2020-6-25'
all_data = {}
tickers = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'IYR', 'EEM', 'EFA']

for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker, start_date, end_date)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.fillna(method = 'ffill')
rets = prices.pct_change(1)

##############################################################################
s = pd.Series(np.arange(prices.shape[0]), index=prices.index)
ep = s.resample("M").max()

lookback = 1

# Create Weight Matrix

wts = list()

for i in range(lookback, len(ep)) :
    ## prices.index[ep[i]]       check the calendar
    cumret = prices.iloc[ep[i]] / prices.iloc[ep[i-1]] - 1
    K = rankdata(-cumret) == 1
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts.append(wt)
    
wts = pd.concat(wts)
wts = wts.shift()

# Calculate Portfolio Return
    
result = ReturnPortfolio(rets, wts[1:])
ReturnCumulative(result['ret']).plot()

spy = fdr.DataReader('spy', start_date, end_date)
spy = spy['Close'].pct_change()

# Performance measure
PerformanceAnalysis(result['ret'], spy)
plot_annual_returns(result['ret'])
plot_monthly_returns_heatmap(result['ret'])
ReturnStats(result['ret'])

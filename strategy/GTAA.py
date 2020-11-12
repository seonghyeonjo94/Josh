"""
US Large Cap, S&P 500 : SPY
Foreign Developed, MSCI EAFE : EFA
US 10-Year Government Bonds : IEF, BIV
Commodities, Goldman Sachs Commodity Index : GSG
Real Estate Investment Trusts, NAREIT Index : VNQ
"""

spy = fdr.DataReader('SPY', '1990', '2013')
spy = pd.DataFrame(spy['Close'])

spy = spy.resample('M').last()
spy10MA = spy.rolling(10).mean()
spy.plot()
spy10MA.plot

ret = spy.pct_change()

rule = spy > spy10MA
ReturnCumulative(ret[rule.shift(1)]).plot()

ReturnStats(ret[rule.shift(1)])
PerformanceAnalysis(ret[rule.shift(1)])
PerformanceAnalysis(ret)

##############################################################################

tickers = ['SPY', 'EFA', 'BIV', 'GSG', 'VNQ']
all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.fillna(method = 'ffill')
prices = prices.dropna()
prices = prices.resample('M').last()
rets = prices.pct_change(1)

ma10 = prices.rolling(10).mean()

rule = prices > ma10
temp = rets[rule.shift(1)]
ReturnCumulative(temp).plot()

wts = pd.DataFrame(np.zeros([temp.shape[0], temp.shape[1]]), index=temp.index, columns=temp.columns)
for i in range(len(temp)):
    wts.iloc[i] = 0.2

result = ReturnPortfolio(temp, wts)

portfolio_ret = result['ret']
PerformanceAnalysis(portfolio_ret)

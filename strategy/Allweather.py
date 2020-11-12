"""
Global Equity : VT 40%
US 20+ Treasury Bond : TLT 20%
US Corporate bond : VCLT 7.5%
Emerging market Treasury bond : VWOB 7.5%
TIPS : LTPZ 15%
Commodities : DBC 5%
Gold : IAU 5%
"""
tickers = ['VT', 'TLT', 'VCLT', 'VWOB', 'LTPZ', 'DBC', 'IAU']
all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.dropna()
prices = prices.resample('M').last()
rets = prices.pct_change()

wts = pd.DataFrame(np.zeros([rets.shape[0], rets.shape[1]]), index=rets.index, columns=rets.columns)
for i in range(len(rets)):
    wts.iloc[i] = [0.4 , 0.2 , 0.075 , 0.075 , 0.15 , 0.05, 0.05]
    
result = ReturnPortfolio(rets, wts)
PerformanceAnalysis(result['ret'])

##############################################################################
"""
S&P500 : SPY 60%
US 20+ Tresury Bond : TLT 40%
"""
tickers = ['SPY', 'TLT']
all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.dropna()
prices = prices.resample('M').last()
rets = prices.pct_change()

wts = pd.DataFrame(np.zeros([rets.shape[0], rets.shape[1]]), index=rets.index, columns=rets.columns)
for i in range(len(rets)):
    wts.iloc[i] = [0.6 , 0.4]
    
result = ReturnPortfolio(rets, wts)
PerformanceAnalysis(result['ret'])
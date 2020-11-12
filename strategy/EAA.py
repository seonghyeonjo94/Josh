"""
US Large Cap, S&P 500 : SPY
Foreign Developed, MSCI EAFE : EFA
Emerging Market : EEM
US tech : IYW
Japan Topix : JPXN(Nikkei400)
US 10-Year Government Bonds : IEF, BIV
US High Yield : HYG -> US Investment Grade : LQD
"""
num = 7
tickers = ['SPY', 'EFA', 'EEM', 'IYW', 'JPXN', 'BIV', 'LQD']
all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.dropna()
prices = prices.resample('M').last()
rets = prices.pct_change()

wts = pd.DataFrame(np.zeros([rets.shape[0], rets.shape[1]]), index=rets.index, columns=rets.columns)
for i in range(len(rets)):
    wts.iloc[i] = 1 / num

bench_result = ReturnPortfolio(rets, wts)
benchmark = bench_result['ret']
benchmark.columns = ['benchmark']
ReturnCumulative(benchmark).plot()
ReturnStats(benchmark)

ret_12m = prices / prices.shift(12) - 1
ret_12m = ret_12m.dropna()
ret_12m_filter = ret_12m[ret_12m > 0]
ret_12m_filter = ret_12m_filter.fillna(0)

z = pd.DataFrame()
for i in range(len(ret_12m)):
    z[i] =ret_12m_filter.iloc[i] > ret_12m_filter.quantile(0.5, axis=1)[i]
z = z.T
z.index = ret_12m.index
ret_12m_filter = ret_12m_filter[z]
ret_12m_filter = ret_12m_filter.fillna(0)

temp = pd.merge(benchmark, rets, how='outer', left_index=True, right_index=True)
corr = temp.rolling(12).corr()['benchmark'].unstack()
corr = corr[corr.columns[0:num]]
corr = corr.loc[ret_12m.index]
corr = corr[ret_12m_filter.columns]

# defensive weights = sqrt{r(1-c)}, for r > 0, else w = 0
# offensive weights = (1-c) * r^2, for r > 0, else w = 0
#z_def = np.sqrt(ret_12m_filter * (1-corr))
z_def = (1-corr) * ret_12m_filter**2
crash_protection = 1 - (ret_12m > 0).sum(axis=1)/num

wt = pd.DataFrame()
for i in range(len(z_def)):
    wt[i] = z_def.iloc[i] / z_def.sum(axis=1)[i]
wt = wt.T
wt.index = z_def.index

wts = pd.DataFrame()
for i in range(len(crash_protection)):
    wts[i] = wt.iloc[i] * (1 - crash_protection[i])
wts = wts.T
wts.index = crash_protection.index

##############################################################################
bond = fdr.DataReader('TLT')
bond = bond['Close'].resample('M').last()
bond = bond.pct_change()
bond = pd.DataFrame(bond[wts.index])
bond.columns = ['protection']

crash_protection = pd.DataFrame(crash_protection)
crash_protection.columns = ['protection']

data = pd.merge(rets, bond, how='outer', left_index=True, right_index=True)
wts = pd.merge(wts, crash_protection, how='outer', left_index=True, right_index=True)
result = ReturnPortfolio(data, wts)

PerformanceAnalysis(result['ret'], benchmark)
plot_annual_returns(result['ret'])
ReturnStats(result['ret'])

##############################################################################
from fredapi import Fred
int_rate = fred.get_series('DGS10')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(ReturnCumulative(result['ret']), label='Strategy', color='forestgreen')
ax_temp = ax.twinx()
ax_temp.plot(int_rate[result['ret'].index[0]:], label='10Y')
ax.legend(loc='best')
ax_temp.legend(loc='best')
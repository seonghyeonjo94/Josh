from pack import *
from Jquant import *

def transpose_to_numeric(x):
    x = x.replace(',','')
    x = pd.to_numeric(x)
    return x

def transpose_to_date(x):
    x = pd.Timestamp(x)
    return x

ticker = get_KOR_ticker()
ticker['시가총액'] = ticker['시가총액'].apply(transpose_to_numeric)

prices = pd.read_csv(r'C:\Users\a\Downloads\quant\Python\data\price.csv')
prices['Unnamed: 0'] = prices['Unnamed: 0'].apply(transpose_to_date)
prices.index = prices['Unnamed: 0']
del prices['Unnamed: 0']

mkt_cap = ticker[['종목코드','시가총액']]
K = mkt_cap['시가총액'] > mkt_cap['시가총액'].quantile(0.3)
mkt_cap = mkt_cap[K]

# prices = prices[mkt_cap['종목코드']]
# Basic Option

fee = 0.0030
lookback = 12

# Find Endpoints of Month

s = pd.Series(np.arange(prices.shape[0]), index=prices.index)
ep = s.resample("M").max()

# Create Weight Matrix using 12-1M Momentum

wts1 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = cumret > cumret.quantile(0.90)
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts1.append(wt)
    
wts1 = pd.concat(wts1)

rets = prices.pct_change()
rets = rets.fillna(0)
rets = rets.loc[wts1.index[0]:]
result1 = ReturnPortfolio(rets, wts1)
print('result:1 Complete')

wts2 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.90) >= cumret) & (cumret > cumret.quantile(0.80))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts2.append(wt)
    
wts2 = pd.concat(wts2)

result2 = ReturnPortfolio(rets, wts2)
print('result:2 Complete')

wts3 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.80) >= cumret) & (cumret > cumret.quantile(0.70))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts3.append(wt)
    
wts3 = pd.concat(wts3)

result3 = ReturnPortfolio(rets, wts3)
print('result:3 Complete')

wts4 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.70) >= cumret) & (cumret > cumret.quantile(0.60))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts4.append(wt)
    
wts4 = pd.concat(wts4)

result4 = ReturnPortfolio(rets, wts4)
print('result:4 Complete')

wts5 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.60) >= cumret) & (cumret > cumret.quantile(0.50))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts5.append(wt)
    
wts5 = pd.concat(wts5)

result5 = ReturnPortfolio(rets, wts5)
print('result:5 Complete')

wts6 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.50) >= cumret) & (cumret > cumret.quantile(0.40))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts6.append(wt)
    
wts6 = pd.concat(wts6)

result6 = ReturnPortfolio(rets, wts6)
print('result:6 Complete')

wts7 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.40) >= cumret) & (cumret > cumret.quantile(0.30))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts7.append(wt)
    
wts7 = pd.concat(wts7)

result7 = ReturnPortfolio(rets, wts7)
print('result:7 Complete')

wts8 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.30) >= cumret) & (cumret > cumret.quantile(0.20))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts8.append(wt)
    
wts8 = pd.concat(wts8)

result8 = ReturnPortfolio(rets, wts8)
print('result:8 Complete')

wts9 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = (cumret.quantile(0.20) >= cumret) & (cumret > cumret.quantile(0.10))
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts9.append(wt)
    
wts9 = pd.concat(wts9)

result9 = ReturnPortfolio(rets, wts9)
print('result:9 Complete')

wts10 = list()

for i in range(lookback, len(ep)) :
    cumret = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    K = cumret.quantile(0.10) >= cumret
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / K.sum()
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    wts10.append(wt)
    
wts10 = pd.concat(wts10)

result10 = ReturnPortfolio(rets, wts10)
print('result:10 Complete')
# Calculate Cumulative Return
data = pd.merge(result1['ret'], result2['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result3['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result4['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result5['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result6['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result7['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result8['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result9['ret'], how='outer', right_index=True, left_index=True)
data = pd.merge(data, result10['ret'], how='outer', right_index=True, left_index=True)
data.columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']

ReturnCumulative(data).plot()

# Calculate RankIC
s = pd.Series(np.arange(prices.shape[0]), index=prices.index)
ep = s.resample("M").max()
prices_monthly = prices.resample('M').max()
rets_monthly = prices_monthly.pct_change()
lookback = 12

rank = list()

for i in range(lookback, len(ep)) :

    x = prices.iloc[ep[i-1]] / prices.iloc[ep[i-12]] - 1
    temp = np.array(x)
    temp = pd.DataFrame(data = temp.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = prices.columns)
    rank.append(temp)
    
rank = pd.concat(rank)

rankic = RankIC(rets_monthly,rank)
rankic.mean()
rankic.mean() / (rankic.std()/np.sqrt(len(rankic)))

rolling = list()

for i in range(lookback, len(rankic)):
    x = rankic.iloc[i-lookback:i].last('12M').mean()
    
    temp = np.array(x)
    temp = pd.DataFrame(data = temp,
                        index = [rankic.index[i]],
                        columns = rankic.columns)
    rolling.append(temp)
    
rolling = pd.concat(rolling)
rolling.plot()
rankic.mean()

# lowvol
ret_last_12m = rets.last('12M')
std_last_12m = ret_last_12m.std()
K = rankdata(std_last_12m) <= 30

# momentum
ret_last_12m = rets.last('12M')
ret_cum_12m = ReturnCumulative(ret_last_12m)
K = rankdata(-ret_cum_12m) <= 30

# Sharpe Ratio
sharpe_12m = ret_cum_12m / std_last_12m

##############################################################################
fs = pd.read_pickle(r'C:\Users\a\Downloads\quant\Python\data\fs.pickle')

ticker = list(fs.keys())

gp = pd.DataFrame()
for tic in ticker:
    try:
        gp[tic] = fs[tic].loc['매출총이익']
    except KeyError:
        continue
    
asset = pd.DataFrame()
for tic in ticker:
    try:
        asset[tic] = fs[tic].loc['자산']
    except KeyError:
        continue
    

asset = asset[pd.Index.intersection(sales.columns, asset.columns)]
gp = sales[pd.Index.intersection(sales.columns, asset.columns)]

gpa = gp/asset
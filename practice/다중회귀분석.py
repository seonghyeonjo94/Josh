from pack import *
from josh import *
from data_crawler import *

import statsmodels.api as sm
import statsmodels.formula.api as smf

start_date = datetime.datetime(2005, 1, 1)
end_date = datetime.datetime.today()
start_test = datetime.datetime(2016, 1, 1)
 
# Create a lagged series of the SPY
df = create_lagged_series('spy', start_date, start_test)
df.dropna(inplace=True)

a = ['Today']
for i in range(1,51):
    print('Lag'+ str(i))
    a.append('Lag'+str(i))

data = df[a]

#model = smf.ols(formula = 'Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Lag6 + Lag7 + Lag8 + Lag9 + Lag10 + Lag11 + Lag12 + Lag13 + Lag14 + Lag15 + Lag16 + Lag17 + Lag18 + Lag19 + Lag20 + Lag21 + Lag22 + Lag23 + Lag24 + Lag25 + Lag26 + Lag27 + Lag28 + Lag29 + Lag30 + Lag31 + Lag32 + Lag33 + Lag34 + Lag35 + Lag36 + Lag37 + Lag38 + Lag39 + Lag40 + Lag41 + Lag42 + Lag43 + Lag44 + Lag45 + Lag46 + Lag47 + Lag48 + Lag49 + Lag50' , data=data)
model = smf.ols(formula = 'Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Lag6 + Lag7 + Lag8 + Lag9 + Lag10 + Lag11 + Lag12 + Lag13 + Lag14 + Lag15 + Lag16 + Lag17 + Lag18 + Lag19 + Lag20 + Lag21 + Lag22 + Lag23 + Lag24 + Lag25 + Lag26 + Lag27 + Lag28 + Lag29 + Lag30', data=data)
result = model.fit()
result.summary()

##############################################################################
kospi = fdr.DataReader('KS11', start_date, start_test)
kospi = pd.DataFrame(kospi['Close'].pct_change())
kospi.dropna(inplace=True)
kospi.columns = ['kospi']

test = pd.merge(df['Today'], kospi, how='outer', right_index=True, left_index=True)
model = smf.ols(formula = 'Today ~ kospi', data=test)
result = model.fit()
result.summary()

##############################################################################
stoxx = fdr.DataReader('STOXX50', start_date, start_test)
stoxx = pd.DataFrame(stoxx['Open'].pct_change())
stoxx.dropna(inplace=True)
stoxx.columns = ['stoxx']

test = pd.merge(df['Today'], stoxx, how='outer', right_index=True, left_index=True)
model = smf.ols(formula = 'Today ~ stoxx', data=test)
result = model.fit()
result.summary()

##############################################################################
csi = fdr.DataReader('CSI300', start_date, start_test)
csi = pd.DataFrame(csi['Close'].pct_change())
csi.dropna(inplace=True)
csi.columns = ['csi']

test = pd.merge(df['Today'], csi, how='outer', right_index=True, left_index=True)
model = smf.ols(formula = 'Today ~ csi', data=test)
result = model.fit()
result.summary()

##############################################################################
exchange = fdr.DataReader('USD/KRW', start_date, start_test)
exchange = pd.DataFrame(exchange['Close'].pct_change())
exchange = exchange.shift()
exchange.dropna(inplace=True)
exchange.columns = ['exchange']

test = pd.merge(df['Today'], exchange, how='outer', right_index=True, left_index=True)
model = smf.ols(formula = 'Today ~ exchange', data=test)
result = model.fit()
result.summary()

##############################################################################
gold = fdr.DataReader('GC', start_date, start_test)
gold = pd.DataFrame(gold['Close'].pct_change())
gold = gold.shift()
gold.dropna(inplace=True)
gold.columns = ['gold']

test = pd.merge(df['Today'], gold, how='outer', right_index=True, left_index=True)
model = smf.ols(formula = 'Today ~ gold', data=test)
result = model.fit()
result.summary()

##############################################################################
wti = fdr.DataReader('CL', start_date, start_test)
wti = pd.DataFrame(wti['Close'].pct_change())
wti = wti.shift()
wti.dropna(inplace=True)
wti.columns = ['wti']

test = pd.merge(df['Today'], wti, how='outer', right_index=True, left_index=True)
model = smf.ols(formula = 'Today ~ wti', data=test)
result = model.fit()
result.summary()

##############################################################################
tickers = ['SPY', 'KS11', 'STOXX50', 'CSI300', 'USD/KRW', 'GC', 'CL']

all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker, start_date, start_test)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.fillna(method = 'ffill')
rets = prices.pct_change(1)
rets.columns = ['SPY', 'KS11', 'STOXX50','CSI300', 'USDKRW', 'GC', 'CL']
rets['Direction'] = np.sign(rets['SPY'])
model = smf.ols(formula = 'SPY ~ KS11 + STOXX50 + CSI300 + USDKRW + GC + CL', data=rets)
result = model.fit()
result.summary()

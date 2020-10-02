import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata

price = pd.read_excel(r'C:\Users\USER\price.xlsx', index_col='Unnamed: 0')
fs = pd.read_excel(r'C:\Users\USER\fs.xlsx', index_col='Unnamed: 0')

# KOSPI
mkt_price = pd.read_excel(r'C:\Users\USER\mktprice.xlsx', index_col='Unnamed: 0')
mkt_ret = ( mkt_price / mkt_price.shift(1)) - 1

# 수익률 구하기
price = price.fillna(method = 'ffill')
n = len(price)
ret = np.divide(price[1:n], price[0:(n-1)]) - 1
ret = (price / price.shift(1)) - 1 # shift 사용
ret = ret.fillna(method = 'ffill')

# 누적수익률 구하기
cum_ret = np.cumprod(1+ret) - 1

# 연율화, 월별 수익률 구하기
sub_mon_first = price.resample('M').first()
sub_mon_last = price.resample('M').last()
monthly_ret = (sub_mon_last / sub_mon_first) - 1

sub_yr_first = price.resample('A').first()
sub_yr_last = price.resample('A').last()
yearly_ret = (sub_yr_last / sub_yr_first) - 1

# 저변동성
ret_std_12m = ret.resample('M').std().last('12M')
invest_lowvol = ret_std_12m.mean().rank().sort_values().iloc[0:30]

# 모멘텀
ret_last_12m = ret.last('12M')
ret_cum_12m = np.cumprod(1+ret_last_12m) - 1
invest_mom = (-ret_cum_12m.iloc[-1]).rank().sort_values().iloc[0:30]

# linear regression
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# 1개 칼럼 np.array로 변환
independent_var = np.array(mkt_ret['KOSPI'].dropna()).reshape(-1, 1)
dependent_var = np.array(ret['005930'].dropna()).reshape(-1, 1)

# Linear Regression
regr = LinearRegression()
regr.fit(independent_var, dependent_var)

result = {'Slope':regr.coef_[0,0], 'Intercept':regr.intercept_[0], 'R^2':regr.score(independent_var, dependent_var)}
result # R^2이 1에 가까울수록 상관관계가 높음

# 추세선을 그래프로 그리기
plt.figure(figsize=(5,5))
plt.scatter(independent_var, dependent_var, marker='.', color='skyblue')
plt.plot(independent_var, regr.predict(independent_var), color='r', linewidth=3)
plt.grid(True, color='0.7', linestyle=':', linewidth=1)
plt.xlabel('KOSPI')
plt.ylabel('삼성전자')

# Efficient frontier
price = pd.read_excel(r'C:\Users\USER\price.xlsx', index_col='Unnamed: 0')
symbols = ['005930', '068270', '005380', '055550', '017670']
names = ['삼성전자', '셀트리온', '현대차', '신한지주', 'SK텔레콤']

prices = price[symbols]
rets = (prices / prices.shift(1)) - 1
cum_rets = np.cumprod(1+rets) - 1
prices.columns = names
rets.columns = names
covs = rets.cov() * 252

cum_rets.plot()

sub_yr_first = prices.resample('A').first()
sub_yr_last = prices.resample('A').last()
yearly_mean = ((sub_yr_last / sub_yr_first) - 1).mean()
yearly_vol = rets.resample('A').std().mean()

yearly_stat = pd.DataFrame([yearly_mean, yearly_vol])
yearly_stat.index = [['yearly_mean', 'yearly_vol']]

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.grid(True)
plt.xlim(0.01, 0.035)
plt.ylim(-0.15, 0.4)
plt.scatter(yearly_stat.iloc[1], yearly_stat.iloc[0])

# Simulation
numAssets = len(symbols)
numPortfolio = 10000
sim = pd.DataFrame()
for i in range(numPortfolio):
    wt = np.random.rand(numAssets)
    wt = wt / np.sum(wt)
    expected_return = wt @ yearly_mean
    expected_vol = (wt.T @ covs @ wt)
    expected_sharpe = (expected_return-0.05) / expected_vol
    sim[i] = list([wt, expected_return, expected_vol, expected_sharpe])
    sim.index = ['wt', 'Return', 'Volatility', 'Sharpe']

target_minvol = sim.loc['Volatility'].min()
target_maxsharpe = sim.loc['Sharpe'].max()

# plot simulation
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.grid(True)
plt.xlim(0.00, 0.15)
plt.ylim(-0.15, 0.4)
plt.scatter(sim.loc['Volatility'], sim.loc['Return'], c=sim.loc['Sharpe'])

# plot min vol and max sharpe
x = sim.loc['Volatility'][sim.loc['Sharpe'][sim.loc['Sharpe'] == target_maxsharpe].index]
y = sim.loc['Return'][sim.loc['Sharpe'][sim.loc['Sharpe'] == target_maxsharpe].index]

plt.xlabel('Volatility')
plt.ylabel('Return')
plt.grid(True)
plt.xlim(0.00, 0.15)
plt.ylim(-0.15, 0.4)
plt.scatter(sim.loc['Volatility'], sim.loc['Return'], c=sim.loc['Sharpe'])
plt.plot([0.00, x],[0.05, y])

# Moving average
price = price['005930']
price = pd.DataFrame(price)
ma5 = price.rolling(window=5).mean()
ma20 = price.rolling(window=20).mean()
ma60 = price.rolling(window=60).mean()
ma120 = price.rolling(window=120).mean()

# Insert columns
price.insert(len(price.columns), "MA5", ma5)
price.insert(len(price.columns), "MA20", ma20)
price.insert(len(price.columns), "MA60", ma60)
price.insert(len(price.columns), "MA120", ma120)

# Plot
plt.plot(price.index, price['005930'], label='Adj Close')
plt.plot(price.index, price['MA5'], label='MA5')
plt.plot(price.index, price['MA20'], label='MA20')
plt.plot(price.index, price['MA60'], label='MA60')
plt.plot(price.index, price['MA120'], label='MA120')

plt.legend(loc="best")
plt.grid()
plt.show()

# BollingerBand
pd.options.display.float_format = '{:.2f}'.format

df = pd.DataFrame(price['005930'])

w_size = 20 # 볼린져밴드 이동평균 산출 기간
pb     = 2  # 볼린져밴드 승수(percent bandwidth)
df['Moving Average']      = df['005930'].rolling(window=w_size, min_periods=1).mean()
df['Standard Deviation']  = df['005930'].rolling(window=w_size, min_periods=1).std()
df['Upper BollingerBand'] = df['Moving Average'] + (df['Standard Deviation'] * pb)
df['Lower BollingerBand'] = df['Moving Average'] - (df['Standard Deviation'] * pb)

df[['005930', 'Upper BollingerBand', 'Lower BollingerBand']].plot(figsize=(10,6))

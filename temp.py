import os

os.getcwd()
os.chdir(r'C:\Users\Samsung\Downloads\quant\Python\code')

from pack import *
from josh import *
from data_crawler import *

ticker = get_KOR_ticker()
get_KOR_fs()
get_KOR_price()

prices = get_US_price('2000-1-1', '2020-8-10')

fs = pd.read_pickle(r'C:\Users\Samsung\Downloads\quant\Python\data\fs.pickle')

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

##############################################################################
"""
WALCL : Assets: Total Assets: Total Assets (Less Eliminations from Consolidation)
M2 : M2 Money Stock
T10Y2Y : 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
DGS10 : 10-Year Treasury Constant Maturity Rate
T10YIE : 10-Year Breakeven Inflation Rate
DFII10 : 10-Year Treasury Inflation-Indexed Security, Constant 
USD3MTD156N : 3-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
TEDRATE : TED Spread
BAMLH0A0HYM2 : ICE BofA US High Yield Index Option-Adjusted Spread
GDPC1 : Real Gross Domestic Product
CPIAUCSL : Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
INDPRO : Industrial Production Index
UNRATE : Unemployment Rate
DCOILWTICO : Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma
DTWEXBGS : Trade Weighted U.S. Dollar Index: Broad, Goods and Services
DEXUSEU : U.S. / Euro Foreign Exchange Rate
DGORDER : Manufacturers' New Orders: Durable Goods
PCEPI : Personal Consumption Expenditures: Chain-type Price Index (PCE)
PCEPILFE : Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index) (Core PCE)
"""

from fredapi import Fred
fred = Fred(api_key='cc3ef5555bab6801e384d94c871cc4ce')

tickers = ['WALCL', 'M2', 'T10Y2Y', 'DGS10', 'T10YIE', 'DFII10', 'USD3MTD156N', 'TEDRATE',
           'BAMLH0A0HYM2', 'GDPC1', 'CPIAUCSL', 'INDPRO', 'UNRATE', 'DCOILWTICO', 'DTWEXBGS',
           'DEXUSEU', 'DGORDER', 'PCEPI', 'PCEPILFE']
all_data = {}
for ticker in tickers:
    all_data[ticker] = fred.get_series(ticker)
    print(ticker)
    
df = pd.DataFrame({tic: data for tic, data in all_data.items()})
df = df.fillna(method='ffill')
df['2010':]['T10Y2Y'].plot()

plt.style.use("seaborn")
plt.figure(figsize=(10,6))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.plot(df['2010':]['T10Y2Y'], label='10Y-2Y', color='forestgreen', alpha=0.7, lw=0.5)
ax1.set_title('10Y-2Y')
ax1.legend(loc='best')

ax_temp = ax1.twinx()
ax_temp.plot(df['2010':]['DTWEXBGS'], label='Dollar Index',alpha=0.7, lw=0.5)

##############################################################################
plt.style.use("seaborn")
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

ax1.plot(df['2010':]['DFII10'], label='10Y TIPS', color='forestgreen', alpha=0.7, lw=0.5)
ax1.set_title('TIPS & WTI')
ax1.legend(loc='best')
ax1_temp = ax1.twinx()
ax1_temp.plot(df['2010':]['DCOILWTICO'], label = 'WTI', alpha=0.7, lw=0.5)
ax1_temp.legend(loc='upper left')

ax1.plot(df['2010':]['DFII10'], label='10Y TIPS', alpha=0.7, lw=0.5)

##############################################################################
plt.style.use("seaborn")
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

ax1.plot(df['2010':][['T10Y2Y', 'DGS10', 'DFII10']])
ax1.plot(df['2010':]['T10Y2Y'], label='10Y-2Y')
ax1.plot(df['2010':]['DGS10'], label='10Y')
ax1.plot(df['2010':]['DFII10'], label='TIPS')
ax1.legend(loc='best')

df.corr()
df_corr = df.corr()

##############################################################################
### 이거는 딥러닝
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

spy = fdr.DataReader('spy', '2000')
spy = pd.DataFrame(spy['Close'])

data = spy.resample('M').last()

for i in [1, 3, 6, 12, 24, 36]:
    data['%sM_ret' % str(i)] = data['Close'].pct_change(i)
    
data['forward_ret'] = data['1M_ret'].shift(-1)
data = data.drop('Close', axis=1)

data = data.dropna()

X = data[data.columns[0:-1]]
y = data[data.columns[-1]]

X = np.array(X)
y = np.array(y).reshape(y.shape[0],1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size = 0.3, shuffle=False)

model = Sequential()
model.add(LSTM(256,input_shape=(6,1)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
#Fit model
early_stopping = EarlyStopping(patience=20)
model.fit(X_train, y_train, validation_split=0.2, verbose=1,
          batch_size=1, epochs=300, callbacks=[early_stopping])


Xt = model.predict(X_test)

#plt.plot(scl.inverse_transform(np.reshape(y_test, (-1,1))), label='actual')
#plt.plot(scl.inverse_transform(Xt), label='prediction')
plt.plot(Xt, label='prediction')
plt.plot(y_test, label='actual')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
plt.show()


actual = scl.inverse_transform(np.reshape(y_test, (-1, 1)))
pred = scl.inverse_transform(Xt)

##############################################################################
### 이거는 머신러닝
from pack import *
from Jquant import *

import sklearn
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA, 
    QuadraticDiscriminantAnalysis as QDA
)
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

random_state = 42
n_estimators = 1000
n_jobs = 1

price = fdr.DataReader('AAPL', '2000')
price = pd.DataFrame(price['Close'])

data = price.resample('M').last()

for i in [1, 3, 6, 12, 24, 36]:
    data['%sM_ret' % str(i)] = data['Close'].pct_change(i)
    
data['forward_ret'] = data['1M_ret'].shift(-1)
data = data.drop('Close', axis=1)

data = data.dropna()

X = data[data.columns[0:-1]]
y = np.sign(data['forward_ret'])

start_test = datetime.datetime(2016, 1, 1)

X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

#model = LDA()
    
#model = QDA()
    
#model = RandomForestClassifier(
#    n_estimators=n_estimators,
#    n_jobs=n_jobs,
#    random_state=random_state,
#    max_depth=10
#)
    
model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=n_estimators,
    random_state=random_state,
    n_jobs=n_jobs
)
        
#model = GradientBoostingClassifier(
#    n_estimators=n_estimators,
#    random_state=random_state
#)

#model = LogisticRegression()
    
model.fit(X_train, y_train)
model

# Make an array of predictions on the test set
pred = pd.DataFrame(model.predict(X_test))
pred.index = y_test.index

invest = (pred.shift(1) == 1)
price_mon = price.resample('M').last()
price_mon.columns = pred.columns
ret_mon = price_mon.pct_change()

PerformanceAnalysis(ret_mon[invest].loc[start_test:])
plot_annual_returns(ret_mon[invest].loc[start_test:])
plot_monthly_returns_heatmap(ret_mon[invest].loc[start_test:])
ReturnStats(ret_mon[invest].loc[start_test:])

##############################################################################
# 미국주식 재무데이터 크롤링
url = 'https://www.macrotrends.net/stocks/charts/AAPL/apple/financial-statements'
request_headers = {
    'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like gecko) Chrome/85.0.4183.102 Safari/537.36')
                    }
html = requests.get(url, headers = request_headers)
tables = pd.read_html(html.text)


myURL = 'https://www.macrotrends.net/stocks/charts/MSFT/microsoft/income-statement?freq=A'
page = requests.get(myURL)
soup = bs4.BeautifulSoup(page.content, 'html.parser')
stockData = soup.find_all(id_="contentjqxgrid")
print(stockData)

url = 'https://www.macrotrends.net/stocks/charts/MSFT/microsoft/revenue'
request_headers = {
    'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like gecko) Chrome/85.0.4183.102 Safari/537.36')
                    }
html = requests.get(url)
tables = pd.read_html(html.text)
tables[1]
##############################################################################
# Kind 크롤링
import requests

url = 'https://pjt3591oo.github.io/'

res = requests.get(url)

res.text
res.content

res = requests.get(url, params={"key1": "value1", "key2": "value2"})
res.url

url = 'http://www.example.com'
res = requests.post(url, data={"key1": "value1", "key2": "value2"})
res.url

import json
url = 'http://www.example.com'
res = requests.post(url, data=json.dumps({"key1": "value1", "key2": "value2"}))
res.url

url = 'https://kind.krx.co.kr/disclosure/details.do'
data = {"method": "searchDetailsSub",
        "currentPageSize": "15",
        "pageIndex": "1",
        "orderMode": "1",
        "orderStat": "D",
        "forward":"details_sub",
        "disclosureType01":"",
        "disclosureType02":"",
        "disclosureType03":"",
        "disclosureType04":"",
        "disclosureType05":"",
        "disclosureType06":"",
        "disclosureType07":"",
        "disclosureType08":"",
        "disclosureType09":"",
        "disclosureType10":"",
        "disclosureType11":"",
        "disclosureType13":"",
        "disclosureType14":"",
        "disclosureType20":"",
        "pDisclosureType01":"",
        "pDisclosureType02":"",
        "pDisclosureType03":"",
        "pDisclosureType04":"",
        "pDisclosureType05":"",
        "pDisclosureType06":"",
        "pDisclosureType07":"",
        "pDisclosureType08":"",
        "pDisclosureType09":"",
        "pDisclosureType10":"",
        "pDisclosureType11":"",
        "pDisclosureType13":"",
        "pDisclosureType14":"",
        "pDisclosureType20":"",
        "searchCodeType":"",
        "repIsuSrtCd":"",
        "allRepIsuSrtCd":"",
        "oldSearchCorpName":"",
        "disclosureType":"",
        "disTypevalue":"",
        "reportNm":"주권매매거래정지기간변경",
        "reportCd":"",
        "searchCorpName":"",
        "business":"",
        "marketType":"",
        "settlementMonth":"",
        "securities":"",
        "submitOblgNm":"",
        "enterprise":"",
        "fromDate":"2020-08-29",
        "toDate":"2020-09-29",
        "reportNmTemp":"주권매매거래정지기간변경",
        "reportNmPop":"",
        "bfrDsclsType":"on"
}
res = requests.get(url, data)
res.text

import bs4
soup = bs4.BeautifulSoup(res.text, 'lxml')
soup
print(soup.prettify())

soup.body
soup.a

soup.find_all('a')
soup.find_all(id='companysum')
soup.find_all(class_='vmiddle legend')
soup.find_all('img', class_='vmiddle legend')
soup.find_all('tr')
soup.find_all('a', href='#page_link_2')
soup.find_all('a')[2]

soup.select('a') # 모든 a요소
soup.select('#companysum') # 아이디가 companysum인 모든 요소

soup.select('body a#companysum')

soup.select('a')
soup.find_all('a', class_='btn ico chart-00')
soup.find_all('a', class_='btn ico chart-01')

for tag in soup.find_all('a', class_='btn ico chart-00'):
    tag.extract()
    
for tag in soup.find_all('a', class_='btn ico chart-01'):
    tag.extract()
    
soup.find_all('a')
soup.find_all('a', href='#page_link_2')

title = soup.find_all(id='companysum')
title[0].get_text().strip()

soup.find('tbody')

container = soup.find('tbody')
container
contents = []
for p in container.find_all(id='companysum'):
    contents.append(p.get_text().strip())

contents

data_100 = {"method": "searchDetailsSub",
        "currentPageSize": "100",
        "pageIndex": "1",
        "orderMode": "1",
        "orderStat": "D",
        "forward":"details_sub",
        "disclosureType01":"",
        "disclosureType02":"",
        "disclosureType03":"",
        "disclosureType04":"",
        "disclosureType05":"",
        "disclosureType06":"",
        "disclosureType07":"",
        "disclosureType08":"",
        "disclosureType09":"",
        "disclosureType10":"",
        "disclosureType11":"",
        "disclosureType13":"",
        "disclosureType14":"",
        "disclosureType20":"",
        "pDisclosureType01":"",
        "pDisclosureType02":"",
        "pDisclosureType03":"",
        "pDisclosureType04":"",
        "pDisclosureType05":"",
        "pDisclosureType06":"",
        "pDisclosureType07":"",
        "pDisclosureType08":"",
        "pDisclosureType09":"",
        "pDisclosureType10":"",
        "pDisclosureType11":"",
        "pDisclosureType13":"",
        "pDisclosureType14":"",
        "pDisclosureType20":"",
        "searchCodeType":"",
        "repIsuSrtCd":"",
        "allRepIsuSrtCd":"",
        "oldSearchCorpName":"",
        "disclosureType":"",
        "disTypevalue":"",
        "reportNm":"주권매매거래정지기간변경",
        "reportCd":"",
        "searchCorpName":"",
        "business":"",
        "marketType":"",
        "settlementMonth":"",
        "securities":"",
        "submitOblgNm":"",
        "enterprise":"",
        "fromDate":"2020-08-29",
        "toDate":"2020-09-29",
        "reportNmTemp":"주권매매거래정지기간변경",
        "reportNmPop":"",
        "bfrDsclsType":"on"
}
res = requests.get(url, data_100)
res.text

soup = bs4.BeautifulSoup(res.text, 'lxml')
soup

container = soup.find('tbody')
container
contents = []
for p in container.find_all(id='companysum'):
    contents.append(p.get_text().strip())

contents

##############################################################################
url = 'https://finance.yahoo.com/quote/AAPL/financials?p=AAPL'
html = requests.get(url)

soup = bs4.BeautifulSoup(html.text, 'lxml')
soup.find_all(class_ = "D(tbrg)")
//*[@id="Col1-1-Financials-Proxy"]/section/div[4]/div[1]/div[1]/div[1]/div/div[3]/span
//*[@id="Col1-1-Financials-Proxy"]/section/div[4]/div[1]/div[1]/div[1]/div/div[4]/span


soup.find_all(id = "Col1-1-Financials-Proxy")[0].find_all('span')[16:-1]

a = soup.find_all(id = "Col1-1-Financials-Proxy")[0].find_all('span')[16:-1]
b = pd.DataFrame(np.zeros(len(a)).reshape(int(len(a)/4),4))


temp = []
for i in range(len(a)):
    temp.append(a[i].text)
    
pd.DataFrame(np.array(temp).reshape(int(len(temp)/4), 5))

pd.read_html(soup.find_all(id = "Col1-1-Financials-Proxy")[0].find_all('span'))

ls= [] # Create empty list
for l in soup.find_all('div'): 
  #Find all data structure that is ‘div’
  ls.append(l.string) # add each element one by one to the list
 
ls = [e for e in ls if e not in ('Operating Expenses','Non-recurring Events')] # Exclude those columns
new_ls = list(filter(None,ls))
new_ls = new_ls[12:]
is_data = list(zip(*[iter(new_ls)]*6))
Income_st = pd.DataFrame(is_data[0:])

Income_st.columns = Income_st.iloc[0] # Name columns to first row of dataframe
Income_st = Income_st.iloc[1:,] # start to read 1st row
Income_st = Income_st.T # transpose dataframe
Income_st.columns = Income_st.iloc[0] #Name columns to first row of dataframe
Income_st.drop(Income_st.index[0],inplace=True) #Drop first index row
Income_st.index.name = '' # Remove the index name
Income_st.rename(index={'ttm': '12/31/2019'},inplace=True) #Rename ttm in index columns to end of the year
Income_st = Income_st[Income_st.columns[:-5]] # remove last 5 irrelevant columns


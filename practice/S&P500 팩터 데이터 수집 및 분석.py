from pack import *
from josh import *
from data_crawler import *

plt.rcParams["figure.figsize"] = (14,8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
plt.rcParams["axes.grid"] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["axes.formatter.limits"] = -10000, 10000
# matplotlib 컬러맵 생성
import matplotlib as mpl
import numpy as np

def make_colors(n, colormap=plt.cm.Spectral):
    return colormap(np.linspace(0.1, 1.0, n))

def make_explode(n):
    explodes = np.zeros(n)
    explodes[0] = 0.15
    return explodes

sp500 = fdr.StockListing('S&P500')
sp500.head(10)

len(sp500)

### 팩터 데이터 - finviz.com
# https://finviz.com/quote.ashx?t=AAPL
url = 'https://finviz.com/quote.ashx?t=AAPL'
html = requests.get(url)

request_headers = {
    'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like gecko) Chrome/68.0.3440.75 Safari/537.36')
                    }
html = requests.get(url, headers = request_headers)
tables = pd.read_html(html.text)
df = tables[6] # 페이지 구성 따라 [7]을 읽어야 하는 경우도 있음
df.columns = ['key', 'value'] * 6

# 컬럼을 행으로 만들기
df_list = [df.iloc[:, i*2: i*2+2] for i in range(6)]
df_factor = pd.concat(df_list, ignore_index=True)

df_factor.set_index('key', inplace=True)
df_factor.head(20)

'''
주요 팩터
Market Cap
Dividend yield
PER
PBR
ROE
Beta
'''
v = df_factor.value

marcap = v['Market Cap']
dividend = v['Dividend %']
per = v['P/E']
pbr = v['P/B']
beta = v['Beta']
roe = v['ROE']

marcap, dividend, per, pbr, beta, roe 

# 데이터 전처리
# 1. 단위(B, M, K) 수치값으로 변환
# 2. '%' 삭제
# 3. float 타입 변환
import re

def _conv_to_float(s):
    if s[-1] == '%':
        s = s.replace('%', '')
    if s[-1] in list('BMK'):
        powers = {'B': 10 ** 9, 'M': 10 ** 6, 'K': 10 ** 3, '': 1}
        m = re.search("([0-9\.]+)(M|B|K|)", s)
        if m:
            val, mag = m.group(1), m.group(2)
            return float(val) * powers[mag]
    try:
        result = float(s)
    except:
        result = None
    return result
marcap = _conv_to_float(marcap)
dividend = _conv_to_float(dividend)
per = _conv_to_float(per)
pbr = _conv_to_float(pbr)
beta = _conv_to_float(beta)
roe = _conv_to_float(roe)

marcap, dividend, per, pbr, beta, roe  

# 팩터 크롤링, 함수로 정리하기
import pandas as pd
import re

## 데이터 전처리 변환
def _conv_to_float(s):
    if s[-1] == '%':
        s = s.replace('%', '')
    if s[-1] in list('BMK'):
        powers = {'B': 10 ** 9, 'M': 10 ** 6, 'K': 10 ** 3, '': 1}
        m = re.search("([0-9\.]+)(M|B|K|)", s)
        if m:
            val, mag = m.group(1), m.group(2)
            return float(val) * powers[mag]
    try:
        result = float(s)
    except:
        result = None
    return result

def stock_factors(sym):
    url = 'https://finviz.com/quote.ashx?t=' + sym
    request_headers = {
    'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like gecko) Chrome/68.0.3440.75 Safari/537.36')
                    }
    html = requests.get(url, headers = request_headers)
    df = pd.read_html(html.text)[6]
    df.columns = ['key', 'value'] * 6

    ## 컬럼을 행으로 만들기
    df_list = [df.iloc[:, i*2: i*2+2] for i in range(6)]
    df_factor = pd.concat(df_list, ignore_index=True)
    df_factor.set_index('key', inplace=True)

    v = df_factor.value
    marcap = _conv_to_float(v['Market Cap'])
    dividend = _conv_to_float(v['Dividend %'])
    per = _conv_to_float(v['P/E'])
    pbr = _conv_to_float(v['P/B'])
    beta = _conv_to_float(v['Beta'])
    roe = _conv_to_float(v['ROE'])

    return {'MarCap':marcap, 'Dividend':dividend, 'PER':per, 'PBR':pbr, 'Beta':beta, 'ROE':roe}

stock_factors('AAPL')
stock_factors('NFLX')

### 전종목 팩터 데이터 수집과 저장(JSON)
# 디렉토리가 없으면 생성
import os

folder = "sp500_factors/"

if not os.path.isdir(folder):
    os.mkdir(folder)
import json
import pandas as pd

re_map_sym = {'BRKB': 'BRK-B', 'BR': 'BRK-A', 'BFB':'BF-B'}

for ix, row in sp500.iterrows():
    sym, name = row['Symbol'], row['Name']
    json_fn = folder + '%s.json' % (sym)
    if os.path.exists(json_fn):
        print('skip', json_fn)
        continue

    if sym in re_map_sym:
        sym = re_map_sym[sym]
    factors = stock_factors(sym)
    with open(json_fn, 'w') as f:
        json.dump(factors, f)
    print(sym, name)
    
### JSON 팩터 데이터 읽기
# JSON 팩터 데이터 읽기
for ix, row in sp500.iterrows():
    sym, name = row['Symbol'], row['Name']
    json_fn = folder + '%s.json' % (sym)

    with open(json_fn, 'r') as f:
        factors = json.load(f)
        
        for f in ['MarCap', 'Dividend', 'PER', 'PBR', 'Beta', 'ROE']:
            sp500.loc[ix,f] = factors[f]
            
### 섹터별 종목수
sector_count = sp500.groupby('Sector')['Symbol'].count().sort_values(ascending=False)
sector_count

# sector_count 섹터별 종목수
values = sector_count.values
labels = sector_count.index

n = len(labels)
plt.pie(values, labels=labels, colors=make_colors(n), explode=make_explode(n), autopct='%1.1f%%', shadow=True, startangle=135)
plt.axis('equal')
plt.show()

### 섹터별 시가총액
sector_marcap = sp500.groupby('Sector')['MarCap'].sum().sort_values(ascending=False)
sector_marcap

# sector_marcap 시가총액
values = sector_marcap.values
labels = sector_marcap.index

n = len(labels)

plt.pie(values, labels=labels, colors=make_colors(n), explode=make_explode(n), autopct='%1.1f%%', shadow=True, startangle=135)
plt.axis('equal')
plt.show()

_ = sector_marcap.plot(kind='bar', color='orange', alpha=0.7)

## 섹터별 팩터 분석
sp500.groupby('Sector').describe()['PER'].sort_values('mean', ascending=False)

sp500[sp500['Sector']=='Consumer Discretionary'].sort_values(by='PER', ascending=False)[:10]

sp500.groupby('Sector').describe()['PBR'].sort_values('mean', ascending=False)

sp500.groupby('Sector').describe()['ROE'].sort_values('mean', ascending=False)

sp500.groupby('Sector').describe()['Beta'].sort_values('mean', ascending=False)

sp500.groupby('Sector').describe()['Dividend'].sort_values('mean', ascending=False)
from pack import *
from josh import *
from data_crawler import *

### 한국거래소(KRX)의 전체 종목 가져오기

krx = fdr.StockListing('KRX')
krx

krx.groupby('Sector')

krx.groupby('Sector').count()

krx.groupby('Sector').count().sort_values('Symbol', ascending=False)[:30]

sectors = dict(list(krx.groupby('Sector')))

print('count:', len(sectors))
list(sectors.keys())[:10]

med_sec = sectors['의료용품 및 기타 의약 관련제품 제조업']

print('row count:', len(med_sec))
med_sec.head(10)

### 개별 종목의 가격 데이터 가져오기
med = pd.DataFrame()
for ix, row in med_sec.iterrows():
  code, name = row['Symbol'], row['Name']
  print(code, name)
  # 개별 종목의 가격을 가져옵니다
  df = fdr.DataReader(code, '2019-01-01', '2019-12-31')

  # 가격 데이터의 종가(Close)를 컬럼으로 추가합니다
  # (컬럼명은 종목명을 지정합니다)
  med[name] = df['Close']
  
med

med = med.dropna(axis=1)
med

# 수익률 계산
acc_rets = med / med.iloc[0] - 1.0
acc_rets

returns = acc_rets.iloc[-1]
returns

returns.sort_values(ascending=False)

returns.mean()

### 다양한 기간에 대한 수익률
df = med['2019-12-01':'2019-12-30'] # 특정 기간(12월 1달) 동안
acc_rets = df / df.iloc[0] - 1.0
acc_rets.iloc[-1]

# 2019-12-30 시점을 기준으로 과거 5일, 10일, 20일, 60일, 120일, 240일 각각 수익률을 구해봅니다.
the_day = datetime.datetime(2019, 12, 30)

for days in [5, 10, 20, 60, 120, 240]:
    start = the_day - datetime.timedelta(days)
    end = the_day
    print(start, '~', end)
  
the_day = datetime.datetime(2019, 12, 30)
row_dict = {}
for days in [5, 10, 20, 60, 120, 240]:
  start = the_day - datetime.timedelta(days)
  end = the_day

  df = med[start:end] # 특정 기간
  acc_rets = df / df.iloc[0] - 1.0
  row_dict[days] = acc_rets.iloc[-1] 

df_rets = pd.DataFrame(row_dict)
df_rets

# 섹터 전체 기간별 수익률
df_rets.mean()

# 엑셀로 저장
df_rets.to_excel('기간별수익률데이터(2019.12.30).xlsx', engine='openpyxl')
# 구글 colab 에서 실행한 경우 엑셀 파일을 다음과 같이 다운로드 합니다.

from google.colab import files
files.download('기간별수익률데이터(2019.12.30).xlsx')
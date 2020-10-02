import FinanceDataReader as fdr
import matplotlib.pyplot as plt
'''
한국
심볼	    거래소
KRX	    KRX 종목 전체 # KRX는 KOSPI,KOSDAQ,KONEX 모두 포함
KOSPI	KOSPI 종목
KOSDAQ	KOSDAQ 종목
KONEX	KONEX 종목

미국
심볼	    거래소
NASDAQ	나스닥 종목
NYSE	뉴욕 증권거래소 종목
AMEX	AMEX 종목
SP500	S&P 500 종목
'''
# 한국거래소 상장종목 전체
df_krx = fdr.StockListing('KRX')
df_krx.head()
len(df_krx)

# S&P 500 종목 전체
df_spx = fdr.StockListing('S&P500')
df_spx.head()
len(df_spx)

'''
가격 데이터 - 국내주식
단축 코드(6자리)를 사용

코스피 종목: 068270(셀트리온), 005380(현대차) 등
코스닥 종목: 215600(신라젠), 151910(나노스) 등
'''
# 신라젠, 2018년
df = fdr.DataReader('215600', '2018')
df.head(10)

# 셀트리온, 2017년~현재
df = fdr.DataReader('068270', '2017')
df['Close'].plot()

'''
가격 데이터 - 미국 주식
티커를 사용. 'AAPL'(애플), 'AMZN'(아마존), 'GOOG'(구글)
'''
# 애플(AAPL), 2018-01-01 ~ 2018-03-30
df = fdr.DataReader('AAPL', '2018-01-01', '2018-03-30')
df.tail()

# 애플(AAPL), 2017년
df = fdr.DataReader('AAPL', '2017')
df['Close'].plot()

# 아마존(AMZN), 2010~현재
df = fdr.DataReader('AMZN', '2010')
df['Close'].plot()

'''
한국 지수
심볼	    설명
KS11	KOSPI 지수
KQ11	KOSDAQ 지수
KS50	KOSPI 50 지수
KS100	KOSPI 100
KRX100	KRX 100
KS200   코스피 200

미국 지수
심볼	    설명
DJI	    다우존스 지수
IXIC	나스닥 지수
US500	S&P 500 지수
VIX	    S&P 500 VIX

국가별 주요 지수
심볼  	    설명
JP225	    닛케이 225 선물
STOXX50E	Euro Stoxx 50
CSI300	    CSI 300 (중국)
HSI	       항셍 (홍콩)
FTSE	   영국 FTSE
DAX	       독일 DAX 30
CAC	       프랑스 CAC 40
'''

# KS11 (KOSPI 지수), 2015년~현재
df = fdr.DataReader('KS11', '2015')
df['Close'].plot()

# 다우지수, 2015년~현재
df = fdr.DataReader('DJI', '2015')
df['Close'].plot()

'''
환율
심볼	    설명
USD/KRW	달러당 원화 환율
USD/EUR	달러당 유로화 환율
USD/JPY	달러당 엔화 환율
CNY/KRW	위엔화 원화 환율
EUR/USD	유로화 달러 환율
USD/JPY	달러 엔화 환율
JPY/KRW	엔화 원화 환율
AUD/USD	오스트레일리아 달러 환율
EUR/JPY	유로화 엔화 환율
USD/RUB	달러 루블화
'''
# 원달러 환율, 1995년~현재
df = fdr.DataReader('USD/KRW', '1995')
df['Close'].plot()

# 위엔화 환율, 1995년~현재
df = fdr.DataReader('CNY/KRW', '1995')
df['Close'].plot()

# fdr.EtfListing(country=['US', 'CN', 'HK', 'JP', 'UK', 'FR'])
fdr.EtfListing(country='KR')
fdr.EtfListing(country='US')

import pyfolio

df = fdr.DataReader('US500')
return_series = df['Close'].pct_change().fillna(0)
pyfolio.create_returns_tear_sheet(return_series)

import os
import pandas as pd
import numpy as np
import requests
import bs4
import json
import re
import time
import FinanceDataReader as fdr
import sys
import pickle
import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import stats
from scipy.stats import norm
from scipy.optimize import minimize
import empyrical as ep
import seaborn as sns
import yahoo_fin.stock_info as si
from selenium import webdriver

def get_KOR_ticker():
    
    def make_code_price(x):
        x = str(x)
        return '0' * (6-len(x)) + x
    
    def make_numeric(x):
        x = x.replace(',', '')
        x = pd.to_numeric(x)
        return x

    url="http://marketdata.krx.co.kr/mdi#document=040402"
    driver=webdriver.Chrome('chromedriver')
    driver.get(url)
    selector = driver.find_element_by_xpath('//*[@id="c9f0f895fb98ab9159f51fd0297e236d"]/button[4]')
    selector.click()

    time.sleep(15)
    
    path = r'C:\Users\Samsung\Downloads\data.csv'
    data = pd.read_csv(path)

    K = list()
    for i in range(len(data)):
        K.append((re.search('\w+우', data['종목명'][i]) == None) and \
             (re.search('\w+우B', data['종목명'][i]) == None) and \
                 (re.search('\w+스팩', data['종목명'][i]) == None) and \
                     (re.search('\w+\d호', data['종목명'][i]) == None) and \
                         (re.search('\w+\d', data['종목명'][i]) == None)) 
    data = data.iloc[K]
    data.index = range(len(data))
    data['종목코드'] = data['종목코드'].apply(make_code_price)
    data['시가총액'] = data['시가총액'].apply(make_numeric)
    data.to_csv(r'C:\Users\Samsung\Downloads\quant\Python\data\ticker.csv')
    return data

def get_KOR_fs():
    """
    Download Financial statements.
    ticker Downloaded from http://marketdata.krx.co.kr/mdi#document=040402

    Returns
    -------
    total_fs : pickle
        All Financial statements listed in KRX.

    """
    ticker = get_KOR_ticker()
    ticker = ticker[['종목코드', '종목명']]
    
    total_fs = {}
    for num, code in enumerate(ticker['종목코드']):
        try:
            print(num, code)
            time.sleep(1)
            try:
                fs_url = 'https://comp.fnguide.com/SVO2/asp/SVD_Finance.asp?pGB=1&cID=&MenuYn=Y&ReportGB=D&NewMenuID=103&stkGb=701&gicode=A' + code
                fs_page = requests.get(fs_url)
                fs_tables = pd.read_html(fs_page.text, displayed_only=False)
            
                temp_IS = fs_tables[0]
                for i in range(len(temp_IS.index)):
                    temp_IS['IFRS(연결)'][i] = temp_IS['IFRS(연결)'][i].replace('계산에 참여한 계정 펼치기','')
                temp_IS = temp_IS.set_index(temp_IS.columns[0])
                temp_IS = temp_IS[temp_IS.columns[:4]]
            
                temp_BS = fs_tables[2]
                for i in range(len(temp_BS.index)):
                    temp_BS['IFRS(연결)'][i] = temp_BS['IFRS(연결)'][i].replace('계산에 참여한 계정 펼치기','')
                temp_BS = temp_BS.set_index(temp_BS.columns[0])

                temp_CF = fs_tables[4]
                for i in range(len(temp_CF.index)):
                    temp_CF['IFRS(연결)'][i] = temp_CF['IFRS(연결)'][i].replace('계산에 참여한 계정 펼치기','')
                temp_CF = temp_CF.set_index(temp_CF.columns[0])
            
                fs_df = pd.concat([temp_IS, temp_BS, temp_CF])
                fs_df = fs_df.fillna(0)
        
            except requests.exceptions.Timeout:
                time.sleep(60)
                fs_url = 'https://comp.fnguide.com/SVO2/asp/SVD_Finance.asp?pGB=1&cID=&MenuYn=Y&ReportGB=D&NewMenuID=103&stkGb=701&gicode=A' + ticker
                fs_page = requests.get(fs_url)
                fs_tables = pd.read_html(fs_page.text, displayed_only=False)
            
                temp_IS = fs_tables[0]
                for i in range(len(temp_IS.index)):
                    temp_IS['IFRS(연결)'][i] = temp_IS['IFRS(연결)'][i].replace('계산에 참여한 계정 펼치기','')
                temp_IS = temp_IS.set_index(temp_IS.columns[0])
                temp_IS = temp_IS[temp_IS.columns[:4]]
            
                temp_BS = fs_tables[2]
                for i in range(len(temp_BS.index)):
                    temp_BS['IFRS(연결)'][i] = temp_BS['IFRS(연결)'][i].replace('계산에 참여한 계정 펼치기','')
                temp_BS = temp_BS.set_index(temp_BS.columns[0])

                temp_CF = fs_tables[4]
                for i in range(len(temp_CF.index)):
                    temp_CF['IFRS(연결)'][i] = temp_CF['IFRS(연결)'][i].replace('계산에 참여한 계정 펼치기','')
                temp_CF = temp_CF.set_index(temp_CF.columns[0])
            
                fs_df = pd.concat([temp_IS, temp_BS, temp_CF])
                fs_df = fs_df.fillna(0)
            total_fs[code] = fs_df
        except ValueError:
            continue
        except KeyError:
            continue
    with open(r'C:\Users\Samsung\Downloads\quant\Python\data\fs.pickle', 'wb') as f:
        pickle.dump(total_fs, f)
    return total_fs

def get_KOR_value():
    """
    Download Financial statements.
    ticker Downloaded from http://marketdata.krx.co.kr/mdi#document=040402

    Returns
    -------
    total_fs : pickle
        All Financial statements listed in KRX.

    """
    ticker = get_KOR_ticker()
    ticker = ticker[['종목코드', '종목명']]
    
    total_value = {}
    for num, code in enumerate(ticker['종목코드']):
        try:
            print(num, code)
            time.sleep(1)
            try:
                value_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp?pGB=1&gicode=A' + code
                value_page = requests.get(value_url)
                value_tables = pd.read_html(value_page.text)
            
                temp = value_tables[1].iloc[9:]
                for i in range(len(temp.index)):
                    temp[temp.columns[0]].iloc[i] = temp[temp.columns[0]].iloc[i].replace('계산에 참여한 계정 펼치기','')
                temp = temp.set_index(temp.columns[0])
                temp = temp.drop(index='Multiples')
                temp = temp.drop(index='FCF')
            
        
            except requests.exceptions.Timeout:
                time.sleep(60)
                value_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp?pGB=1&gicode=A' + code
                value_page = requests.get(value_url)
                value_tables = pd.read_html(value_page.text)
            
                temp = value_tables[1].iloc[9:]
                for i in range(len(temp.index)):
                    temp[temp.columns[0]].iloc[i] = temp[temp.columns[0]].iloc[i].replace('계산에 참여한 계정 펼치기','')
                temp = temp.set_index(temp.columns[0])
                temp = temp.drop(index='Multiples')
                temp = temp.drop(index='FCF')
            total_value[code] = temp
        except ValueError:
            continue
        except KeyError:
            continue
    with open(r'C:\Users\Samsung\Downloads\quant\Python\data\value.pickle', 'wb') as f:
        pickle.dump(total_value, f)
    return total_value

def get_KOR_ratio():
    """
    Download Financial statements.
    ticker Downloaded from http://marketdata.krx.co.kr/mdi#document=040402

    Returns
    -------
    total_fs : pickle
        All Financial statements listed in KRX.

    """
    ticker = get_KOR_ticker()
    ticker = ticker[['종목코드', '종목명']]
    
    total_ratio = {}
    for num, code in enumerate(ticker['종목코드']):
        try:
            print(num, code)
            time.sleep(1)
            try:
                ratio_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A' + code
                ratio_page = requests.get(ratio_url)
                ratio_tables = pd.read_html(ratio_page.text)
            
                temp = ratio_tables[0]
                for i in range(len(temp.index)):
                    temp[temp.columns[0]].iloc[i] = temp[temp.columns[0]].iloc[i].replace('계산에 참여한 계정 펼치기','')
                temp = temp.set_index(temp.columns[0])
                temp = temp.drop(index='안정성비율')
                temp = temp.drop(index='성장성비율')
                temp = temp.drop(index='수익성비율')
                temp = temp.drop(index='활동성비율')
            
        
            except requests.exceptions.Timeout:
                time.sleep(60)
                ratio_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A' + code
                ratio_page = requests.get(ratio_url)
                ratio_tables = pd.read_html(ratio_page.text)
            
                temp = ratio_tables[0]
                for i in range(len(temp.index)):
                    temp[temp.columns[0]].iloc[i] = temp[temp.columns[0]].iloc[i].replace('계산에 참여한 계정 펼치기','')
                temp = temp.set_index(temp.columns[0])
                temp = temp.drop(index='안정성비율')
                temp = temp.drop(index='성장성비율')
                temp = temp.drop(index='수익성비율')
                temp = temp.drop(index='활동성비율')
            total_ratio[code] = temp
        except ValueError:
            continue
        except KeyError:
            continue
    with open(r'C:\Users\Samsung\Downloads\quant\Python\data\ratio.pickle', 'wb') as f:
        pickle.dump(total_ratio, f)
    return total_ratio

def get_KOR_price():    
    """
    Download Korea stock prices.
    ticker Downloaded from http://marketdata.krx.co.kr/mdi#document=040402

    Returns
    -------
    pd.DataFrame
        All stock prices listed in KRX.

    """
    ticker = get_KOR_ticker()
    ticker = ticker[['종목코드', '종목명']]

    folder = "KOR_price/"

    if not os.path.isdir(r'C:\Users\Samsung\Downloads\quant\Python\data' + '\\' + folder):
        os.mkdir(r'C:\Users\Samsung\Downloads\quant\Python\data' + '\\' + folder)
        
    for num, code in enumerate(ticker['종목코드']):
        try:
            print(num, code)
            time.sleep(1)
            try:
                url = 'https://fchart.stock.naver.com/sise.nhn?requestType=0'
                price_url = url + '&symbol=' + code + '&timeframe=day' + '&count=3000'
                price_data = requests.get(price_url)
                price_data_bs = bs4.BeautifulSoup(price_data.text, 'lxml')
                item_list = price_data_bs.find_all('item')
                 
                date_list = []
                price_list = []
                for item in item_list:
                    temp_data = item['data']
                    datas = temp_data.split('|')
                    date_list.append(datas[0])
                    price_list.append(datas[4])
                    price_df = pd.DataFrame({code:price_list}, index=date_list)
                    price_df.index = pd.to_datetime(price_df.index)
                price_df.to_csv(r'C:\Users\a\Downloads\quant\Python\data\\' + folder + code + '.csv')
            except requests.exceptions.Timeout:
                time.sleep(60)
                url = 'https://fchart.stock.naver.com/sise.nhn?requestType=0'
                price_url = url + '&symbol=' + code + '&timeframe=day' + '&count=3000'
                price_data = requests.get(price_url)
                price_data_bs = bs4.BeautifulSoup(price_data.text, 'lxml')
                item_list = price_data_bs.find_all('item')
                 
                date_list = []
                price_list = []
                for item in item_list:
                    temp_data = item['data']
                    datas = temp_data.split('|')
                    date_list.append(datas[0])
                    price_list.append(datas[4])
                    price_df = pd.DataFrame({code:price_list}, index=date_list)
                    price_df.index = pd.to_datetime(price_df.index)
                price_df.to_csv(r'C:\Users\Samsung\Downloads\quant\Python\data\\' + folder + code + '.csv')
            if num == 0 :
                total_price = price_df
            else:
                total_price = pd.merge(total_price, price_df, how='outer', right_index=True, left_index=True)
        except ValueError:
            continue
        except KeyError:
            continue
    total_price.index = pd.to_datetime(total_price.index)
    total_price.to_csv(r'C:\Users\Samsung\Downloads\quant\Python\data\price.csv')
    return total_price

def get_KOR_index(ticker):
    """
    Download Korea Factor Index.
    Data Download from http://www.fnindex.co.kr

    Parameters
    ----------
    ticker : str
        Factor Index Ticker.

    Returns
    -------
    pd.DataFrame
        Factor index prices.

    Examples
    -------
    all_data = {}
    ticker = ['HTQ', 'HTV', 'HTM', 'HTL']
    name = ['Quality', 'Value', 'Momentum', 'Lowvol']

    for name, tic in zip(name, ticker):
        all_data[name] = get_KOR_index(tic)

    prices = pd.DataFrame({tic: data['Price'] for tic, data in all_data.items()})
    """
    def cleansing(x):
        x = x.replace('.', '')
        x = pd.Timestamp(x)
        return x
    
    url = 'http://www.fnindex.co.kr/api/realtime/line/FI00.WLT.' + ticker +'/10Y?_=1601542070322'
    html = requests.get(url)
    data = json.loads(html.text)
    data = pd.DataFrame(data['OUT_CURSOR'])
    
    columns = ['Date', 'Price']
    data.columns = columns
    data['Date'] = data['Date'].apply(cleansing)
    data = data.set_index('Date')
    
    return data

def get_US_ETF_ticker():
    """
    Download US ETF ticker lists.
    Data download from https://www.investing.com

    Returns
    -------
    ticker : pd.DataFrame
        US ETF ticker lists.

    """
    url = 'https://www.investing.com/etfs/usa-etfs'
    
    request_headers = {
    'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like gecko) Chrome/68.0.3440.75 Safari/537.36')
                    }
    html = requests.get(url, headers = request_headers)
    tables = pd.read_html(html.text)
    
    ticker = tables[0]
    ticker = ticker[['Name', 'Symbol', 'Vol.']]
    ticker.to_csv(r'C:\Users\Samsung\Downloads\quant\Python\data\US_ETF_ticker.csv')
    return ticker

def get_US_ETF_price(start_date, end_date):
    """
    Download US ETF prices.
    Data downloaded using get_US_ETF_ticker

    Parameters
    ----------
    start_date : string
        example : '2000-01-01'.
    end_date : string
        example : '2019-12-31'.

    Returns
    -------
    total_df : pd.DataFrame
        ETF prices from start_date to end_date.

    """
    path = r'C:\Users\Samsung\Downloads\quant\Python\data\US_ETF_ticker.csv'
    data = pd.read_csv(path)
    data = data[['Name', 'Symbol']]
    
    for num, code in enumerate(data['Symbol']):
        try:
            print(num, code)
            time.sleep(1)
            try:
                df = fdr.DataReader(code, start_date, end_date)
                df = df['Close']
                df = pd.DataFrame(df)
                df.columns = [code]
            except Exception as e:
                print(e, 'Error in Ticker', code)
                continue
            if num == 0:
                total_df = df
            else:
                total_df = pd.merge(total_df, df, how='outer', right_index=True, left_index=True)
        except ValueError:
            continue
        except KeyError:
            continue
    total_df.to_csv(r'C:\Users\Samsung\Downloads\quant\Python\data\US_ETF_price.csv')
    return total_df

def get_US_price(start_date, end_date):
    """
    Download US stocks prices.

    Parameters
    ----------
    start_date : string
        example : '2000-01-01'.
    end_date : string
        example : '2019-12-31'.

    Returns
    -------
    total_df : pd.DataFrame
        Stocks prices from start_date to end_date.

    """
    dow_list = si.tickers_dow()
    sp500_list = si.tickers_sp500()
    nasdaq_list = si.tickers_nasdaq()
    other_list = si.tickers_other()

    ticker = dow_list + sp500_list + nasdaq_list + other_list
    ticker = [ticker[i] for i in range(len(ticker)) if not ticker[i] in ticker[:i]]

    ticker = pd.DataFrame(ticker, columns=['Ticker'])
    
    for num, code in enumerate(ticker['Ticker']):
        try:
            print(num, code)
            time.sleep(1)
            try:
                df = fdr.DataReader(code, start_date, end_date)
                df = df['Close']
                df = pd.DataFrame(df)
                df.columns = [code]
            except Exception as e:
                print(e, 'Error in Ticker', code)
                continue
            if num == 0:
                total_df = df
            else:
                total_df = pd.merge(total_df, df, how='outer', right_index=True, left_index=True)
        except ValueError:
            continue
        except KeyError:
            continue
    total_df.to_csv(r'C:\Users\Samsung\Downloads\quant\Python\data\US_price.csv')
    return total_df
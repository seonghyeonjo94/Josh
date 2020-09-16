bs4q2015 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2015.xlsx', sheet_name='2015_4Q_BS')
print(1)
bs4q2016 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2016.xlsx', sheet_name='2016_4Q_BS')
print(2)
bs4q2017 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2017.xlsx', sheet_name='2017_4Q_BS')
print(3)
bs4q2018 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2018.xlsx', sheet_name='2018_4Q_BS')
print(4)
bs4q2019 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2019.xlsx', sheet_name='2019_4Q_BS')
print(5)

is4q2015 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2015.xlsx', sheet_name='2015_4Q_IS')
print(6)
is4q2016 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2016.xlsx', sheet_name='2016_4Q_IS')
print(7)
is4q2017 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2017.xlsx', sheet_name='2017_4Q_IS')
print(8)
is4q2018 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2018.xlsx', sheet_name='2018_4Q_IS')
print(9)
is4q2019 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2019.xlsx', sheet_name='2019_4Q_IS')
print(10)

cf4q2015 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2015.xlsx', sheet_name='2015_4Q_CF')
print(11)
cf4q2016 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2016.xlsx', sheet_name='2016_4Q_CF')
print(12)
cf4q2017 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2017.xlsx', sheet_name='2017_4Q_CF')
print(13)
cf4q2018 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2018.xlsx', sheet_name='2018_4Q_CF')
print(14)
cf4q2019 = pd.read_excel(r'C:\Users\a\Downloads\fs\data\2019.xlsx', sheet_name='2019_4Q_CF')
print(15)

def code_cleansing(x):
    x = x.replace('[', '')
    x = x.replace(']', '')
    return x

bs4q2015['종목코드'] = bs4q2015['종목코드'].apply(code_cleansing)
bs4q2016['종목코드'] = bs4q2016['종목코드'].apply(code_cleansing)
bs4q2017['종목코드'] = bs4q2017['종목코드'].apply(code_cleansing)
bs4q2018['종목코드'] = bs4q2018['종목코드'].apply(code_cleansing)
bs4q2019['종목코드'] = bs4q2019['종목코드'].apply(code_cleansing)

bs4q2015['항목명'] = pd.DataFrame(bs4q2015['항목명'].apply(str.lstrip))
bs4q2016['항목명'] = pd.DataFrame(bs4q2016['항목명'].apply(str.lstrip))
bs4q2017['항목명'] = pd.DataFrame(bs4q2017['항목명'].apply(str.lstrip))
bs4q2018['항목명'] = pd.DataFrame(bs4q2018['항목명'].apply(str.lstrip))
bs4q2019['항목명'] = pd.DataFrame(bs4q2019['항목명'].apply(str.lstrip))

bs4q2015 = bs4q2015.set_index(['종목코드', '항목명'])
bs4q2016 = bs4q2016.set_index(['종목코드', '항목명'])
bs4q2017 = bs4q2017.set_index(['종목코드', '항목명'])
bs4q2018 = bs4q2018.set_index(['종목코드', '항목명'])
bs4q2019 = bs4q2019.set_index(['종목코드', '항목명'])

bs = pd.merge(bs4q2015, bs4q2016['4Q2016'], how='outer',on=['종목코드', '항목명'])
bs = pd.merge(bs, bs4q2017['4Q2017'], how='outer',on=['종목코드', '항목명'])
bs = pd.merge(bs, bs4q2018['4Q2018'], how='outer',on=['종목코드', '항목명'])
bs = pd.merge(bs, bs4q2019['4Q2019'], how='outer',on=['종목코드', '항목명'])

bs = bs[['재무제표종류', '회사명', '시장구분', '업종', '업종명', '결산월', 
         '결산기준일', '보고서종류', '통화', '항목코드',
         '4Q2013', '4Q2014', '4Q2015', '4Q2016', '4Q2017',
         '4Q2018', '4Q2019']]

is4q2015['종목코드'] = is4q2015['종목코드'].apply(code_cleansing)
is4q2016['종목코드'] = is4q2016['종목코드'].apply(code_cleansing)
is4q2017['종목코드'] = is4q2017['종목코드'].apply(code_cleansing)
is4q2018['종목코드'] = is4q2018['종목코드'].apply(code_cleansing)
is4q2019['종목코드'] = is4q2019['종목코드'].apply(code_cleansing)

is4q2015 = is4q2015.set_index(['종목코드', '항목명'])
is4q2016 = is4q2016.set_index(['종목코드', '항목명'])
is4q2017 = is4q2017.set_index(['종목코드', '항목명'])
is4q2018 = is4q2018.set_index(['종목코드', '항목명'])
is4q2019 = is4q2019.set_index(['종목코드', '항목명'])

pl = pd.merge(is4q2015, is4q2016['4Q2016'], how='outer',on=['종목코드', '항목명'])
pl = pd.merge(pl, is4q2017['4Q2017'], how='outer',on=['종목코드', '항목명'])
pl = pd.merge(pl, is4q2018['4Q2018'], how='outer',on=['종목코드', '항목명'])
pl = pd.merge(pl, is4q2019['4Q2019'], how='outer',on=['종목코드', '항목명'])

pl = pl[['재무제표종류', '회사명', '시장구분', '업종', '업종명', '결산월', 
         '결산기준일', '보고서종류', '통화', '항목코드',
         '4Q2013', '4Q2014', '4Q2015', '4Q2016', '4Q2017',
         '4Q2018', '4Q2019']]

cf4q2015['종목코드'] = cf4q2015['종목코드'].apply(code_cleansing)
cf4q2016['종목코드'] = cf4q2016['종목코드'].apply(code_cleansing)
cf4q2017['종목코드'] = cf4q2017['종목코드'].apply(code_cleansing)
cf4q2018['종목코드'] = cf4q2018['종목코드'].apply(code_cleansing)
cf4q2019['종목코드'] = cf4q2019['종목코드'].apply(code_cleansing)

cf4q2015['항목명'] = pd.DataFrame(cf4q2015['항목명'].apply(str.lstrip))
cf4q2016['항목명'] = pd.DataFrame(cf4q2016['항목명'].apply(str.lstrip))
cf4q2017['항목명'] = pd.DataFrame(cf4q2017['항목명'].apply(str.lstrip))
cf4q2018['항목명'] = pd.DataFrame(cf4q2018['항목명'].apply(str.lstrip))
cf4q2019['항목명'] = pd.DataFrame(cf4q2019['항목명'].apply(str.lstrip))

cf4q2015 = cf4q2015.set_index(['종목코드', '항목명'])
cf4q2016 = cf4q2016.set_index(['종목코드', '항목명'])
cf4q2017 = cf4q2017.set_index(['종목코드', '항목명'])
cf4q2018 = cf4q2018.set_index(['종목코드', '항목명'])
cf4q2019 = cf4q2019.set_index(['종목코드', '항목명'])

cf = pd.merge(cf4q2015, cf4q2016['4Q2016'], how='outer',on=['종목코드', '항목명'])
cf = pd.merge(cf, cf4q2017['4Q2017'], how='outer',on=['종목코드', '항목명'])
cf = pd.merge(cf, cf4q2018['4Q2018'], how='outer',on=['종목코드', '항목명'])
cf = pd.merge(cf, cf4q2019['4Q2019'], how='outer',on=['종목코드', '항목명'])

cf = cf[['재무제표종류', '회사명', '시장구분', '업종', '업종명', '결산월', 
         '결산기준일', '보고서종류', '통화', '항목코드',
         '4Q2013', '4Q2014', '4Q2015', '4Q2016', '4Q2017',
         '4Q2018', '4Q2019']]


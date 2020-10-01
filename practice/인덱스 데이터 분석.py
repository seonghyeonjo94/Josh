from pack import *
from josh import *
from data_crawler import *

'''
퀄리티: HTQ
밸류: HTV
모멘텀: HTM
로우볼: HTL
'''

all_data = {}
ticker = ['HTQ', 'HTV', 'HTM', 'HTL']
name = ['Quality', 'Value', 'Momentum', 'Lowvol']

for name, tic in zip(name, ticker):
    all_data[name] = get_KOR_index(tic)

prices = pd.DataFrame({tic: data['Price'] for tic, data in all_data.items()})
rets = prices.pct_change()

prices.plot()
ReturnCumulative(rets['2017':]).plot()


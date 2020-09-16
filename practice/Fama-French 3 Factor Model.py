####### Data Source #######
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
import pandas as pd
import FinanceDataReader as fdr
import statsmodels.formula.api as smf

three_factor = pd.read_csv(r'C:\Users\USER\Downloads\F-F_Research_Data_Factors_daily_CSV\F-F_Research_Data_Factors_daily.CSV', skiprows=3)
mom_factor = pd.read_csv(r'C:\Users\USER\Downloads\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor_daily.CSV', skiprows=13)
three_factor = three_factor[:-1]
mom_factor = mom_factor[:-1]

def transpose_to_date(x):
    x = pd.Timestamp(x)
    return x

three_factor['Unnamed: 0'] = three_factor['Unnamed: 0'].apply(transpose_to_date)
mom_factor['Unnamed: 0'] = mom_factor['Unnamed: 0'].apply(transpose_to_date)

three_factor.index = three_factor['Unnamed: 0']
mom_factor.index = mom_factor['Unnamed: 0']

three_factor = three_factor.drop('Unnamed: 0', axis=1)
mom_factor = mom_factor.drop('Unnamed: 0', axis=1)
three_factor.columns = ['MRP', 'SMB', 'HML', 'Rf']
mom_factor.columns = ['Mom']

three_factor = three_factor.loc[mom_factor.index[0]:]
factor_data = pd.merge(three_factor, mom_factor, how='outer', right_index=True, left_index=True)

start_date = '2016-01-01'
end_date = '2020-04-30'

y = fdr.DataReader('spy', start_date, end_date)
y = pd.DataFrame(y['Close']).pct_change().fillna(0)
y.columns = ['y']
y = y * 100

factor_data = factor_data.loc[start_date:]

ff_data = factor_data.join(y)

ff_model = smf.ols(formula='y ~ MRP + SMB + HML + Mom', data=ff_data).fit()
ff_model.summary()

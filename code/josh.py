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

def apply_yearly(returns, Type = 'A'):
    """
    Transpose daily returns to monthly returns or Annual returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.
    Type : string, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'M', 'A'. 
        - The default is 'A'.

    Returns
    -------
    Annual returns or Monthly returns.

    """    
    
    if (Type == 'A'):
        s = pd.Series(np.arange(returns.shape[0]), index=returns.index)
        ep = s.resample("A").max()
        temp = pd.DataFrame(data = np.zeros(shape = (ep.shape[0], returns.shape[1])), index = ep.index.year, columns = returns.columns)
    
    if (Type == 'M'):
        s = pd.Series(np.arange(returns.shape[0]), index=returns.index)
        ep = s.resample("M").max()
        temp = pd.DataFrame(data = np.zeros(shape = (ep.shape[0], returns.shape[1])), index = ep.index, columns = returns.columns)

    for i in range(0 , len(ep)) :
        if (i == 0) :
            sub_ret = returns.iloc[ 0 : ep[i] + 1, :]
        else :
            sub_ret = returns.iloc[ ep[i-1]+1 : ep[i] + 1, :]
        temp.iloc[i, ] = (1 + sub_ret).prod() - 1
    
    return(temp)

def drawdown(returns):
    """
    Calculate Drawdown.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.

    Returns
    -------
    Drawdown.

    """    
    
    dd = pd.DataFrame(data = np.zeros(shape = (returns.shape[0], returns.shape[1])), index = returns.index, columns = returns.columns)
    returns[np.isnan(returns)] = 0
    
    for j in range(0, returns.shape[1]):
        
        if (returns.iloc[0, j] > 0) :
            dd.iloc[0, j] = 0
        else :
            dd.iloc[0, j] = returns.iloc[0, j]
            
        for i in range(1 , len(returns)):
            temp_dd = (1+dd.iloc[i-1, j]) * (1+returns.iloc[i, j]) - 1
            if (temp_dd > 0) :
                dd.iloc[i, j] = 0
            else:
                dd.iloc[i, j] = temp_dd
    
    return(dd)
    
def ReturnCumulative(returns):
    """
    Calculate Cumulative Returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.

    Returns
    -------
    Cumulative Returns.

    """
    
    returns[np.isnan(returns)] = 0
    
    temp = (1+returns).cumprod()-1
    print("Total Return: ", round(temp.iloc[-1, :], 4)) 
    return(temp)

def ReturnPortfolio(returns, weights):
    """
    Calculate weighted returns for a portfolio of assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.
    weights : pd.DataFrame
        Asset weights, as decimal percentages, treated as beginning of period weights.

    Returns
    -------
    returns : pd.DataFrame
        Daily weighted returns of a portfolio assets
    bop_weights : pd.DataFrame
        Beginning of Period (BOP) Weight for each asset.
    eop_weights : pd.DataFrame
        End of Period (BOP) Weight for each asset.

    """
    
    if returns.isnull().values.any() :
        print("NA's detected: filling NA's with zeros")
        returns[np.isnan(returns)] = 0

    if returns.shape[1] != weights.shape[1] :
        print("Columns of Return and Weight is not same")        ## Check The Column Dimension
               
    if returns.index[-1] < weights.index[0] + pd.DateOffset(days=1) :
        print("Last date in series occurs before beginning of first rebalancing period")
           
    if returns.index[0] < weights.index[0] :
        returns = returns.loc[returns.index > weights.index[0] + pd.DateOffset(days=1)]   ## Subset the Return object if the first rebalance date is after the first date 
     
    bop_value = pd.DataFrame(data = np.zeros(shape = (returns.shape[0], returns.shape[1])), index = returns.index, columns = returns.columns)
    eop_value = pd.DataFrame(data = np.zeros(shape = (returns.shape[0], returns.shape[1])), index = returns.index, columns = returns.columns)
    bop_weights = pd.DataFrame(data = np.zeros(shape = (returns.shape[0], returns.shape[1])), index = returns.index, columns = returns.columns)
    eop_weights = pd.DataFrame(data = np.zeros(shape = (returns.shape[0], returns.shape[1])), index = returns.index, columns = returns.columns)
    
    bop_value_total = pd.DataFrame(data = np.zeros(shape = returns.shape[0]), index = returns.index)
    eop_value_total = pd.DataFrame(data = np.zeros(shape = returns.shape[0]), index = returns.index)
    ret = pd.DataFrame(data = np.zeros(shape = returns.shape[0]), index = returns.index)
                       
    end_value = 1   # The end_value is the end of period total value from the prior period
    
    k = 0
    
    for i in range(0 , len(weights) -1 ) :
        fm = weights.index[i] + pd.DateOffset(days=1)
        to = weights.index[i + 1]            
        sub_ret = returns.loc[fm : to, ]

        jj = 0
        
        for j in range(0 , len(sub_ret) ) :
            if jj == 0 :
                bop_value.iloc[k, :] = end_value * weights.iloc[i, :]
            else :
                bop_value.iloc[k, :] = eop_value.iloc[k-1, :]
            
            bop_value_total.iloc[k] = bop_value.iloc[k, :].sum()
                        
            # Compute end of period values
            eop_value.iloc[k, :] = (1 + sub_ret.iloc[j, :]) * bop_value.iloc[k, :]
            eop_value_total.iloc[k] = eop_value.iloc[k, :].sum()
            
            # Compute portfolio returns
            ret.iloc[k] = eop_value_total.iloc[k] / end_value - 1
            end_value = float(eop_value_total.iloc[k])
            
            # Compute BOP and EOP weights
            bop_weights.iloc[k, :] = bop_value.iloc[k, :] / float(bop_value_total.iloc[k])
            eop_weights.iloc[k, :] = eop_value.iloc[k, :] / float(eop_value_total.iloc[k])
    
            jj += 1
            k += 1
    
    result = {'ret' : ret, 'bop_weights' : bop_weights, 'eop_weights' : eop_weights}
    return(result)
    
def ReturnStats(returns):
    """
    Return Statistics of Portfolio Return & Risk.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.

    Returns
    -------
    stats : pd.DataFrame
        Annual Arithmetic mean, Geometric mean, Standard deviation, Sharpe Ratio, Win Ratio, Max Drawdown.

    """
    
    temp = ep.aggregate_returns(returns, 'yearly')
    temp_ret = temp
    temp_std = temp
    
    dd = pd.DataFrame(drawdown(returns))
        
    Ann_ret_Arith = temp_ret.sum() / len(temp_ret)
    Ann_ret_CAGR = (1+temp_ret).prod() ** (1 / len(temp_ret)) - 1
    Ann_std = temp_std.std()
    Ann_Sharpe = (Ann_ret_CAGR-0.05) / Ann_std
    Win_Ratio = (returns > 0).sum() / ((returns > 0 ).sum() + (returns < 0).sum())
    MDD = dd.min()
    
    stats = pd.DataFrame([Ann_ret_Arith, Ann_ret_CAGR, Ann_std, Ann_Sharpe, Win_Ratio, MDD], index=['Ann_ret (Arith)', 'Ann_ret (CAGR)', 'Ann_std', 'Ann_sharpe', 'Win_Ratio', 'MDD'])
    stats.columns = returns.columns
    stats = round(stats, 4)
    return stats

def wt_normal(MarketCap, MaxWeight = 0.3, Type = 'VW'):
    for i in range(len(MarketCap)):
        MarketCap.iloc[i] = MarketCap.iloc[i].replace(',', '')
    MarketCap = pd.to_numeric(MarketCap)
    MarketCap = MarketCap / MarketCap.sum()
    
    if (Type == 'VW'):
        wt = MarketCap / MarketCap.sum()
        while wt.max() > MaxWeight:
            wt[wt > MaxWeight] = MaxWeight
            wt = wt / wt.sum()
            
    if (Type == 'EW'):
        wt = np.repeat((1/len(MarketCap)), len(MarketCap))
    
    return(wt)
    
def RiskParity(covmat) :
    
    def RiskParity_objective(x) :
    
        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        mrc = 1/sigma * (covmat @ x)
        rc = x * mrc
        a = np.array(rc).reshape(len(rc),1)
        risk_diffs = a - a.T
        sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
        return (sum_risk_diffs_squared)

    def weight_sum_constraint(x) :
        return (x.sum() - 1.0 )


    def weight_longonly(x) :
        return (x)
    
    x0 = np.repeat(1/covmat.shape[1], covmat.shape[1]) 
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                  {'type': 'ineq', 'fun': weight_longonly})
    options = {'ftol': 1e-20, 'maxiter': 800}
    
    result = minimize(fun = RiskParity_objective,
                      x0 = x0,
                      method = 'SLSQP',
                      constraints = constraints,
                      options = options)
    return(result.x)
    
def MinVol(covmat, lb = 0, ub = 1) :
    
    def weight_sum_constraint(x) :
        return(x.sum() - 1.0 )
    
    def MinVol_objective(x) :
        
        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        return (sigma)
    
    x0 = np.repeat(1/covmat.shape[1], covmat.shape[1]) 
    lbound  = np.repeat(lb, covmat.shape[1])
    ubound  = np.repeat(ub, covmat.shape[1])
    bnds = tuple(zip(lbound, ubound))
    
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint})
    options = {'ftol': 1e-20, 'maxiter': 800}
    
    result = minimize(fun = MinVol_objective,
                      x0 = x0,
                      method = 'SLSQP',
                      constraints = constraints,
                      options = options,
                      bounds = bnds)
    return(result.x)
    
def RC(weight, covmat) :
    weight = np.array(weight)
    variance = weight.T @ covmat @ weight
    sigma = variance ** 0.5
    mrc = 1/sigma * (covmat @ weight)
    rc = weight * mrc
    rc = rc / rc.sum()
    return (rc)

def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.
    
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    temp = returns[returns.columns[0]]
    monthly_ret_table = ep.aggregate_returns(temp, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    sns.heatmap(
        monthly_ret_table.fillna(0) *
        100.0,
        annot=True,
        annot_kws={"size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn,
        ax=ax, **kwargs)
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title("Monthly returns (%)")
    return ax



def plot_annual_returns(returns, ax=None, **kwargs):
    """
    Plots a bar graph of returns by year.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    plt.style.use("seaborn")
    
    if ax is None:
        ax = plt.gca()

    ann_ret_df = pd.DataFrame(
        ep.aggregate_returns(
            returns,
            'yearly'))

    ax.axvline(
        100 *
        ann_ret_df.values.mean(),
        color='steelblue',
        linestyle='--',
        lw=4,
        alpha=0.7)
    (100 * ann_ret_df.sort_index(ascending=False)
     ).plot(ax=ax, kind='barh', alpha=0.70, **kwargs)
    ax.axvline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylabel('Year')
    ax.set_xlabel('Returns')
    ax.set_title("Annual returns")
    ax.legend(['Mean'], frameon=True, framealpha=0.5)
    return ax



def PerformanceAnalysis(returns, benchmark=None):
    """
    Plots a Cumulative returns and Drawdown.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns of the strategy, noncumulative.
    benchmark : pd.Series, optional
        Daily returns of the Benchmark, noncumulative. 
        - The default is None.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """
    
    plt.style.use("seaborn")
    
    if benchmark is None:        
        port_cumret = ReturnCumulative(returns)
        port_cumret.columns = ['Backtest']
        ax = port_cumret.plot(label = 'Cumulative Return', color = 'forestgreen', alpha=0.7, lw=2, figsize=(8,4))
        ax.set_title('Cummulative Return')
        ax.legend(loc='best')
    else:
        port_cumret = ReturnCumulative(returns)
        bench_cumret = ReturnCumulative(pd.DataFrame(benchmark))
        port_cumret.columns = ['Backtest']
        port_cumret['Benchmark'] = bench_cumret
        ax = port_cumret.plot(label = 'Cumulative Return', color = ('forestgreen', 'gray'), alpha=0.7, lw=2, figsize=(8,4))
        ax.set_title('Cumulative Return')
        ax.legend(loc='best')
        ax.grid(True)
        
    port_dd = drawdown(returns)
    port_dd.columns = ['DrawDown']
    ax = port_dd.plot(label = 'DrawDown', kind='area', color = 'coral', alpha=0.7, figsize=(7,3))
    ax.set_title('DrawDown')
    ax.legend(loc='best')

def create_lagged_series(symbol, start_date, end_date):
    """
    Create lagged return series

    Parameters
    ----------
    symbol : string
        example : SPY.
    start_date : datetime.datetime or string
        example : '2000-01-01'.
    end_date : datetime.datetime or string
        example : '2020-05-15'.

    Returns
    -------
    tsret : pd.DataFrame
        lagged return.

    """

    # Obtain stock price
    adj_start_date = start_date - datetime.timedelta(days=365)
    ts = fdr.DataReader(symbol, adj_start_date, end_date)
 
    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag['Today'] = ts['Close']
    tslag['Volume_lag'] = ts['Volume'].shift()
    tslag['Volume'] = ts['Volume']

    # Create the shifted lag series of prior trading period close values
    for i in range(1,51):
        tslag['Lag%s' % str(i)] = ts['Close'].shift(i)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret['Today'] = tslag['Today'].pct_change()
    tsret['Volume'] = tslag['Volume']


    # Create the lagged percentage returns columns
    for i in range(1,51):
        tsret['Lag%s' % str(i)] = \
            tslag['Lag%s' % str(i)].pct_change()

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret['Direction'] = np.sign(tsret['Today'])
    tsret = tsret[tsret.index >= start_date]

    return tsret

def VaR(returns, P=1e6, c=0.99):
    """
    Variance-Covariance calculation of daily Value-at-Risk

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Portfolio daily returns.
    P : Float, optional
        a portfolio value. The default is 1e6.
    c : Float, optional
        Confidence level. The default is 0.99.

    Returns
    -------
    float
        DESCRIPTION.

    """
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = norm.ppf(1-c, mu, sigma)
    return P - P*(alpha + 1)

def RankIC(returns, rank):
    
    if returns.isnull().values.any() :
        print("NA's detected: filling NA's with zeros")
        returns[np.isnan(returns)] = 0
    
    if returns.shape[1] != rank.shape[1] :
        print("Columns of Return and Weight is not same")        ## Check The Column Dimension
               
    if returns.index[-1] < rank.index[0] + pd.DateOffset(days=1) :
        print("Last date in series occurs before beginning of first rebalancing period")
           
    if returns.index[0] < rank.index[0] :
        returns = returns.loc[rank.index[0]:]   ## Subset the Return object if the first rebalance date is after the first date 
    
    if returns.index[-1] > rank.index[-1]:
        rank = rank.resample('M').last()
        
    returns_monthly = apply_yearly(returns, 'M')
    
    rank_corr = list()

    for i in range(0 , len(rank) - 1 ) :
        fm = rank.index[i] + pd.DateOffset(days=1)
        to = rank.index[i + 1]            
        sub_ret = returns_monthly.loc[fm : to, ]
        sub_rank = rank.loc[fm : to,]
        
        temp_corr = pd.merge(sub_ret.T.rank(ascending=False), sub_rank.T, how='outer', right_index=True, left_index=True).corr(method='spearman').iloc[0][1]
        temp_corr = pd.DataFrame([temp_corr], index=sub_ret.index)
        rank_corr.append(temp_corr)
    
    rank_corr = pd.concat(rank_corr)
    rank_corr.columns = ['Rank Correlation']

    return(rank_corr)
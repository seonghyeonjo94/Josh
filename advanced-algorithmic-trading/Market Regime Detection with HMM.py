from pack import *
from josh import *
from data_crawler import *

from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

start_date = '2010-1-29'
end_date = datetime.datetime.today()
prices = fdr.DataReader('SPY', start = start_date, end = end_date)

prices["Returns"] = prices["Close"].pct_change()
prices.dropna(inplace=True)


def plot_in_sample_hidden_states(hmm_model, df):
    """
    Plot the adjusted closing prices masked by 
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    hidden_states = hmm_model.predict(rets_stack)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        hmm_model.n_components, 
        sharex=True, sharey=True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df.index[mask], 
            df["Close"][mask], 
            ".", linestyle='none', 
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()

rets_stack = np.column_stack([prices["Returns"]])

# Create the Gaussian Hidden markov Model and fit it
# to the SPY returns data, outputting a score
hmm_model = GaussianHMM(
    n_components=2, covariance_type='full', n_iter=1000
    ).fit(rets_stack)
print("Model Score:", hmm_model.score(rets_stack))
  
# Plot the in sample hidden states closing values
plot_in_sample_hidden_states(hmm_model, prices)


# Create hidden_state matrix
hidden_state = hmm_model.predict(rets_stack)

#--- Create Weight Matrix ---#

rets = pd.DataFrame(prices['Returns'])
wts = pd.DataFrame(hidden_state, index=rets.index, columns=rets.columns)
wts = wts.shift(1)
wts = wts.fillna(0)

K = wts == 1    
rets.iloc[K] = 0

#--- Calculate Cumulative Return ---#

cumret = ReturnCumulative(rets)

#--- Calculate Drawdown ---#

dd = drawdown(rets)

#--- Graph: Portfolio Return and Drawdown ---#

fig, axes = plt.subplots(2, 1)
cumret.plot(ax = axes[0], legend = None)
dd.plot(ax = axes[1], legend = None)

#--- Daily Return Frequency To Yearly Return Frequency ---#

yr_ret = apply_yearly(rets)
yr_ret.plot(kind = 'bar')

#--- Calculate Portfolio Stats ---#

ReturnStats(rets)

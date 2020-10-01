from pack import *
from josh import *
from data_crawler import *

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

data = data.iloc[36:]

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

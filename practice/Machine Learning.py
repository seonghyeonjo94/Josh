from pack import *
from josh import *
from data_crawler import *

start_date = datetime.datetime(2005, 1, 1)
end_date = datetime.datetime.today()
start_test = datetime.datetime(2016, 1, 1)

tickers = ['SPY', 'KS11', 'STOXX50', 'CSI300', 'USD/KRW']

all_data = {}
for ticker in tickers:
    all_data[ticker] = fdr.DataReader(ticker, start_date, end_date)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices['STOXX50'] = fdr.DataReader('STOXX50', start_date, end_date)['Open']
prices = prices.fillna(method = 'ffill')
rets = prices.pct_change(1)

rets.columns = ['SPY', 'KS11', 'STOXX50','CSI300', 'USDKRW']
rets['Direction'] = np.sign(rets['SPY'])
rets['Lag1'] = rets['SPY'].shift(1)
rets['Lag2'] = rets['SPY'].shift(2)  
rets.dropna(inplace=True)
  
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA, 
    QuadraticDiscriminantAnalysis as QDA
)
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler

X = rets[['KS11', 'STOXX50','CSI300', 'USDKRW', 'Lag1', 'Lag2']]
y = rets['Direction']

# Create training and test sets
X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

# Standardization
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Create the (parametrised) models
print('Hit Rates/Confusion Matrices:\n')
models = [    
    ('LR', LogisticRegression(solver='liblinear')), 
    ('LDA', LDA(solver='svd')), 
    ('QDA', QDA()),
    ('LSVC', LinearSVC(max_iter=10000)),
    ('RSVM', SVC(
        C=1000000.0, cache_size=200, class_weight=None,
        coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
        max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
    )#,
    #('RF', RandomForestClassifier(
    #    n_estimators=1000, criterion='gini', 
    #    max_depth=None, min_samples_split=2, 
    #    min_samples_leaf=1, max_features='auto', 
    #    bootstrap=True, oob_score=False, n_jobs=1, 
    #    random_state=None, verbose=0)
    #)
    ]

# Iterate through the models
for m in models:
    # Train each of the models on the training set
    m[1].fit(X_train_std, y_train)

    # Make an array of predictions on the test set
    pred = m[1].predict(X_test_std)

    # Output the hit-rate and the confusion matrix for each model
    print('%s:\n%0.3f' % (m[0], m[1].score(X_test_std, y_test)))
    print('%s\n' % confusion_matrix(pred, y_test))
    

###############################################################################
from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_

plt.bar(range(1, 7), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 7), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

from sklearn.linear_model import LogisticRegression
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr = lr.fit(X_train_pca, y_train)
pred = lr.predict(X_test_pca)

print('훈련 정확도:', lr.score(X_train_pca, y_train))
print('테스트 정확도:', lr.score(X_test_pca, y_test))


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv('data/PS_20174392719_1491204439457_log.csv', engine='c') # engine='c' makes it faster but prone to error

X = train.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = train['isFraud']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X_numerical = X.select_dtypes(numerics).drop('step', axis=1)

X_categorical = X[['type']]

X_categorical = pd.get_dummies(X_categorical)

X = pd.concat([
    np.log1p(X_numerical),
    X_categorical
], axis=1)

X = X.fillna(X.mean())


from sklearn.ensemble import GradientBoostingRegressor
cross_val_score(GradientBoostingRegressor(), X, y, cv=5, n_jobs=-1)

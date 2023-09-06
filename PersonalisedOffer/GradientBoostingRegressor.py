import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('data/train.csv')
X = train.drop(['Response', 'Id'], axis=1)
y = train['Response']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X_numerical = X.select_dtypes(numerics)

X_categorical = X.select_dtypes('object')

X_categorical = pd.get_dummies(X_categorical)

X = pd.concat([
    X_numerical,
    X_categorical
], axis=1)

X = X.fillna(X.mean())

cross_val_score(GradientBoostingRegressor(), X, y, cv=5, n_jobs=-1)

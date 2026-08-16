"""Classical machine-learning forecasting baselines."""
from __future__ import annotations

import numpy as np


def _flat(x):
    x=np.asarray(x); return x.reshape(len(x),-1)


class DirectMultiOutputRegressor:
    def __init__(self, estimator): self.estimator=estimator
    def fit(self,x,y): self.estimator.fit(_flat(x),np.asarray(y)); return self
    def predict(self,x): return self.estimator.predict(_flat(x))


def make_svr(c=10.0,epsilon=0.1,gamma="scale"):
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR
    return DirectMultiOutputRegressor(MultiOutputRegressor(SVR(C=c,epsilon=epsilon,gamma=gamma)))


def make_random_forest(n_estimators=300,random_state=0,n_jobs=-1):
    from sklearn.ensemble import RandomForestRegressor
    return DirectMultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,random_state=random_state,n_jobs=n_jobs))


def make_xgboost(random_state=0,**kwargs):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("XGBoost baseline requires optional dependency `xgboost`.") from exc
    from sklearn.multioutput import MultiOutputRegressor
    base=XGBRegressor(random_state=random_state,n_estimators=500,max_depth=6,learning_rate=0.03,subsample=.9,colsample_bytree=.9,**kwargs)
    return DirectMultiOutputRegressor(MultiOutputRegressor(base,n_jobs=-1))

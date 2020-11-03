from sklearn.base import BaseEstimator
from lightgbm import LGBMRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg =  LGBMRegressor(colsample_bytree= 0.8,
                                  min_child_weight= 0.01,
                                  min_data_in_leaf=5,
                                  num_leaves= 70,
                                  n_estimators=800,
                                  reg_alpha= 0,
                                  reg_lambda= 0.1,
                                  subsample= 0.5,
                                 learning_rate=0.15,
                                 max_bin=100)
 
    def fit(self, X, y):
        self.reg.fit(X, y)
        
 
    def predict(self, X):
        return self.reg.predict(X)
   
   
   
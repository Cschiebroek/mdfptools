# models/xgboost.py
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params if params else {"objective": "reg:squarederror"}
        self.model = xgb.XGBRegressor(**self.params)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

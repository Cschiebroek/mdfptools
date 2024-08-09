from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RandomForestModel:
    def __init__(self, max_depth=10, n_estimators=100, random_state=42):
        # Initialize the RandomForestRegressor with a specified max_depth
        self.model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state,max_features='sqrt')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

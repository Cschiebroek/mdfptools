from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class RidgeModel:
    def __init__(self, alpha=1.0):
        # Alpha is the regularization strength
        self.model = Ridge(alpha=alpha)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred
    
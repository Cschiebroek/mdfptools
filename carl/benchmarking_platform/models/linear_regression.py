# models/linear_regression.py
from .base_model import Model
from sklearn.linear_model import LinearRegression

class LinearRegressionModel(Model):
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        # Implement evaluation logic, e.g., RMSE, R^2, etc.
        pass

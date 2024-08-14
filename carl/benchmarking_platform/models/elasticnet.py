from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

class ElasticNetModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        # Alpha is the regularization strength
        # l1_ratio is the mixing parameter
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred
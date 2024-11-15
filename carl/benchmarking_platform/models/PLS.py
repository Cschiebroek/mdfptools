from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

class PLSModel:
    def __init__(self, n_components=2):
        self.model = PLSRegression(n_components=n_components)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
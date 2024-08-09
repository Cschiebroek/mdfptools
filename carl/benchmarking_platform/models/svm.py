# models/svm.py
from sklearn.svm import SVR
import joblib

class SVMModel:
    def __init__(self, C=170, epsilon=0.07, gamma=0.00011):
        self.model = SVR(C=C, epsilon=epsilon, gamma=gamma)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)

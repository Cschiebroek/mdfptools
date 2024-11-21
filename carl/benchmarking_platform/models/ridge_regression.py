from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred
    
  
    def get_alpha_from_cv(self, X_train, y_train, alphas = np.logspace(-3, 3, 13), scoring='neg_mean_squared_error'):
        """
        Perform cross-validation to determine the best alpha value.

        Parameters:
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target values for training.
        alphas (list): List of alpha values to try.
        scoring (str): Scoring metric for cross-validation. Default is 'neg_mean_squared_error'.

        Returns:
        float: Best alpha value found via cross-validation.
        """
        grid = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, scoring=scoring, cv=10)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        best_alpha = grid.best_params_['alpha']
        return best_alpha
    
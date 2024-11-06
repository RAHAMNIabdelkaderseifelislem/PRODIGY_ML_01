"""
House Price Prediction Model

This module defines a linear regression model for predicting house prices.
It provides methods for training, predicting, evaluating, saving, and loading the model.

Classes:
    HousePriceModel: A class for building and managing a house price prediction model.
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class HousePriceModel:
    """A class for building and managing a house price prediction model using linear regression."""

    def __init__(self):
        """Initializes the HousePriceModel instance with a linear regression model."""
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """
        Trains the linear regression model.

        Parameters:
            X_train (numpy.ndarray or pandas.DataFrame): The training features.
            y_train (numpy.ndarray or pandas.Series): The training target values.
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """
        Predicts target values using the trained model.

        Parameters:
            X (numpy.ndarray or pandas.DataFrame): The input features for prediction.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model performance on the test data.

        Parameters:
            X_test (numpy.ndarray or pandas.DataFrame): The test features.
            y_test (numpy.ndarray or pandas.Series): The true target values for the test set.

        Returns:
            dict: A dictionary containing Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).
        """
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {
            'MSE': mse,
            'RMSE': mse ** 0.5,
            'R2': r2
        }
    
    def save_model(self, filepath):
        """
        Saves the trained model to a file.

        Parameters:
            filepath (str): The path where the model will be saved.
        """
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load_model(filepath):
        """
        Loads a model from a file.

        Parameters:
            filepath (str): The path from which the model will be loaded.

        Returns:
            HousePriceModel: An instance of HousePriceModel with the loaded model.
        """
        model = HousePriceModel()
        model.model = joblib.load(filepath)
        return model

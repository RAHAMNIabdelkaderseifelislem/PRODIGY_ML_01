"""
Utility for training and evaluating a linear regression model for house price prediction
and launching a GUI for making predictions.

This script trains a linear regression model using preprocessed data from the
'house_data.csv' file, evaluates the model, saves it to a file, and launches a
GUI for making predictions.

"""

import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.model import HousePriceModel
from src.gui import PredictionGUI


def main():
    """
    Main entry point for the script.
    """
    # Load data
    df = pd.read_csv('data/house_data.csv')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Get feature columns (excluding target variable)
    feature_columns = [col for col in df.columns if col != 'SalePrice']
    
    # Preprocess data
    processed_df = preprocessor.preprocess(df)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(processed_df)
    
    # Train model
    model = HousePriceModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Save model
    model.save_model('models/linear_regression_model.pkl')
    
    # Launch GUI with all feature columns
    gui = PredictionGUI(model, preprocessor, feature_columns)
    gui.run()


if __name__ == "__main__":
    main()

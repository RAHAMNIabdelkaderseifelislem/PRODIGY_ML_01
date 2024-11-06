"""
Utility for training and deploying a house price prediction model.

This script loads data from a CSV file, preprocesses the data, trains a
linear regression model, evaluates the model, saves the model, and
launches a GUI for making predictions.

"""
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.model import HousePriceModel
from src.gui import PredictionGUI


def main():
    """
    Main entry point of the script.

    """
    # Load data
    df = pd.read_csv('data/house_data.csv')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
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
    
    # Launch GUI
    gui = PredictionGUI(model, preprocessor)
    gui.run()


if __name__ == "__main__":
    main()

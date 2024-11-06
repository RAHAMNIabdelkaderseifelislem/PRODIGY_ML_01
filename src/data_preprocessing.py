"""
Data Preprocessing Module

This module provides functionality for preprocessing house data for model training.
It handles feature selection, missing value imputation, and feature scaling.

Classes:
    DataPreprocessor: A class for preprocessing data, including scaling and splitting datasets.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """A class to preprocess house data for training a regression model."""

    def __init__(self):
        """Initializes the DataPreprocessor with a StandardScaler."""
        self.scaler = StandardScaler()
        
    def preprocess(self, df):
        """
        Preprocesses the input DataFrame by selecting features, handling missing values, and scaling.

        Parameters:
            df (pandas.DataFrame): The input DataFrame containing house data.

        Returns:
            pandas.DataFrame: The processed DataFrame with selected and scaled features.
        """
        # Select only the required features
        required_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
        
        # Create a copy of input data with only required features
        if 'SalePrice' in df.columns:
            processed_df = df[required_features + ['SalePrice']].copy()
        else:
            processed_df = df[required_features].copy()
        
        # Handle missing values if any
        processed_df = processed_df.fillna(0)
        
        # Scale the features
        processed_df[required_features] = self.scaler.fit_transform(processed_df[required_features])
        
        return processed_df
    
    def prepare_data(self, df):
        """
        Prepares the training and testing datasets from the input DataFrame.

        Parameters:
            df (pandas.DataFrame): The preprocessed DataFrame containing features and target.

        Returns:
            tuple: A tuple containing the split datasets: (X_train, X_test, y_train, y_test).
        """
        # Extract features and target variable
        X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
        y = df['SalePrice']
        
        # Split data into training and test sets
        return train_test_split(X, y, test_size=0.2, random_state=42)

"""
Data Preprocessing Utilities

This module contains utilities for preprocessing datasets.

Classes:
    DataPreprocessor: A class for preprocessing datasets.

"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    A class for preprocessing datasets.

    Attributes:
        label_encoders (dict): A dictionary of LabelEncoder instances
            used for encoding categorical variables.
        scaler (StandardScaler): A StandardScaler instance used for
            scaling numerical features.
    """

    def __init__(self):
        """
        Initializes the DataPreprocessor instance.
        """
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def preprocess(self, df):
        """
        Preprocesses a given dataset.

        This method handles missing values, encodes categorical variables
        and scales numerical features.

        Parameters:
            df (pandas.DataFrame): The dataset to be preprocessed.

        Returns:
            pandas.DataFrame: The preprocessed dataset.
        """
        # Handle missing values
        df = df.fillna(0)

        # Convert categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column].astype(str))

        # Scale numerical features
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])

        return df

    def prepare_data(self, df, target_column='SalePrice'):
        """
        Prepares a dataset for training.

        This method separates features and target, and splits the data into
        training and testing sets.

        Parameters:
            df (pandas.DataFrame): The dataset to be prepared.
            target_column (str): The column name of the target variable.

        Returns:
            tuple: A tuple containing the training features, testing features,
                training target and testing target.
        """
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

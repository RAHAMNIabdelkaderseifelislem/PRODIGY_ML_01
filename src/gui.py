"""
Utility for generating a GUI for making predictions with a trained model.

This module defines a class, PredictionGUI, which generates a GUI for making
predictions with a trained model. The GUI has input fields for the features of
the dataset, a predict button, and a label to display the result.

The class has methods to create the input fields, button, and label, to make
predictions when the button is clicked, and to run the GUI event loop.

"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib


class PredictionGUI:
    """
    A class for generating a GUI for making predictions with a trained model.

    This class generates a GUI with input fields for the features of the
    dataset, a predict button, and a label to display the result.

    Attributes:
        window (tkinter.Tk): The main window of the GUI.
        model (HousePriceModel): The trained model to make predictions with.
        preprocessor (DataPreprocessor): The preprocessor to preprocess the
            input data.
    """

    def __init__(self, model, preprocessor):
        """
        Initializes the PredictionGUI instance.

        Parameters:
            model (HousePriceModel): The trained model to make predictions with.
            preprocessor (DataPreprocessor): The preprocessor to preprocess the
                input data.
        """
        self.window = tk.Tk()
        self.window.title("House Price Prediction")
        self.window.geometry("600x400")
        
        self.model = model
        self.preprocessor = preprocessor
        
        self.create_widgets()
        
    def create_widgets(self):
        """
        Creates the input fields, button, and label for the GUI.
        """
        # Create input fields
        ttk.Label(self.window, text="Square Footage:").grid(row=0, column=0, padx=5, pady=5)
        self.sq_footage = ttk.Entry(self.window)
        self.sq_footage.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.window, text="Number of Bedrooms:").grid(row=1, column=0, padx=5, pady=5)
        self.bedrooms = ttk.Entry(self.window)
        self.bedrooms.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.window, text="Number of Bathrooms:").grid(row=2, column=0, padx=5, pady=5)
        self.bathrooms = ttk.Entry(self.window)
        self.bathrooms.grid(row=2, column=1, padx=5, pady=5)
        
        # Predict button
        ttk.Button(self.window, text="Predict Price", command=self.predict).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Result label
        self.result_label = ttk.Label(self.window, text="")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)
        
    def predict(self):
        """
        Makes a prediction when the predict button is clicked.

        This method creates a sample dataframe with the input values, preprocesses
        the input, makes a prediction, and displays the result.
        """
        try:
            # Create a sample dataframe with the input values
            input_data = pd.DataFrame({
                'GrLivArea': [float(self.sq_footage.get())],
                'BedroomAbvGr': [int(self.bedrooms.get())],
                'FullBath': [int(self.bathrooms.get())]
            })
            
            # Preprocess the input
            processed_input = self.preprocessor.preprocess(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_input)[0]
            
            # Display result
            self.result_label.config(text=f"Predicted House Price: ${prediction:,.2f}")
            
        except ValueError:
            self.result_label.config(text="Please enter valid numbers")
    
    def run(self):
        """
        Runs the GUI event loop.
        """
        self.window.mainloop()

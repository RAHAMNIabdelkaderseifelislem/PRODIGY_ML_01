"""
GUI for House Price Prediction

This module provides a graphical user interface for predicting house prices
using a trained machine learning model. Users can input feature values and
get predictions on house prices.

Classes:
    PredictionGUI: A class for creating and managing the prediction GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

class PredictionGUI:
    """
    A class for creating and managing the prediction GUI.

    Attributes:
        model: The trained prediction model.
        preprocessor: The preprocessor for input data.
        feature_columns: List of feature column names.
    """

    def __init__(self, model, preprocessor, feature_columns):
        """
        Initializes the PredictionGUI instance.

        Parameters:
            model: The trained prediction model.
            preprocessor: The preprocessor for input data.
            feature_columns: List of feature column names.
        """
        self.window = tk.Tk()
        self.window.title("House Price Prediction")
        self.window.geometry("800x600")
        
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        
        # Create main frame with scrollbar
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add canvas and scrollbar
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar components
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.entries = {}
        self.create_widgets()
        
    def create_widgets(self):
        """Creates input fields and buttons for each feature."""
        # Create input fields for each feature
        for i, feature in enumerate(self.feature_columns):
            ttk.Label(self.scrollable_frame, text=f"{feature}:").grid(row=i, column=0, padx=5, pady=5)
            self.entries[feature] = ttk.Entry(self.scrollable_frame)
            self.entries[feature].grid(row=i, column=1, padx=5, pady=5)
            
            # Add a default value button for each entry
            ttk.Button(
                self.scrollable_frame, 
                text="Default", 
                command=lambda f=feature: self.set_default(f)
            ).grid(row=i, column=2, padx=5, pady=5)
        
        # Predict button
        ttk.Button(
            self.scrollable_frame, 
            text="Predict Price", 
            command=self.predict
        ).grid(row=len(self.feature_columns), column=0, columnspan=3, pady=20)
        
        # Result label
        self.result_label = ttk.Label(self.scrollable_frame, text="")
        self.result_label.grid(row=len(self.feature_columns)+1, column=0, columnspan=3, pady=10)
    
    def set_default(self, feature):
        """Set default values for each feature."""
        defaults = {
            'GrLivArea': '1500',
            'BedroomAbvGr': '3',
            'FullBath': '2',
            # Add reasonable defaults for other features
        }
        
        # If we don't have a specific default, use '0'
        self.entries[feature].delete(0, tk.END)
        self.entries[feature].insert(0, defaults.get(feature, '0'))
    
    def predict(self):
        """Predicts house price based on user input and displays the result."""
        try:
            # Create a dictionary to store the input values
            input_data = {}
            
            # Get values from entries
            for feature in self.feature_columns:
                value = self.entries[feature].get()
                # Try to convert to float, if fails, keep as string
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    input_data[feature] = value
            
            # Create a dataframe with a single row
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_columns]
            
            # Preprocess the input
            processed_input = self.preprocessor.preprocess(input_df)
            
            # Make prediction
            prediction = self.model.predict(processed_input)[0]
            
            # Display result
            self.result_label.config(
                text=f"Predicted House Price: ${prediction:,.2f}",
                font=('Arial', 12, 'bold')
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}\nPlease check your inputs.")
    
    def run(self):
        """Runs the Tkinter main loop to start the GUI."""
        self.window.mainloop()

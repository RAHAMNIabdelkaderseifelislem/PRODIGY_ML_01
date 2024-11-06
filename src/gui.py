"""
Utility for generating a GUI for house price prediction using a trained model.

This module provides a class for creating a GUI that allows users to input
features of a house and predict its price using a trained model.

Classes:
    PredictionGUI: A class for creating a GUI for house price prediction.

"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import math


class PredictionGUI:
    """
    A class for creating a GUI for house price prediction.

    Attributes:
        window (tkinter.Tk): The main window of the GUI.
        model (sklearn.linear_model.LinearRegression): The trained model for house price prediction.
        preprocessor (DataPreprocessor): The preprocessor for the input data.
        feature_columns (list): The list of feature columns used for prediction.
    """
    def __init__(self, model, preprocessor, feature_columns):
        """
        Initializes the PredictionGUI instance.

        Parameters:
            model (sklearn.linear_model.LinearRegression): The trained model for house price prediction.
            preprocessor (DataPreprocessor): The preprocessor for the input data.
            feature_columns (list): The list of feature columns used for prediction.
        """
        self.window = tk.Tk()
        self.window.title("House Price Prediction 🏠")
        self.window.geometry("1200x800")

        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns

        # Create main frame with scrollbar
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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
        """
        Creates all the widgets in the GUI.
        """
        # Title
        title_label = ttk.Label(
            self.scrollable_frame, 
            text="House Price Prediction System",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=10, pady=20)

        # Calculate number of rows needed (5 columns)
        NUM_COLUMNS = 5
        num_rows = math.ceil(len(self.feature_columns) / NUM_COLUMNS)

        # Create input fields in a grid layout
        for i, feature in enumerate(self.feature_columns):
            row = (i // NUM_COLUMNS) + 1  # +1 because title is in row 0
            col = (i % NUM_COLUMNS) * 2  # *2 because each feature takes 2 columns (label + entry)
            
            # Create a frame for each feature
            frame = ttk.Frame(self.scrollable_frame)
            frame.grid(row=row, column=col, columnspan=2, padx=5, pady=5, sticky="nsew")
            
            # Add label and entry to the frame
            ttk.Label(frame, text=f"{feature}:").pack(anchor="w")
            entry = ttk.Entry(frame, width=20)
            entry.pack(fill="x", padx=2, pady=2)
            self.entries[feature] = entry

        # Control buttons frame
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=num_rows+1, column=0, columnspan=10, pady=20)

        # Add Fill Default Values button
        ttk.Button(
            button_frame,
            text="Fill Default Values",
            command=self.fill_default_values,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)

        # Add Predict button
        ttk.Button(
            button_frame,
            text="Predict Price 🎯",
            command=self.predict,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)

        # Add Clear button
        ttk.Button(
            button_frame,
            text="Clear All ❌",
            command=self.clear_all,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)

        # Result label with custom styling
        self.result_label = ttk.Label(
            self.scrollable_frame, 
            text="",
            font=('Helvetica', 12, 'bold')
        )
        self.result_label.grid(row=num_rows+2, column=0, columnspan=10, pady=20)

    def fill_default_values(self):
        """
        Fills all entries with default values.
        """
        default_values = {
            # Numerical features
            'GrLivArea': '1500',
            'BedroomAbvGr': '3',
            'FullBath': '2',
            'GarageArea': '400',
            'GarageCars': '2',
            'TotalBsmtSF': '1000',
            'WoodDeckSF': '100',
            'OpenPorchSF': '50',
            'LotArea': '8000',
            # Categorical features
            'GarageType': '1',
            'GarageFinish': '1',
            'GarageQual': '1',
            'GarageCond': '1',
            'PavedDrive': '1',
            'Fence': '1',
            'FireplaceQu': '1',
            'PoolQC': '1',
            'SaleType': '1',
            'SaleCondition': '1'
        }
        
        # Clear all entries first
        self.clear_all()
        
        # Fill with default values
        for feature in self.feature_columns:
            self.entries[feature].insert(0, default_values.get(feature, '0'))
            
        messagebox.showinfo("Success", "Default values have been filled!")

    def clear_all(self):
        """
        Clears all entry fields.
        """
        for entry in self.entries.values():
            entry.delete(0, tk.END)
    
    def predict(self):
        """
        Makes a prediction using the input values and displays the result.
        """
        try:
            # Create a dictionary to store the input values
            input_data = {}

            # Get values from entries
            for feature in self.feature_columns:
                value = self.entries[feature].get()
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

            # Display result with formatting
            formatted_price = "${:,.2f}".format(prediction)
            self.result_label.config(
                text=f"Predicted House Price: {formatted_price}",
                foreground='green'
            )

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}\nPlease check your inputs.")

    def run(self):
        """
        Runs the GUI event loop.
        """
        # Center the window on the screen
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
        
        self.window.mainloop()

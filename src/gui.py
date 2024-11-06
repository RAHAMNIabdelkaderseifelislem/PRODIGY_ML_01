"""
A GUI for making house price predictions using a trained model.

This module provides a class `PredictionGUI` that creates a GUI for making
house price predictions using a trained model. The GUI has input fields for
entering the features of a house, a button for making a prediction, and a
label for displaying the predicted price.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd


class PredictionGUI:
    """
    A GUI for making house price predictions using a trained model.

    Attributes:
        window (tkinter.Tk): The main window of the GUI.
        model (HousePriceModel): The trained model used for making predictions.
        preprocessor (DataPreprocessor): The preprocessor used for preprocessing
            the input data.
        entries (dict): A dictionary of input fields, where each key is the
            name of a feature and each value is the corresponding input field.
        result_label (tkinter.Label): The label for displaying the predicted
            price.
    """

    def __init__(self, model, preprocessor):
        """
        Initializes the PredictionGUI instance.

        Parameters:
            model (HousePriceModel): The trained model used for making predictions.
            preprocessor (DataPreprocessor): The preprocessor used for preprocessing
                the input data.
        """
        self.window = tk.Tk()
        self.window.title("House Price Prediction 🏠")
        self.window.geometry("600x400")

        self.model = model
        self.preprocessor = preprocessor

        # Create main frame
        self.main_frame = ttk.Frame(self.window, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        """
        Creates the input fields, buttons, and result label for the GUI.
        """
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="House Price Prediction System",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Input fields
        self.create_input_field("Square Footage (GrLivArea)", "GrLivArea", 1)
        self.create_input_field("Number of Bedrooms", "BedroomAbvGr", 2)
        self.create_input_field("Number of Bathrooms", "FullBath", 3)

        # Buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)

        # Add buttons
        ttk.Button(
            button_frame,
            text="Fill Default Values",
            command=self.fill_default_values
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Predict Price 🎯",
            command=self.predict
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Clear All ❌",
            command=self.clear_all
        ).pack(side=tk.LEFT, padx=5)

        # Result label
        self.result_label = ttk.Label(
            self.main_frame,
            text="",
            font=('Helvetica', 12, 'bold')
        )
        self.result_label.grid(row=5, column=0, columnspan=2, pady=20)

    def create_input_field(self, label_text, field_name, row):
        """
        Creates a frame for each input field.

        Parameters:
            label_text (str): The text for the label of the input field.
            field_name (str): The name of the feature for the input field.
            row (int): The row number for the input field in the grid layout.
        """
        # Create frame for each input field
        frame = ttk.Frame(self.main_frame)
        frame.grid(row=row, column=0, columnspan=2, pady=5, sticky="ew")

        # Add label
        ttk.Label(frame, text=label_text).pack(anchor="w")

        # Add entry
        entry = ttk.Entry(frame, width=30)
        entry.pack(fill="x", padx=2, pady=2)
        self.entries[field_name] = entry

    def fill_default_values(self):
        """
        Fills all entries with default values.
        """
        default_values = {
            'GrLivArea': '1500',
            'BedroomAbvGr': '3',
            'FullBath': '2'
        }

        self.clear_all()
        for field, value in default_values.items():
            self.entries[field].insert(0, value)

        messagebox.showinfo("Success", "Default values have been filled!")

    def clear_all(self):
        """
        Clears all entry fields.
        """
        for entry in self.entries.values():
            entry.delete(0, tk.END)

    def predict(self):
        """
        Makes a prediction using the trained model.
        """
        try:
            # Get values from entries
            input_data = {
                'GrLivArea': float(self.entries['GrLivArea'].get()),
                'BedroomAbvGr': float(self.entries['BedroomAbvGr'].get()),
                'FullBath': float(self.entries['FullBath'].get())
            }

            # Create a dataframe with a single row
            input_df = pd.DataFrame([input_data])

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

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

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

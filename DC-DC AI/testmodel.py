import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from sklearn.model_selection import train_test_split
from colorama import Fore, Style
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


print(tf.config.list_physical_devices('GPU'))

def printc(message, color):
    match color:
        case "green":
            print(Fore.GREEN + message + Style.RESET_ALL)
            return 1
        case "red":
            print(Fore.RED + message + Style.RESET_ALL)
            return 1
        case "yellow":
            print(Fore.YELLOW + message + Style.RESET_ALL)

# Function to convert strings or lists to float
def convert_to_float(val):
    try:
        # If the value is a string representation of a list (e.g. '[1.69497325e-05]')
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            val = val.strip('[]')  # Remove brackets
        return float(val)
    except ValueError:
        return np.nan  # Return NaN if conversion fails

# Path to the folder with CSV files
folder_path = r'C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\testingData'
printc("Starting program", "green")

# Placeholder for training data
X = []
y = []
progressTot = 23
index = 0
printc("Parsing through files...", "yellow")

# Iterate through CSV files
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        
        #Progress Bar
        percent = index / progressTot * 100
        index += 1
        bar = 'X' * int(percent / 2) + '-' * (50 - int(percent / 2))
        sys.stdout = sys.__stdout__
        sys.stdout.write(f'\r|{bar}| {percent:.2f}%   adding file: ' + filename)  
        sys.stdout.flush()

        #Data pull
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)
        input_data = data.iloc[1:, [0, 1, 2, 3]].values  # Get entire 3rd and 4th columns

        # Convert the values to float
        input_data = np.array([[convert_to_float(cell) for cell in row] for row in input_data])

        # Ensure the data is numeric
        input_data = input_data.astype(np.float32)

        # Extract output (second value from the fifth column)
        output = convert_to_float(data.iloc[1, 4])  # Convert to float

        # Store the data
        X.append(input_data)
        y.append(np.full(input_data.shape[0], output, dtype=np.float32))

printc("Complete", "green")

# Convert to numpy arrays
printc("Converting to numpy array", "yellow")
X = np.vstack(X)
y = np.hstack(y)
printc("Complete", "green")

# Check if there are any NaN or infinite values in X or y
if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    printc("Warning: Found NaN values in X or y", "red")
if np.any(np.isinf(X)) or np.any(np.isinf(y)):
    printc("Warning: Found infinite values in X or y", "red")

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Ensure the input shape is correct
print("X_train shape:", X_train.shape)

# Define a simple neural network
model = load_model('trained_model.h5', compile=False)  # Load without compiling
model.compile(optimizer='adam', loss=MeanSquaredError())  # Recompile with explicit loss function






printc("Testing...", "yellow")

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)   
printc("Complete", "green")

print("Test Loss:", loss)



printc("All finished!", "green")

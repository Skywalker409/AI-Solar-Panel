import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from colorama import Fore, Style

print(tf.config.list_physical_devices('GPU'))

def printc(message, color):
    match color:
        case "green":
            print(Fore.GREEN + message + Style.RESET_ALL)
        case "red":
            print(Fore.RED + message + Style.RESET_ALL)
        case "yellow":
            print(Fore.YELLOW + message + Style.RESET_ALL)

# Function to convert strings or lists to float
def convert_to_float(val):
    try:
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            val = val.strip('[]')  # Remove brackets
        return float(val)
    except ValueError:
        return np.nan  # Return NaN if conversion fails

# Path to the folder with CSV files
folder_path = r'C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\GeneratedDataV3'
printc("Starting program", "green")

# Placeholder for training data
X = []
y = []
progressTot = sum(1 for f in os.listdir(folder_path) if f.endswith('.csv'))
index = 0
printc("Parsing through files...", "yellow")

# Iterate through CSV files
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Progress Bar
        percent = index / progressTot * 100
        index += 1
        bar = 'X' * int(percent / 2) + '-' * (50 - int(percent / 2))
        sys.stdout = sys.__stdout__
        sys.stdout.write(f'\r|{bar}| {percent:.2f}%   adding file: ' + filename)
        sys.stdout.flush()

        # Data pull
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)
        input_data = data.iloc[1:, [0, 1, 2, 3]].values  # First 4 columns after header

        # Convert to float
        input_data = np.array([[convert_to_float(cell) for cell in row] for row in input_data])
        input_data = input_data.astype(np.float32)

        # Extract output (second value from the fifth column)
        output = convert_to_float(data.iloc[1, 4])

        # Store the data
        X.append(input_data)
        y.append(np.full(input_data.shape[0], output, dtype=np.float32))

printc("Complete", "green")

# Convert to numpy arrays
printc("Converting to numpy array", "yellow")
X = np.vstack(X)
y = np.hstack(y)
printc("Complete", "green")

# Check for bad values
if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    printc("Warning: Found NaN values in X or y", "red")
if np.any(np.isinf(X)) or np.any(np.isinf(y)):
    printc("Warning: Found infinite values in X or y", "red")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)

# Define model
model = Sequential([
    Dense(15, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),
    Dense(1)
])

printc("Compiling model...", "yellow")
model.compile(optimizer='adam', loss='mse')

# Train the model and store training history
printc("Training model...", "yellow")
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
printc("Training complete", "green")

# Evaluate
printc("Testing...", "yellow")
loss = model.evaluate(X_test, y_test)
printc("Testing complete", "green")
print("Test Loss:", loss)

# Save model
name = 'DC-DemoModel_2.1.h5'
model.save(name)
printc(f"Model saved to {name}", "green")

# Plot training & validation loss
printc("Plotting loss curve...", "yellow")
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss vs Epochs')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig("improvements.png")
plt.show()

printc("All finished!", "green")

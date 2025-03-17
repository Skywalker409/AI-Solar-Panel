import pandas as pd
import numpy as np
import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from colorama import Fore, Style

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



# Path to the folder with CSV files
folder_path = r'C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\GeneratedData'
printc("Starting program", "green")
# Placeholder for training data
X = []
y = []
progressTot = 312
index = 0
test =1
printc("Parsing through files...", "yellow")
# Iterate through CSV files
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        
        percent = index / progressTot * 100
        index +=1
        bar = 'X' * int(percent / 2) + '-' * (50 - int(percent / 2))
        sys.stdout = sys.__stdout__
        sys.stdout.write(f'\r|{bar}| {percent:.2f}%')  # '\r' moves the cursor back to the start of the line
        sys.stdout.flush()
        sys.stdout = open(os.devnull, 'w')

        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)

        # Extract input features (first two columns excluding header)
        input_data = data.iloc[1:, [0, 1]].values  # Skip header
        if(test):
            test +=1
            print(type(input_data))
        # Extract constant values
        const_val_1 = data.iloc[1, 2]  # Second value of third column
        const_val_2 = data.iloc[1, 3]  # Second value of fourth column

        # Extract output (second value from the fifth column)
        output = data.iloc[1, 4]

        # Combine input and constants
        combined_input = np.hstack([input_data, 
                                    np.full((input_data.shape[0], 1), const_val_1), 
                                    np.full((input_data.shape[0], 1), const_val_2)])

        # Store the data
        X.append(combined_input)
        y.append(np.full(input_data.shape[0], output))

printc("Complete", "green")
# Convert to numpy arrays
printc("Converting to numpy array", "yellow")

X = np.vstack(X)
y = np.hstack(y)
printc("Complete", "green")


# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),  # 4 features (2 from input, 2 constants)
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

printc("Compliling model...", "yellow")

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
printc("Complete", "green")


printc("Testing...", "yellow")

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
printc("Complete", "green")

print("Test Loss:", loss)

# Save the model
model.save('trained_model.h5')

printc("All finished!", "green")


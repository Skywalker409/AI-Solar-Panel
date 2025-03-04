import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the Excel file
file_path = "25TData/1000IR.xlsx"  # Change this to your file name
try:
    df = pd.read_excel(file_path, engine="openpyxl")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
except ValueError as e:
    print(f"Error: Could not open the file. Check the format. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Display basic info
print("Dataset Preview:\n", df.head())

# Drop missing values
df = df.dropna()

# Encode categorical labels (if applicable)
target_column = "Voltage Measurement"  # Change this to the actual column name
if df[target_column].dtype == 'object':
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])

# Separate features and labels
X = df.drop(columns=[target_column])
y = df[target_column]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# Define a neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # Use 'softmax' for multi-class classification
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the trained model and scaler
model.save("deep_learning_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

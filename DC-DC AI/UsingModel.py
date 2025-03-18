import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the model and explicitly define the loss function
model = load_model('trained_model.h5', compile=False)  # Load without compiling
model.compile(optimizer='adam', loss=MeanSquaredError())  # Recompile with explicit loss function


# Function to take user input and convert it to a NumPy array
def get_user_input():
    print("Enter values for the four input features:")
    inputs = []
    inputNames =["Voltage", "Current", "Irradience", "Temperature"]
    for i in range(4):
        value = float(input(f"{inputNames[i]}: "))  # Get user input
        inputs.append(value)
    
    # Convert to NumPy array and reshape for model input
    input_data = np.array([inputs], dtype=np.float32)  # Shape (1,4)
    return input_data

# Get user input
new_input = get_user_input()

# Make a prediction
prediction = model.predict(new_input)

# Display the predicted result
print("\nPredicted Output:", prediction[0][0])

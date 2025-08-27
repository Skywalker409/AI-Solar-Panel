import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tqdm import tqdm

# --- Load trained model ---
model = load_model("DC-DemoModel_2.1.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError())

# --- Helper function to convert bracketed strings like "[0.123]" ---
def convert_to_float(val):

    try:
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            val = val.strip("[]")
        return float(val)
    except ValueError:
        return np.nan

# --- Path to testing folder ---    
folder_path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\testingData"

# Get list of CSV files
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

all_data = []

# --- Loop through all CSVs with progress bar ---
for filename in tqdm(csv_files, desc="Processing CSVs", ncols=100):
    file_path = os.path.join(folder_path, filename)
    data = pd.read_csv(file_path)
    
    # Convert columns to float
    data = data.apply(lambda col: col.map(convert_to_float))

    # Inputs: Vout, Iout, Irradiance, Temperature
    X = data.iloc[1:, [0, 1, 2, 3]].values.astype(np.float32)

    # Ground truth MaxVoltage
    y_actual = convert_to_float(data.iloc[1, 4])

    # Model prediction (average if multiple rows)
    y_pred = model.predict(X, verbose=0).mean()

    # % Error
    error_percent = abs((y_actual - y_pred) / y_actual) * 100 if y_actual != 0 else np.nan

    # Store Irradiance, Temperature, Error%
    irradiance = convert_to_float(data.iloc[1, 2])
    temperature = convert_to_float(data.iloc[1, 3])
    all_data.append([irradiance, temperature, error_percent])

# --- Convert to DataFrame ---
df = pd.DataFrame(all_data, columns=["Irradiance", "Temperature", "Error%"])

# --- Pivot for heatmap ---
heatmap_data = df.pivot(index="Temperature", columns="Irradiance", values="Error%")

# --- Plot heatmap ---
plt.figure(figsize=(12, 8))
plt.title("Error Heatmap (Temp vs Irradiance)", fontsize=14)
plt.xlabel("Irradiance")
plt.ylabel("Temperature")

c = plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap='hot')
plt.colorbar(c, label="Error %")

# Set axis ticks
plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45)
plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)

plt.tight_layout()

save_path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\TestResults\DC-Demo-Heatmap.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Heatmap automatically saved as {save_path}")
plt.show()

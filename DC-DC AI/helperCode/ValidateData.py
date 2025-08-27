import pandas as pd
import os
from tqdm import tqdm

# --- Configuration ---
folder_path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\testingData"  # Change as needed

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

print(f"Found {len(csv_files)} CSV files in {folder_path}")

# --- Loop through files ---
for filename in tqdm(csv_files, desc="Validating CSVs", ncols=100):
    file_path = os.path.join(folder_path, filename)
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Check for NaN
    if df.isna().any().any():
        print(f"❌ NaN values found in {filename}")
    
    # Check for infinite values
    if df.isin([float('inf'), float('-inf')]).any().any():
        print(f"❌ Infinite values found in {filename}")

print("✅ Validation complete.")

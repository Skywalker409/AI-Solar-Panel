import pandas as pd
import numpy as np
import os
from tqdm import tqdm  # optional, for progress bar

# --- Configuration ---
folder_path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\testingData"  # Change to your folder

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

print(f"Found {len(csv_files)} CSV files in {folder_path}")

# --- Loop through files ---
for filename in tqdm(csv_files, desc="Cleaning CSVs", ncols=100):
    file_path = os.path.join(folder_path, filename)
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Count rows before cleaning
    rows_before = len(df)
    
    # Drop rows with any NaN values
    df_clean = df.dropna()
    
    # Count rows after cleaning
    rows_after = len(df_clean)
    rows_removed = rows_before - rows_after
    
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with NaN from {filename}")
    
    # Save cleaned CSV (overwrite original)
    df_clean.to_csv(file_path, index=False)

print("✅ Cleaning complete. All CSVs are now free of NaN rows.")

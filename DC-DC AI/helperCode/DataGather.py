import matlab.engine
from colorama import Fore, Style
import sys
import numpy as np
import csv
import os
from tqdm import tqdm  # ✅ tqdm progress bar

# --- Helper function to clean arrays ---
def clean_vout_iout(vout_data, iout_data):
    """
    Remove rows where Vout or Iout are NaN or Inf.
    Returns cleaned arrays.
    """
    vout_data = np.array(vout_data, dtype=np.float64).flatten()
    iout_data = np.array(iout_data, dtype=np.float64).flatten()
    mask = np.isfinite(vout_data) & np.isfinite(iout_data)
    return vout_data[mask], iout_data[mask]

# --- Define the folder path ---
path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\DataV4"
os.makedirs(path, exist_ok=True)

# Clear folder
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# --- Start MATLAB engine ---
eng = matlab.engine.start_matlab()
model_name = "panelSim"

# --- Parameter ranges ---
MinIrradiance = 1
MaxIrradiance = 2001
IrrInc = 100
MinTemp = -25
MaxTemp = 55
TempInc = 5

# --- Total iterations for tqdm ---
total_iters = len(range(MinIrradiance, MaxIrradiance, IrrInc)) * len(range(MinTemp, MaxTemp, TempInc))

print(Fore.YELLOW + "Running Simulations... Press Ctrl+C to stop anytime." + Style.RESET_ALL)

try:
    with tqdm(total=total_iters, desc="Simulations", unit="run") as pbar:
        for irradiance in range(MinIrradiance, MaxIrradiance, IrrInc):
            for temperature in range(MinTemp, MaxTemp, TempInc):
                try:
                    # Load Simulink model
                    eng.load_system(model_name)
                    eng.workspace["irradiance"] = irradiance
                    eng.workspace["temp"] = temperature

                    # Run simulation
                    eng.sim(model_name, nargout=0)

                    # Extract simulation data
                    vout_data = np.array(eng.eval("Vout")).flatten()
                    iout_data = np.array(eng.eval("Iout")).flatten()

                    # --- Clean vout/iout from NaN or Inf ---
                    vout_data, iout_data = clean_vout_iout(vout_data, iout_data)

                    # Skip if no valid data remains
                    if vout_data.size == 0 or iout_data.size == 0:
                        print(Fore.RED + f"⚠️ No valid data after cleaning temp={temperature}, irr={irradiance}" + Style.RESET_ALL)
                        pbar.update(1)
                        continue

                    # Calculate power
                    power = vout_data * iout_data

                    # Find maximum power and corresponding voltage
                    max_index = int(np.argmax(power))
                    maxpower = float(power[max_index])
                    MaxVoltage = float(vout_data[max_index])

                    # 🚨 Hard stop check if NaN or Inf
                    if not np.isfinite(maxpower) or not np.isfinite(MaxVoltage):
                        print(Fore.RED + f"\n🚨 ERROR: Invalid MaxPower/MaxVoltage detected temp={temperature}, irr={irradiance}. Stopping program." + Style.RESET_ALL)
                        eng.quit()
                        sys.exit(1)

                    # --- Save to CSV ---
                    filename = f"temp_{temperature}_irr_{irradiance}.csv"
                    filepath = os.path.join(path, filename)
                    with open(filepath, "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["Vout", "Iout", "Irradiance", "Temperature", "MaxVoltage", "MaxPower"])
                        for v, i in zip(vout_data, iout_data):
                            writer.writerow([float(v), float(i), irradiance, temperature, MaxVoltage, maxpower])

                except Exception as e:
                    print(Fore.RED + f"Error during simulation for temp={temperature}, irr={irradiance}: {e}" + Style.RESET_ALL)

                pbar.update(1)

except KeyboardInterrupt:
    print(Fore.RED + "\nSimulation interrupted by user. Exiting program..." + Style.RESET_ALL)

finally:
    eng.quit()
    print(Fore.GREEN + "MATLAB engine closed. Program terminated." + Style.RESET_ALL)

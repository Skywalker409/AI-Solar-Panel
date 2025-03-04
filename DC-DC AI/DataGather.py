import matlab.engine
from colorama import Fore, Style
import sys
import numpy as np
import csv
import os  # Added for path handling

# Define the folder path and ensure it's valid
try:
    path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\DC-DC AI\GeneratedData"
    os.makedirs(path, exist_ok=True)  # Create folder if it doesn't exist
except Exception as e:
    print("Failed to find path. \n Error: \n" + e)


# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define Simulink model name
model_name = "panelSim"

# Load the Simulink model
MinIrradiance = 0
MaxIrradiance = 2501
IrrInc = 100

MinTemp = -25
MaxTemp = 35
TempInc = 5

progressTot = 2500

print(Fore.YELLOW + "Running Simulations..." + Style.RESET_ALL)

for irradiance in range(MinIrradiance, MaxIrradiance, IrrInc):
    percent = irradiance / progressTot * 100
    bar = ' ' * int(percent / 2) + '-' * (50 - int(percent / 2))
    sys.stdout.write(f'\r|{bar}| {percent:.2f}%')  # '\r' moves the cursor back to the start of the line
    sys.stdout.flush()

    for temperature in range(MinTemp, MaxTemp, TempInc):
        try:
            # Load Simulink model
            eng.load_system(model_name)
            print(Fore.GREEN + "Loaded the simulation properly" + Style.RESET_ALL)

            # Set input values in MATLAB workspace
            eng.workspace["irradiance"] = irradiance
            eng.workspace["temp"] = temperature

            # Run simulation
            eng.sim(model_name, nargout=0)

            # Extract simulation data
            vout_data = np.array(eng.eval("Vout"))  # Convert MATLAB arrays to NumPy arrays
            iout_data = np.array(eng.eval("Iout"))
            power = vout_data * iout_data  # Element-wise multiplication

            # Find maximum power and corresponding voltage
            maxpower = max(power)
            max_index = np.argmax(power)  # Use NumPy's argmax for efficiency
            MaxVoltage = vout_data[max_index]

            # Prepare data for CSV export
            data = {
                "Vout (Voltage)": vout_data,
                "Iout (Current)": iout_data,
                "Power (W)": power,
                "Max Voltage (V)": [MaxVoltage]  # Single value as a list for consistency
            }

            # Write data to a CSV file
            filename = f"temp_{temperature}_irr_{irradiance}.csv"
            filepath = os.path.join(path, filename)

            with open(filepath, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(data.keys())  # Write headers
                writer.writerows(zip(*data.values()))  # Write data rows

        except Exception as e:
            print(Fore.RED + f"Error during simulation for temp={temperature}, irr={irradiance}: {e}" + Style.RESET_ALL)

print("\nTask completed!")
# Close MATLAB engine
eng.quit()

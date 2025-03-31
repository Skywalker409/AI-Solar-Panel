import matlab.engine
from colorama import Fore, Style
import sys
import numpy as np
import csv
import os  # Added for path handling

# Define the folder path and ensure it's valid
try:
    path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\testingData"
    os.makedirs(path, exist_ok=True)  # Create folder if it doesn't exist
except Exception as e:
    print("Failed to find path. \n Error: \n" + str(e))
    sys.exit(1)  # Exit program if path creation fails

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define Simulink model name
model_name = "panelSim"

# Load the Simulink model
MinIrradiance = 1
MaxIrradiance = 2502
IrrInc = 333

MinTemp = -25
MaxTemp = 35
TempInc = 23

progressTot = 2500

print(Fore.YELLOW + "Running Simulations... Press Ctrl+C to stop anytime." + Style.RESET_ALL)

try:
    for irradiance in range(MinIrradiance, MaxIrradiance, IrrInc):
        percent = irradiance / progressTot * 100
        bar = 'X' * int(percent / 2) + '-' * (50 - int(percent / 2))
        sys.stdout = sys.__stdout__
        sys.stdout.write(f'\r|{bar}| {percent:.2f}%')  # '\r' moves the cursor back to the start of the line
        sys.stdout.flush()
        sys.stdout = open(os.devnull, 'w')

        for temperature in range(MinTemp, MaxTemp, TempInc):
            try:
                # Load Simulink model
                eng.load_system(model_name)

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
                data = [vout_data,iout_data,irradiance,temperature,MaxVoltage]

                # Write data to a CSV file
                filename = f"temp_{temperature}_irr_{irradiance}.csv"
                filepath = os.path.join(path, filename)

                with open(filepath, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Vout", "Iout", "Irradiance", "Temperature", "MaxVoltage"])  # Headers
                    for v, i in zip(vout_data, iout_data):  # Assuming vout_data and iout_data are of equal length
                        writer.writerow([v, i, irradiance, temperature, MaxVoltage])


                    
            except Exception as e:
                print(Fore.RED + f"Error during simulation for temp={temperature}, irr={irradiance}: {e}" + Style.RESET_ALL)

except KeyboardInterrupt:
    print(Fore.RED + "\nSimulation interrupted by user. Exiting program..." + Style.RESET_ALL)

finally:
    # Close MATLAB engine
    eng.quit()
    print(Fore.GREEN + "MATLAB engine closed. Program terminated." + Style.RESET_ALL)

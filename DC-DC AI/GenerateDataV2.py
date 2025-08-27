import matlab.engine
import numpy as np
import csv
import os
from colorama import Fore, Style
import multiprocessing as mp
from tqdm import tqdm

# Folder path for saving data
path = r"C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\GeneratedDataV2"
os.makedirs(path, exist_ok=True)

# Clear existing files
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Simulation parameters
MinIrradiance = 1
MaxIrradiance = 2001
IrrInc = 100
MinTemp = -25
MaxTemp = 55
TempInc = 5
model_name = "panelSim"

# Shared counter for progress
counter = mp.Value('i', 0)

def run_simulation(params):
    irradiance, temperature, path = params
    try:
        eng = matlab.engine.start_matlab()
        eng.load_system(model_name)  # Load model once per engine
        eng.workspace["irradiance"] = irradiance
        eng.workspace["temp"] = temperature
        eng.sim(model_name, nargout=0)

        # Extract simulation data
        vout_data = np.array(eng.eval("Vout"))
        iout_data = np.array(eng.eval("Iout"))
        power = vout_data * iout_data
        max_index = np.argmax(power)
        MaxVoltage = vout_data[max_index]
        maxpower = power[max_index]

        # Write CSV
        filename = f"temp_{temperature}_irr_{irradiance}.csv"
        filepath = os.path.join(path, filename)
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Vout", "Iout", "Irradiance", "Temperature", "MaxVoltage", "MaxPower"])
            for v, i in zip(vout_data, iout_data):
                writer.writerow([v, i, irradiance, temperature, MaxVoltage, maxpower])

        eng.quit()

    except Exception as e:
        print(Fore.RED + f"Error for temp={temperature}, irr={irradiance}: {e}" + Style.RESET_ALL)

    # Increment shared counter
    with counter.get_lock():
        counter.value += 1

# Generate all parameter combinations
params_list = [(irr, temp, path) for irr in range(MinIrradiance, MaxIrradiance, IrrInc)
                             for temp in range(MinTemp, MaxTemp, TempInc)]

def worker_init(shared_counter):
    global counter
    counter = shared_counter

if __name__ == "__main__":
    total_jobs = len(params_list)
    print(Fore.YELLOW + "Running Simulations in parallel..." + Style.RESET_ALL)

    # Use Pool with initializer to share counter
    with mp.Pool(processes=4, initializer=worker_init, initargs=(counter,)) as pool:
        # Use imap_unordered to iterate results and update progress bar
        with tqdm(total=total_jobs, desc="Simulations", ncols=100) as pbar:
            for _ in pool.imap_unordered(run_simulation, params_list):
                pbar.update(1)

    print(Fore.GREEN + "All simulations complete." + Style.RESET_ALL)

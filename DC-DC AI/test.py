import matlab.engine
from colorama import Fore, Style

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define Simulink model name
model_name = "panelSim"

# Load the Simulink model
eng.load_system(model_name)
print(Fore.GREEN + "Loaded the simulation properly" + Style.RESET_ALL)

# Set input values
eng.workspace["irradiance"] = 800  
eng.workspace["temp"] = 25  
print(Fore.GREEN + "Declared Variables" + Style.RESET_ALL)

print(Fore.YELLOW + "Running Simulation" + Style.RESET_ALL)
eng.sim(model_name, nargout=0)
print(Fore.GREEN + "Simulation Complete!" + Style.RESET_ALL)


# Attempt to extract logsout
try:
 

    vout_data = eng.eval("Vout")
    iout_data = eng.eval("Iout")
    print(Fore.CYAN + "Extracted Time Data:" + Style.RESET_ALL)
    print("Second values:" , vout_data[1], iout_data[1])

except Exception as e:
    print(Fore.RED + "Failed to extract logsout: " + str(e) + Style.RESET_ALL)

# Close MATLAB engine
eng.quit()

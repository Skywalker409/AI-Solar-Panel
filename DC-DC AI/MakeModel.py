import os
import sys
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    ipex_available = True
except ImportError:
    ipex_available = False

# --- Utility function for colored prints ---
def printc(message, color):
    match color:
        case "green":
            print(Fore.GREEN + message + Style.RESET_ALL)
        case "red":
            print(Fore.RED + message + Style.RESET_ALL)
        case "yellow":
            print(Fore.YELLOW + message + Style.RESET_ALL)

# --- Convert string/list to float ---
def convert_to_float(val):
    try:
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            val = val.strip('[]')
        return float(val)
    except ValueError:
        return np.nan

# --- Load and preprocess data ---
folder_path = r'C:\Users\lukel\OneDrive\Desktop\CAPSTONE\AI-Solar-Panel\DC-DC AI\GeneratedDataV3'
printc("Starting program", "green")

X_list = []
y_list = []
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for filename in tqdm(csv_files, desc="Processing CSVs"):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)
    
    # Convert first 4 columns to float
    X_data = df.iloc[1:, [0,1,2,3]].applymap(convert_to_float).to_numpy(dtype=np.float32)
    
    # Output: MaxVoltage
    y_val = convert_to_float(df.iloc[1, 4])
    
    # Append data
    X_list.append(X_data)
    y_list.append(np.full(X_data.shape[0], y_val, dtype=np.float32))

# Combine all data
X = np.vstack(X_list)
y = np.hstack(y_list)

# Check for NaN or Inf
if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    printc("Warning: Found NaN values in X or y", "red")
if np.any(np.isinf(X)) or np.any(np.isinf(y)):
    printc("Warning: Found infinite values in X or y", "red")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)

# --- Device selection ---
if ipex_available and torch.backends.xpu.is_available():
    device = torch.device("xpu")  # Intel Arc GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU fallback
else:
    raise RuntimeError("No compatible GPU detected. CPU execution is not allowed.")

printc(f"Using device: {device}", "green")

# --- Convert to torch tensors ---
X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).unsqueeze(1).to(device)
X_test_t = torch.from_numpy(X_test).to(device)
y_test_t = torch.from_numpy(y_test).unsqueeze(1).to(device)

# --- Define PyTorch model ---
class SolarNet(nn.Module):
    def __init__(self):
        super(SolarNet, self).__init__()
        self.fc1 = nn.Linear(4, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SolarNet().to(device)

# Apply Intel Extension optimizations if available
if ipex_available and device.type == "xpu":
    model, optimizer = ipex.optimize(model, dtype=torch.float32, optimizer=optim.Adam(model.parameters(), lr=0.001))
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.MSELoss()

# --- Training ---
epochs = 15
batch_size = 32
n_samples = X_train_t.shape[0]

train_losses = []
val_losses = []

printc("Training model...", "yellow")

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(n_samples)
    epoch_loss = 0.0
    for i in range(0, n_samples, batch_size):
        indices = perm[i:i+batch_size]
        batch_X = X_train_t[indices]
        batch_y = y_train_t[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
    
    epoch_loss /= n_samples
    train_losses.append(epoch_loss)
    
    # Validation loss
    model.eval()
    with torch.no_grad():
        val_output = model(X_test_t)
        val_loss = criterion(val_output, y_test_t).item()
        val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f}")

printc("Training complete", "green")

# --- Evaluate ---
model.eval()
with torch.no_grad():
    test_output = model(X_test_t)
    test_loss = criterion(test_output, y_test_t).item()
printc("Testing complete", "green")
print("Test Loss:", test_loss)

# --- Save model ---
torch.save(model.state_dict(), "DC-DemoModel_PyTorch.pt")
printc("Model saved to DC-DemoModel_PyTorch.pt", "green")

# --- Plot training curve ---
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Model Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("improvements_pytorch.png")
plt.show()

printc("All finished!", "green")

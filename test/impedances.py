import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# --- 1. Setup Paths and Load Data ---
# Define the path to the config folder relative to the script
config_dir = os.path.join('..', 'config')
params_file = os.path.join(config_dir, 'params.yaml')
grid_file = os.path.join(config_dir, 'grid_coords_valid.yaml')

print(f"Loading parameters from: {params_file}")
try:
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find params.yaml at {os.path.abspath(params_file)}")

print(f"Loading grid coordinates from: {grid_file}")
try:
    with open(grid_file, 'r') as f:
        coords = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find grid coordinates at {os.path.abspath(grid_file)}")

# Extract Physical Cortical Coordinates (mm)
x_phys = np.array(coords['x'])
y_phys = np.array(coords['y'])
num_electrodes = len(x_phys)

# --- 2. Initialize Impedances ---
imp_params = params.get('impedance', {})
mean_imp = imp_params.get('mean_impedance', 47000) # Default 47k Ohm
sd_imp = imp_params.get('sd_impedance', 4800)      # Default 4.8k Ohm
seed = imp_params.get('seed', 1234)

print(f"Generating impedances for {num_electrodes} electrodes...")
print(f"Target: Mean={mean_imp} Ohms, SD={sd_imp} Ohms")

# Generate Truncated Normal Distribution
if seed is not None:
    np.random.seed(seed)

# Bounds for truncation ( +/- 2 SD )
lower_bound = mean_imp - 2 * sd_imp
upper_bound = mean_imp + 2 * sd_imp
a = (lower_bound - mean_imp) / sd_imp
b = (upper_bound - mean_imp) / sd_imp

# Generate values
impedances = truncnorm.rvs(a, b, loc=mean_imp, scale=sd_imp, size=num_electrodes)

# --- 3. Plot the Physical Grid ---
plt.figure(figsize=(10, 8))

# Scatter plot: X/Y are physical coordinates (mm). Color is Impedance.
sc = plt.scatter(x_phys, y_phys, c=impedances, cmap='viridis', s=80, 
                 edgecolor='white', linewidth=0.5)

# Aesthetics
cbar = plt.colorbar(sc)
cbar.set_label('Impedance ($\Omega$)', fontsize=12)

plt.title(f'Step 1: Physical Electrode Grid & Impedance Initialization\n'
          f'($\mu={mean_imp:.0f}\Omega, \sigma={sd_imp:.0f}\Omega$)', fontsize=14)
plt.xlabel('Cortical Position X (mm)', fontsize=12)
plt.ylabel('Cortical Position Y (mm)', fontsize=12)
plt.axis('equal')  # Crucial to see the real shape of the array
plt.grid(True, linestyle='--', alpha=0.5)

# Save result
output_filename = 'step1_physical_impedance_map.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"Plot saved to {output_filename}")
plt.show()
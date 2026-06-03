import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Visualize Dynaphos Results")
    parser.add_argument('--data', type=str, required=True, help="Path to the .npy file (e.g., average_power.npy)")
    parser.add_argument('--grid', type=str, required=True, help="Path to the grid coordinates .yaml file")
    args = parser.parse_args()

    # 1. Load Data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        return
    
    print(f"Loading data from {data_path.name}...")
    # Shape: (Frames, Electrodes)
    power_data = np.load(data_path)
    print(f"Data Shape: {power_data.shape} (Frames, Electrodes)")

    # 2. Load Coordinates
    grid_path = Path(args.grid)
    if not grid_path.exists():
        print(f"❌ Error: Grid file not found: {grid_path}")
        return

    print(f"Loading coordinates from {grid_path.name}...")
    with open(grid_path, 'r') as f:
        coords = yaml.safe_load(f)
        x = np.array(coords['x'])
        y = np.array(coords['y'])
    
    print(f"Grid Points: {len(x)}")

    # 3. Check Compatibility
    n_electrodes = power_data.shape[1]
    if n_electrodes != len(x):
        print("\n⚠️  DIMENSION MISMATCH ERROR ⚠️")
        print(f"Data has {n_electrodes} electrodes.")
        print(f"Grid has {len(x)} coordinates.")
        print(">> You must use the exact grid file used during the simulation.")
        return

    # 4. Prepare Visualization
    # Calculate Total Energy (Sum over time) for the heatmap
    total_energy = np.sum(power_data, axis=0)
    
    # Calculate Max Peak (Max over time)
    max_peak = np.max(power_data, axis=0)

    # 5. Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot A: Total Energy
    sc1 = axes[0].scatter(x, y, c=total_energy, cmap='inferno', s=20)
    plt.colorbar(sc1, ax=axes[0], label='Cumulative Power (Sum)')
    axes[0].set_title(f"Total Energy Distribution\n{data_path.name}")
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    # Plot B: Peak Power
    sc2 = axes[1].scatter(x, y, c=max_peak, cmap='magma', s=20)
    plt.colorbar(sc2, ax=axes[1], label='Peak Power (Max)')
    axes[1].set_title(f"Peak Power Hotspots\n(Worst-case moment)")
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    print("✓ Plot generated.")

if __name__ == "__main__":
    main()
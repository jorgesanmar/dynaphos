import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import yaml
import argparse
# --- HOW IT's USED ---

# Example usage:
# python plot_power_per_electrode.py --mode specific --indices 10 240 500
# python plot_power_per_electrode.py --mode rank --n 10 --rank-type highest
# python plot_power_per_electrode.py --mode spatial --n 10

# --- 1. SETUP PATHS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

config_path = project_root / 'config'
output_folder = "instant_power_tracking"
results_dir = project_root / 'test results' / output_folder
npy_path = results_dir / 'power_per_electrode.npy'
params_path = config_path / 'params.yaml'

# Check for grid file
grid_file = config_path / 'grid_coords_dipole.yaml'
if not grid_file.exists():
    grid_file = config_path / 'grid_coords.yaml'

def get_evenly_spaced_indices(x_coords, y_coords, valid_indices, n_samples):
    """Selects n_samples indices distributed evenly using K-Means logic."""
    if len(valid_indices) < n_samples:
        return valid_indices

    active_x = x_coords[valid_indices]
    active_y = y_coords[valid_indices]
    points = np.column_stack((active_x, active_y))

    rng = np.random.RandomState(42)
    centroids = points[rng.choice(len(points), n_samples, replace=False)]

    for _ in range(10):
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([points[labels == i].mean(axis=0) 
                                if np.sum(labels == i) > 0 else centroids[i]
                                for i in range(n_samples)])
        centroids = new_centroids

    selected_subset_indices = []
    for centroid in centroids:
        dists = np.sum((points - centroid)**2, axis=1)
        selected_subset_indices.append(valid_indices[np.argmin(dists)])
    
    return np.unique(selected_subset_indices)

def main():
    # --- 2. ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Analyze electrode power evolution.")
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['specific', 'rank', 'spatial'], default='spatial',
                        help='Mode: specific (by index), rank (highest/lowest power), spatial (evenly spaced).')
    
    # Arguments for 'specific' mode
    parser.add_argument('--indices', type=int, nargs='+', help='List of electrode indices to plot (e.g., 10 240 500).')
    
    # Arguments for 'rank' and 'spatial' modes
    parser.add_argument('--n', type=int, default=5, help='Number of electrodes to plot.')
    parser.add_argument('--rank-type', type=str, choices=['highest', 'lowest'], default='highest',
                        help="For 'rank' mode: choose 'highest' or 'lowest' total power.")

    args = parser.parse_args()

    # --- 3. LOAD DATA ---
    if not npy_path.exists():
        print(f"Error: Data not found at {npy_path}")
        return

    print("Loading power data...")
    power_data = np.load(npy_path) # Shape: (Frames, Electrodes)
    n_frames, n_total = power_data.shape

    print("Loading spatial coordinates...")
    with open(grid_file, 'r') as f:
        coords = yaml.safe_load(f)
    x_all = np.array(coords['x'])
    y_all = np.array(coords['y'])

    # Calculate Total Energy per electrode
    total_energy = np.sum(power_data, axis=0)
    active_mask = total_energy > 1e-6
    active_indices = np.where(active_mask)[0]
    
    print(f"Total Electrodes: {n_total} | Active: {len(active_indices)}")

    # --- 4. SELECT ELECTRODES BASED ON MODE ---
    selected_indices = []
    title_suffix = ""

    if args.mode == 'specific':
        if not args.indices:
            print("Error: --mode specific requires --indices (e.g., --indices 10 20)")
            return
        # Filter out indices that are out of bounds
        selected_indices = [i for i in args.indices if 0 <= i < n_total]
        title_suffix = f"Electrode {selected_indices}"
        
    elif args.mode == 'rank':
        sorted_indices = np.argsort(total_energy)
        if args.rank_type == 'highest':
            selected_indices = sorted_indices[-args.n:][::-1]
        else:
            selected_indices = sorted_indices[:args.n]
        title_suffix = f"Top {args.n} {args.rank_type.title()} Power"

    elif args.mode == 'spatial':
        selected_indices = get_evenly_spaced_indices(x_all, y_all, active_indices, args.n)
        title_suffix = f"{args.n} Evenly Spaced Electrodes"

    print(f"Plotting electrodes: {selected_indices}")

    # --- 5. PLOTTING ---
    fps = 30.0 
    time_axis = np.arange(n_frames) / fps
    
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.15)
    
    ax_plot = plt.subplot(gs[0])
    ax_map = plt.subplot(gs[1])

    # Distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
    if len(selected_indices) > 10:
        colors = plt.cm.jet(np.linspace(0, 1, len(selected_indices)))

    # A. PLOT TIME SERIES (Left)
    for i, idx in enumerate(selected_indices):
        trace = power_data[:, idx]
        total_p = total_energy[idx]
        label_text = f"#{idx}"
        ax_plot.plot(time_axis, trace, color=colors[i], label=label_text, linewidth=1.5, alpha=0.9)

    ax_plot.set_title(f"Power Evolution: {title_suffix}", fontsize=12, fontweight='bold')
    ax_plot.set_xlabel("Time (s)")
    ax_plot.set_ylabel("Power (mW)")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_xlim(0, time_axis[-1])
    
    # B. PLOT SPATIAL LEGEND (Right)
    ax_map.scatter(x_all, y_all, c='lightgray', s=10, alpha=0.6, label='Grid')

    for i, idx in enumerate(selected_indices):
        ax_map.scatter(x_all[idx], y_all[idx], 
                       color=colors[i], s=50, edgecolors='black', linewidth=0.8, zorder=10)
        ax_map.text(x_all[idx] + 0.1, y_all[idx] + 0.1, str(idx), 
                    fontsize=8, fontweight='bold', color='black')

    ax_map.set_title("Electrode Selection Map", fontsize=12, fontweight='bold')
    ax_map.set_aspect('equal')
    ax_map.axis('off')
    
    if len(selected_indices) <= 15:
        ax_plot.legend(loc='upper right', framealpha=0.9, fontsize='small')
    
    plt.tight_layout()
    
    # --- SAVE FILE (Modified Logic) ---
    if args.mode == 'specific':
        # Create filename: power_specific_142.png or power_specific_142_143.png
        indices_str = "_".join(map(str, selected_indices))
        filename = f"power_specific_{indices_str}.png"
    else:
        # Default for other modes
        filename = f"analysis_{args.mode}.png"
        
    output_path = results_dir / filename
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
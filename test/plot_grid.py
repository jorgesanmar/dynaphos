import yaml
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_grid(yaml_path):
    path = Path(yaml_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"Loading {path.name}...")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    x = data['x']
    y = data['y']
    
    count = len(x)
    print(f"✅ Loaded {count} electrodes.")
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, s=10, c='navy', alpha=0.6, edgecolors='none')
    
    plt.title(f"Electrode Grid Layout: {path.stem}\n({count} Electrodes)", fontsize=14)
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save output
    output_file = path.parent / f"plot_{path.stem}.png"
    plt.savefig(output_file, dpi=150)
    print(f"🖼️  Plot saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    # Setup command line argument
    parser = argparse.ArgumentParser(description="Plot electrode coordinates from a YAML file.")
    parser.add_argument("file", nargs='?', default="config/grid_coords_dipole.yaml", 
                        help="Path to the .yaml grid file (default: config/grid_coords_dipole.yaml)")
    
    args = parser.parse_args()
    plot_grid(args.file)
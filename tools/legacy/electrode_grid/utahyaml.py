from pathlib import Path
import yaml
import numpy as np

# --- Grid parameters ---
n = 10
pitch_um = 400  # Utah typical spacing
pitch_mm = pitch_um / 1000.0

# Create positive-only coordinates (0 to 3.6 mm)
coords = np.arange(n) * pitch_mm

X, Y = np.meshgrid(coords, coords, indexing="xy")

x_list = X.ravel().tolist()
y_list = Y.ravel().tolist()

data = {
    "x": x_list,
    "y": y_list
}

# Save file (edit path if needed)
out_path = Path("grid_coords_utah.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print(f"Wrote {out_path.resolve()}")
print(f"Electrodes: {len(x_list)}")
print(f"x range (mm): {min(x_list):.3f} .. {max(x_list):.3f}")
print(f"y range (mm): {min(y_list):.3f} .. {max(y_list):.3f}")

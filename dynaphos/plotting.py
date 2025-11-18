from typing import Optional

from matplotlib import pyplot as plt

from dynaphos.cortex_models import get_cortex_coordinates_default, get_cortex_coordinates_grid


def plot_coordinates(params: dict, n_electrodes_x: int, n_electrodes_y: int,
                     x_max: Optional[int] = None):
    coordinates_cortex = get_cortex_coordinates_default(params)
    grid_cortex = get_cortex_coordinates_grid(params, n_electrodes_x,
                                              n_electrodes_y, x_max)
    plt.scatter(*coordinates_cortex.cartesian, c='r')
    plt.scatter(*grid_cortex.cartesian)
    plt.xlabel('Cortical distance (mm)')
    plt.ylabel('Cortical distance (mm)')
    plt.show()

def visualize_raster_pattern(self):
    """
    Visualize the current raster pattern.
    Displays a heatmap of the electrode array with colors representing different
    raster groups.
    """
    if not self.raster_enabled:
        print("Raster patterns are not enabled.")
        return
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Reshape to 2D if needed
    if len(self.raster_groups_flat) < np.prod(self.electrode_array_shape):
        # Pad with -1 for visualization
        padded = np.full(np.prod(self.electrode_array_shape), -1)
        padded[:len(self.raster_groups_flat)] = self.raster_groups_flat
        groups_2d = padded.reshape(self.electrode_array_shape)
    else:
        groups_2d = self.raster_groups_flat[:np.prod(self.electrode_array_shape)]
        groups_2d = groups_2d.reshape(self.electrode_array_shape)
    
    im = ax.imshow(groups_2d, cmap='tab10', vmin=0, vmax=self.raster_num_groups-1)
    ax.set_title(f'{self.raster_pattern.capitalize()} Raster Pattern\n'
                    f'{self.raster_num_groups} groups @ {self.raster_rate_hz} Hz',
                    fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, self.electrode_array_shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, self.electrode_array_shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.colorbar(im, ax=ax, label='Group Number')
    plt.tight_layout()
    plt.show()
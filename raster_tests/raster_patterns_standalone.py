"""
Standalone raster pattern module for testing without full dependencies.
"""

import numpy as np
from typing import Tuple, Optional


def create_raster_groups(
    array_shape: Tuple[int, int],
    num_groups: int,
    pattern: str = 'checkerboard',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Divide an electrode array into groups for sequential activation (rastering).
    
    Parameters
    ----------
    array_shape : Tuple[int, int]
        Shape of the electrode array (rows, cols). E.g., (10, 10) for a 10x10 grid.
    num_groups : int
        Number of raster groups to create.
    pattern : str
        Type of raster pattern: 'horizontal', 'vertical', 'checkerboard', 'random'
    seed : int, optional
        Random seed for reproducible random patterns.
    
    Returns
    -------
    raster_groups : np.ndarray
        Array where each element indicates the group number (0 to num_groups-1).
    """
    rows, cols = array_shape
    total_electrodes = rows * cols
    
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > total_electrodes:
        raise ValueError(f"num_groups ({num_groups}) cannot exceed total electrodes ({total_electrodes})")
    
    raster_groups = np.zeros(array_shape, dtype=int)
    
    if pattern == 'horizontal':
        rows_per_group = rows / num_groups
        for i in range(rows):
            group_idx = min(int(i / rows_per_group), num_groups - 1)
            raster_groups[i, :] = group_idx
            
    elif pattern == 'vertical':
        cols_per_group = cols / num_groups
        for j in range(cols):
            group_idx = min(int(j / cols_per_group), num_groups - 1)
            raster_groups[:, j] = group_idx
            
    elif pattern == 'checkerboard':
        # Checkerboard pattern with maximal spatial separation
        if num_groups >= 4:
            for i in range(rows):
                for j in range(cols):
                    offset = (i % 2) * (num_groups // 2)
                    group_idx = ((j % num_groups) + offset) % num_groups
                    raster_groups[i, j] = group_idx
        else:
            # Simple checkerboard for fewer groups
            for i in range(rows):
                for j in range(cols):
                    group_idx = (i * cols + j) % num_groups
                    raster_groups[i, j] = group_idx
                    
    elif pattern == 'random':
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        flat_groups = np.arange(total_electrodes) % num_groups
        rng.shuffle(flat_groups)
        raster_groups = flat_groups.reshape(array_shape)
        
    else:
        raise ValueError(f"Unknown pattern type: {pattern}. "
                        f"Choose from 'horizontal', 'vertical', 'checkerboard', or 'random'")
    
    return raster_groups

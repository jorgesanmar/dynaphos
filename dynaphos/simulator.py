import math
from typing import Optional, Tuple, Union

import logging
import numpy as np
import torch
import warnings

from dynaphos.cortex_models import get_cortical_magnification
from dynaphos.image_processing import scale_image, to_n_dim
from dynaphos.utils import (to_tensor, get_data_kwargs, get_truncated_normal,
                            get_deg2pix_coeff, set_deterministic,
                            print_stats, sigmoid, to_numpy, Map)

def create_raster_groups(
    array_shape: Tuple[int, int],
    num_groups: int,
    pattern: str = 'checkerboard',
    seed: Optional[int] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create raster groups as PyTorch tensor.
    
    Returns
    -------
    raster_groups : torch.Tensor
        Integer tensor where each element indicates group number of each electrode (0 to num_groups-1).
    """
    rows, cols = array_shape
    total_electrodes = rows * cols
    
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > total_electrodes:
        raise ValueError(f"num_groups ({num_groups}) cannot exceed total electrodes ({total_electrodes})")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    raster_groups = torch.zeros(array_shape, dtype=torch.long, device=device) # Initialize group assignment torch tensor
    
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
        if num_groups >= 4:
            for i in range(rows):
                for j in range(cols):
                    offset = (i % 2) * (num_groups // 2)
                    group_idx = ((j % num_groups) + offset) % num_groups
                    raster_groups[i, j] = group_idx
        else:
            for i in range(rows):
                for j in range(cols):
                    group_idx = (i * cols + j) % num_groups
                    raster_groups[i, j] = group_idx
                    
    elif pattern == 'random':
        flat_groups = torch.arange(total_electrodes, device=device) % num_groups
        perm = torch.randperm(total_electrodes, device=device)
        flat_groups = flat_groups[perm]
        raster_groups = flat_groups.reshape(array_shape)
        
    else:
        raise ValueError(f"Unknown pattern type: {pattern}")
    
    return raster_groups
class State:
    def __init__(self, params: dict, shape: Tuple[int, ...],
                 verbose: Optional[bool] = False):
        self.params = params
        self.shape = shape
        self.verbose = verbose
        self.state = None
        self.data_kwargs = get_data_kwargs(self.params)

        self.reset()

    def reset(self):
        self.state = torch.zeros(self.shape, **self.data_kwargs)

    def get(self) -> torch.Tensor:
        return self.state

    def update(self, x: torch.Tensor):
        raise NotImplementedError

    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return to_tensor(x, **self.data_kwargs)


class Activation(State):
    def __init__(self, params: dict, shape: Tuple[int, ...],
                 verbose: Optional[bool] = False):
        super().__init__(params, shape, verbose)

        self.fps = self.to_tensor(self.params['run']['fps'])
        # By default, the stimulus lasts as long as a frame. Can be adjusted:
        self.rel_stim_duration = self.to_tensor(
            self.params['default_stim']['relative_stim_duration'])
        # Convert decay-per-second to exponential decay constant.
        self.decay_rate = -torch.log(self.to_tensor(
            self.params['temporal_dynamics']['activation_decay_per_second']))
        self.num_steps = int(torch.ceil(self.decay_rate / self.fps))

    def update(self, x: torch.Tensor):
        """Update activation with leaky integrator.

        :param x: Effective stimulation current.
        """

        # If decay rate > frame rate, perform extra simulation steps for
        # numerical stability.
        for _ in range(self.num_steps):
            # eq: \Delta A = (-\gamma * A + I) * \Delta t
            self.state = self.state.detach() + (
                (-self.state.detach() * self.decay_rate + x * self.rel_stim_duration) /
                self.fps / self.num_steps)

        print_stats('activation', self.state, self.verbose)

class ActivationThreshold(State):
    def __init__(self, params: dict, shape: Tuple[int, ...],
                 rng: np.random.Generator, verbose: Optional[bool] = False):
        super().__init__(params, shape, verbose)
        self.rng = rng
        self._mu = self.params['thresholding']['activation_threshold']
        self._sd = self.params['thresholding']['activation_threshold_sd']
        self.reinitialize()

    def reinitialize(self, activation_thresholds: Optional[np.ndarray] = None):
        """Set or re-initialize the activation thresholds for each electrode. Default: sample from truncated random
        normal distribution."""
        if self.params['thresholding']['use_threshold']:
            if activation_thresholds is None:
                activation_thresholds = self.rng.standard_normal(self.shape) * self._sd + self._mu
            self.state = self.to_tensor(activation_thresholds).clip(0, None)
        else:
            self.state = torch.zeros(self.shape, **self.data_kwargs)

class Trace(State):
    def __init__(self, params: dict, shape: Tuple[int, ...],
                 verbose: Optional[bool] = False):
        super().__init__(params, shape, verbose)

        fps = self.params['run']['fps']

        # By default, the stimulus lasts as long as a frame. Can be adjusted:
        rel_stim_duration = \
            self.params['default_stim']['relative_stim_duration']

        # Convert decay-per-second to exponential decay constant.
        decay_rate = -math.log(
            self.params['temporal_dynamics']['trace_decay_per_second'])

        # Scaling of the trace increment.
        scale = self.params['temporal_dynamics']['trace_increase_rate']

        # If decay rate > frame rate, perform extra simulation steps for
        # numerical stability.
        self.num_steps = math.ceil(decay_rate / fps)

        self._a = self.to_tensor(-decay_rate / fps / self.num_steps)
        self._b = self.to_tensor(rel_stim_duration * scale / fps /
                                 self.num_steps)

    def update(self, x: torch.Tensor):
        """Update memory trace using a leaky integrator.

        :param x: Effective stimulation current.
        """
        for _ in range(self.num_steps):
            self.state = self.state.detach() + self._a * self.state.detach() + self._b * x

        print_stats('trace', self.state, self.verbose)


class Brightness(State):
    def __init__(self, params: dict, shape: Tuple[int, ...], verbose: Optional[bool] = False):
        super().__init__(params, shape, verbose)
        self.slope = self.to_tensor(
            self.params['brightness_saturation']['slope_brightness'])
        self.cps_half = self.to_tensor(
            self.params['brightness_saturation']['cps_half'])

    def update(self, x: torch.Tensor):
        """Saturate activation values."""

        self.state = sigmoid(self.slope * (x - self.cps_half))
        print_stats('sigmoided activation', self.state, self.verbose)


class Sigma(State):
    def __init__(self, params: dict, shape: Tuple[int, ...],
                 magnification: torch.Tensor, verbose: Optional[bool] = False):
        super().__init__(params, shape, verbose)

        p = self.params['size']
        if p['size_equation'] == 'sqrt':  # Tehovnik 2007
            def f(x):
                return torch.sqrt(torch.div(x, p['current_spread']))
        elif p['size_equation'] == 'sigmoid':  # Bosking et al., 2017
            def f(x):
                return 0.5 * p['MD'] * sigmoid(p['slope_size'] *
                                               (x - p['I_half']))
        else:
            raise ValueError("Size equation should be 'sqrt' or 'sigmoid'.")
        self.f = f
        self.scale = p['radius_to_sigma'] / magnification
    def update(self, x: torch.Tensor):
        """Compute the effect of the input current on phosphene size."""

        # Current spread to sigma in pixels.
        self.state = torch.mul(self.f(x), self.scale)

        print_stats('Sigma (in degrees)', self.state, self.verbose)


class GaussianSimulator:
    def __init__(self, params: dict, coordinates: Map,
                 rng: Optional[np.random.Generator] = None, 
                 theta: Optional[np.ndarray] = None,
                 raster_enabled: bool = False,
                 raster_pattern: str = 'checkerboard',
                 raster_num_groups: int = 5,
                 raster_rate_hz: float = 4.5):
        """
        Initialize a simulator
        :param params: Dictionary of simulation parameters.
        :param coordinates: Coordinates of phosphenes.
        :param rng: Random number generator.
        :param theta: Orientations for gabor filtering (if 'gabor_filtering' set to True)
        :param raster_enabled: Whether to enable raster pattern stimulation.
        :param raster_pattern: Raster pattern type ('horizontal', 'vertical', 'checkerboard', 'random').
        :param raster_num_groups: Number of raster groups.
        :param raster_rate_hz: Overall raster rate in Hz.
        """

        self.params = params
        self.data_kwargs = get_data_kwargs(self.params)

        rng = np.random.default_rng() if rng is None else rng
        set_deterministic(self.params['run']['seed'])

        self.deg2pix_coeff = get_deg2pix_coeff(self.params['run'])

        self.phosphene_maps = \
            self.generate_phosphene_maps(coordinates, theta=theta)

        batch_size = self.params['run']['batch_size']
        if batch_size != 0:
            self.shape = (batch_size, self.num_phosphenes, 1, 1)
            self._electrode_dimension = 1
        else:
            self.shape = (self.num_phosphenes, 1, 1)
            self._electrode_dimension = 0

        r, phi = coordinates.polar
        r = torch.reshape(self.to_tensor(r), self.shape[-3:])
        self.magnification = get_cortical_magnification(
            r, self.params['cortex_model'])

        verbose = self.params['run']['print_stats']
        self.activation = Activation(params, self.shape, verbose=verbose)
        self.trace = Trace(params, self.shape)
        self.sigma = Sigma(params, self.shape, self.magnification)
        self.brightness = Brightness(params, self.shape)
        self.threshold = ActivationThreshold(params, self.shape, rng)
        self.effective_charge_per_second = None

        # Pre-allocate some helper variables
        self._sampling_mask = None
        self._phosphene_centers = None
        params_sampling = self.params['sampling']
        self._sampling_method = params_sampling['sampling_method']
        self._sqrt_pi_inv = 1 / torch.sqrt(self.to_tensor(torch.pi))
        self._pulse_width = (self.params['default_stim']['pw_default'] *
                             torch.ones(self.shape, **self.data_kwargs))
        self._frequency = (self.params['default_stim']['freq_default'] *
                           torch.ones(self.shape, **self.data_kwargs))
        
        self._zero = self.to_tensor(0)
        self._inf = self.to_tensor(torch.inf)

        # ===== RASTER PATTERN INITIALIZATION =====
        self.raster_enabled = raster_enabled
        self.raster_pattern = raster_pattern
        self.raster_num_groups = raster_num_groups
        self.raster_rate_hz = raster_rate_hz
        
        # Calculate per-group activation rate
        fps = self.params['run']['fps']
        self.raster_group_interval_s = 1.0 / (raster_num_groups * raster_rate_hz)
        
        # Infer electrode array shape from coordinates
        # Assume square grid for now (approximate if necessary)
        num_electrodes = self.num_phosphenes
        grid_size = int(np.sqrt(num_electrodes))
        if grid_size * grid_size != num_electrodes:
            # Non-square array - use best approximation
            import math
            rows = int(math.sqrt(num_electrodes))
            cols = int(math.ceil(num_electrodes / rows))
            self.electrode_array_shape = (rows, cols)
        else:
            self.electrode_array_shape = (grid_size, grid_size)
        
        
        # Create raster groups based on PHYSICAL COORDINATES
        if self.raster_enabled:
            # 1. Get Cartesian coordinates of all electrodes
            x_coords, y_coords = coordinates.cartesian
            # Normalize coordinates to 0-1 range for binning
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Avoid division by zero if all points are identical
            w = (x_max - x_min) if x_max != x_min else 1.0
            h = (y_max - y_min) if y_max != y_min else 1.0
            
            x_norm = (x_coords - x_min) / w
            y_norm = (y_coords - y_min) / h
            
            # Initialize groups array
            groups_np = np.zeros(self.num_phosphenes, dtype=int)
            
            if self.raster_pattern == 'horizontal':
                # Group based on Y position (Top to Bottom)
                # We invert y because typically y is positive up, but raster scans top-down
                # Check your coordinate system. Assuming standard Cartesian:
                groups_np = np.floor((1.0 - y_norm) * self.raster_num_groups).astype(int)
                
            elif self.raster_pattern == 'vertical':
                # Group based on X position (Left to Right)
                groups_np = np.floor(x_norm * self.raster_num_groups).astype(int)
                
            elif self.raster_pattern == 'checkerboard':
                # Create a virtual grid for checkerboard assignment
                # We use sqrt(num_groups) to approximate the grid density
                grid_dim = int(np.sqrt(self.num_phosphenes)) 
                
                # Determine row and col indices for each point
                row_idx = np.floor((1.0 - y_norm) * grid_dim).astype(int)
                col_idx = np.floor(x_norm * grid_dim).astype(int)
                
                if self.raster_num_groups >= 4:
                    # Complex checkerboard logic
                    offset = (row_idx % 2) * (self.raster_num_groups // 2)
                    groups_np = ((col_idx % self.raster_num_groups) + offset) % self.raster_num_groups
                else:
                    # Simple interleave
                    groups_np = (row_idx + col_idx) % self.raster_num_groups
                    
            elif self.raster_pattern == 'random':
                rng_local = np.random.default_rng(self.params['run']['seed'])
                groups_np = rng_local.integers(0, self.raster_num_groups, size=self.num_phosphenes)

            # Clamp to ensure no index goes out of bounds (e.g. 1.0 maps to num_groups)
            groups_np = np.clip(groups_np, 0, self.raster_num_groups - 1)
            
            # Convert to Tensor
            self.raster_groups_flat = self.to_tensor(groups_np).long()
            
            # Create schedule masks (one per group) - all operations on GPU
            self.raster_schedule = []
            for group_idx in range(self.raster_num_groups):
                mask = (self.raster_groups_flat == group_idx).float()
                mask = mask.reshape(self.shape[-3:])
                self.raster_schedule.append(mask)
            
            # Initialize raster state
            self.current_raster_group = 0
            self.raster_time_accumulator = 0.0
            
            # For random pattern, store when to reshuffle
            if self.raster_pattern == 'random':
                self.raster_reshuffle_interval = 5  # frames
                self.raster_frame_counter = 0
        else:
            self.raster_groups = None
            self.raster_groups_flat = None
            self.raster_schedule = None
            self.current_raster_group = 0
            self.raster_time_accumulator = 0.0

        # Cumulative charge guard
        safety = self.params.get('safety', {}) or {}
        self.enable_charge_guard = bool(safety.get('enable_charge_guard', True))
        self.charge_limit_uC = float(safety.get('cumulative_charge_limit_uC', 30.0))
        self.charge_warn_only = bool(safety.get('charge_warn_only', True))
        self.cumulative_charge_uC = torch.zeros(self.num_phosphenes, **self.data_kwargs)
        self._charge_guard_step = 0  
        self._log_every_steps = int(safety.get('charge_log_every', 30))
        
        rel_stim_duration = float(self.params['default_stim']['relative_stim_duration'])
        self._dt_s = rel_stim_duration / fps

        self.reset()

    @property
    def num_phosphenes(self):
        return len(self.phosphene_maps)

    def to_tensor(self, x: Union[int, float, np.ndarray]) -> torch.Tensor:
        return to_tensor(x, **self.data_kwargs)

    def reset(self):
        """Reset memory of previous timestep and raster state."""
        self.activation.reset()
        self.trace.reset()
        self.sigma.reset()
        self.reset_cumulative_charge()
        
        # Reset raster state
        if self.raster_enabled:
            self.current_raster_group = 0
            self.raster_time_accumulator = 0.0
            if self.raster_pattern == 'random':
                self.raster_frame_counter = 0
    
    def reset_cumulative_charge(self):
        """Reset cumulative charge accounting for all electrodes."""
        self.cumulative_charge_uC.zero_()
        self._charge_guard_step = 0 
    
    def get_charge_status(self):
        """Return current cumulative charge status for monitoring."""
        max_charge = self.cumulative_charge_uC.max().item()
        min_charge = self.cumulative_charge_uC.min().item()
        total_charge = self.cumulative_charge_uC.sum().item()
        max_electrode_idx = self.cumulative_charge_uC.argmax().item()
        
        return {
            'cumulative_charge_uC': self.cumulative_charge_uC, 
            'max_charge_uC': max_charge,
            'min_charge_uC': min_charge,
            'total_charge_uC': total_charge,
            'max_electrode_idx': max_electrode_idx,
            'limit_uC': self.charge_limit_uC
        }

    def gabor_rotation(self, x, y, theta=None) -> torch.Tensor:
        """Rotation of ellipsis."""
        num_phosphenes = len(x)
        if theta is None:
            theta = torch.mul(2 * math.pi, torch.rand((num_phosphenes, 1, 1), **self.data_kwargs)) # Random rotation
        else:
            theta = torch.reshape(self.to_tensor(theta), (-1, 1, 1))
        y_rotated = -x * torch.sin(theta) + y * torch.cos(theta)
        x_rotated = x * torch.cos(theta) + y * torch.sin(theta)
        gamma = self.params['gabor']['gamma']
        phosphene_maps = torch.sqrt(x_rotated ** 2 + y_rotated ** 2 * gamma ** 2)
        return phosphene_maps

    def generate_phosphene_maps(self, coordinates: Map,
                                remove_invalid: Optional[bool] = True,
                                theta: Optional[np.ndarray] = None,
                                ) -> torch.Tensor:
        """Generate phosphene maps (for each phosphene distance to each pixel).

        :param coordinates: Coordinates of phosphenes.
        :param remove_invalid: Whether to remove phosphenes out of view.
        :param theta: Orientations for gabor filtering (if 'gabor_filtering' set to True)
        :return: an (n_phosphenes x resolution[0] x resolution[1]) array
        describing distances from phosphene locations
        """

        # Phosphene coordinates
        x_coords, y_coords = coordinates.cartesian
        x_coords = torch.reshape(self.to_tensor(x_coords), (-1, 1, 1))
        y_coords = torch.reshape(self.to_tensor(y_coords), (-1, 1, 1))

        # x,y limits of the simulation
        res_x, res_y = self.params['run']['resolution']
        x_org, y_org = self.params['run']['origin']
        hemi_fov = self.params['run']['view_angle'] / 2
        x_min, x_max = x_org - hemi_fov, x_org + hemi_fov
        y_min, y_max = y_org - hemi_fov, y_org + hemi_fov
     
        if remove_invalid:
            # Check if phosphene locations are inside of view angle.
            valid = (
                torch.ge(x_coords, x_min) & torch.less(x_coords, x_max) &
                torch.ge(y_coords, y_min) & torch.less(y_coords, y_max)).ravel()
            num_total = len(x_coords)
            num_valid = torch.sum(valid)
            logging.debug(f"{num_total - num_valid} of {num_total} phosphenes "
                          f"are outside of view and will be removed.")
            x_coords = x_coords[valid]
            y_coords = y_coords[valid]
            coordinates.use_subset(to_numpy(valid))

        # Get distance maps to phosphene centres (in degrees of visual angle).
        device = self.data_kwargs['device']
        num_phosphenes = len(x_coords)

        x_range = torch.linspace(x_min, x_max, res_x, device=device)
        y_range = torch.linspace(y_min, y_max, res_y, device=device)

        grid = torch.meshgrid(x_range, y_range, indexing='xy')
        grid_x = torch.tile(grid[0], (num_phosphenes, 1, 1))
        grid_y = torch.tile(grid[1], (num_phosphenes, 1, 1))
        x = grid_x - x_coords
        y = grid_y - y_coords

        if self.params['gabor']['gabor_filtering']:
            phosphene_maps = self.gabor_rotation(x, y, theta)
        else:
            phosphene_maps = torch.sqrt(x ** 2 + y ** 2)

        return phosphene_maps

    def update(self, amplitude: torch.Tensor,
               pulse_width: Optional[torch.Tensor] = None,
               frequency: Optional[torch.Tensor] = None,
               dt: Optional[float] = None):
        """
        Update phosphene states with raster pattern support.
        
        Parameters
        ----------
        amplitude : torch.Tensor
            Stimulation amplitudes for each electrode.
        pulse_width : torch.Tensor, optional
            Stimulation pulse widths for each electrode.
        frequency : torch.Tensor, optional
            Stimulation frequencies for each electrode.
        dt : float, optional
            Time step in seconds. If None, uses 1/fps.
        """
        
        # Update raster timing
        self._update_raster_state(dt)
        
        # Get current raster mask
        raster_mask = self.get_current_raster_mask()
        
        # Apply raster mask to amplitude
        # Only electrodes in the current group receive stimulation
        masked_amplitude = amplitude.view(self.shape) * raster_mask
        
        if pulse_width is None:
            pulse_width = self._pulse_width
        if frequency is None:
            frequency = self._frequency

        charge_per_s = self.get_current(masked_amplitude,
                                        frequency.view(self.shape),
                                        pulse_width.view(self.shape))

        self.activation.update(charge_per_s)
        self.trace.update(charge_per_s)
        self.sigma.update(masked_amplitude)
        self.brightness.update(self.activation.get())

        # Cumulative charge guard 
        if self.enable_charge_guard:
            dt_s = self._dt_s if dt is None else dt
            dims = (0, 2, 3) if charge_per_s.dim() == 4 else (1, 2)
            i_avg_A_per_elec = charge_per_s.sum(dim=dims)
            delta_C = i_avg_A_per_elec * dt_s
            delta_uC = delta_C * 1e6
            self.cumulative_charge_uC = self.cumulative_charge_uC + delta_uC
            
            self._charge_guard_step += 1
            if self._log_every_steps > 0 and (self._charge_guard_step % self._log_every_steps == 0):
                max_uC = float(self.cumulative_charge_uC.max().item())
                import logging
                logging.info(f"[ChargeGuard] max_cum_charge_uC={max_uC:.1f} (limit={self.charge_limit_uC:.1f})")
            
            over = self.cumulative_charge_uC > self.to_tensor(self.charge_limit_uC)
            if torch.any(over):
                idx = torch.nonzero(over, as_tuple=False).view(-1).tolist()
                values = [float(self.cumulative_charge_uC[i]) for i in idx]
                msg = (f"[ChargeGuard] Cumulative charge limit exceeded for electrodes {idx}. "
                       f"limit={self.charge_limit_uC:.1f} µC, values={values}")
                if self.charge_warn_only:
                    import warnings
                    warnings.warn(msg)
                else:
                    raise RuntimeError(msg)



    def get_current(self, amplitude: torch.Tensor, frequency: torch.Tensor,
                    pulse_width: torch.Tensor) -> torch.Tensor:
        """Caclulate effective current (charge per second) from the square wave
        pulse. Cannot be negative.

        :param amplitude: Stimulation amplitudes for each electrode.
        :param pulse_width: Stimulation pulse widths for each electrode.
        :param frequency: Stimulation frequencies for each electrode.
        """

        leak_current = \
            self.trace.get() + self.params['thresholding']['rheobase']
        charge_per_s = torch.relu((amplitude - leak_current) *
                                  pulse_width * frequency)
        self.effective_charge_per_second = charge_per_s

        print_stats('charge per second', charge_per_s)

        return charge_per_s

    def gaussian_activation(self) -> torch.Tensor:
        """Generate gaussian activation maps, based on sigmas and phosphene
        mapping.

        :return: Stack of Gaussian-shaped phosphene images
        (n_phosphenes, resolution_y, resolution_x)
        """

        # Calculate normalized Gaussian (peak has value 1).
        sigma = self.sigma.get().clamp(1e-22, None)  # TODO: clamping redundant? Default division by zero gives inf.
        exp = torch.exp(-0.5 * (self.phosphene_maps / sigma) ** 2)
        return exp

    def get_state(self):
        state = {
            'brightness': self.brightness.get(),
            'sigma': self.sigma.get(),
            'activation': self.activation.get(),
            'trace': self.trace.get(),
            'threshold': self.threshold.get(),
            'effective_charge_per_second':
                self.effective_charge_per_second}
        return state

    def __call__(self, amplitude, pulse_width=None, frequency=None):
        self.update(amplitude, pulse_width, frequency)
        activation = self.gaussian_activation()
        supra_threshold = torch.greater(self.activation.get(), self.threshold.get())
        intensity = torch.where(supra_threshold, self.brightness.get(), self._zero)
        
        # Apply raster mask to zero out inactive electrodes
        if self.raster_enabled:
            raster_mask = self.get_current_raster_mask()
            # Expand mask to match activation shape (add spatial dimensions)
            # mask shape: (n_electrodes, 1, 1)
            intensity = intensity * raster_mask
        
        return torch.sum(intensity * activation, dim=self._electrode_dimension).clamp(0, 1)

    @property
    def phosphene_centers(self):
        """Indices (flat indexing) of the phosphene centers"""
        if self._phosphene_centers is None:
            self._phosphene_centers = self.phosphene_maps.flatten(start_dim=1).argmin(dim=-1)
        return self._phosphene_centers

    def sample_centers(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the value of the activation mask at the center pixel of each phosphene"""
        return x.flatten(-2)[..., self.phosphene_centers]

    def sample_receptive_fields(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the maximum value of activation mask x within the 'receptive field' of each phosphene"""
        return torch.amax(self.sampling_mask * x, dim=(-2,-1))

    @property
    def sampling_mask(self):
        """Boolean mask (tensor) that defines which pixels are inside the receptive field / center of each phosphene"""
        if self._sampling_mask is None:
            params = self.params['sampling']
            if self._sampling_method == 'receptive_fields':
                self._sampling_mask = torch.less(self.phosphene_maps, params['RF_size'] / self.magnification)
            elif self._sampling_method == 'center':
                # Sampling mask is not used anymore in 'center' mode (pixels are directly retrieved using indexing),
                # but still implemented here for backwards compatibility
                p_map = self.phosphene_maps
                flat_idx = torch.arange(p_map.shape[0], device=p_map.device) * p_map.shape[-2] * p_map.shape[-1]
                self._sampling_mask = torch.zeros_like(p_map)
                self._sampling_mask.flatten()[self.phosphene_centers + flat_idx] = 1
            else:
                raise NotImplementedError
        return self._sampling_mask

    def sample_stimulus(self, activation_mask: Union[np.ndarray, torch.Tensor], rescale=False,
                        ) -> torch.Tensor:
        """Obtain a stimulation vector from an activation mask image that indicates the regional stimulation intensity.

        param activation_mask: Image (or batch of images: N, 1, H, W). The pixel intensities indicate the
                                stimulation amplitude for each visual region.

        param rescale: If False (default), the pixel intensities indicate the stimulation amplitude in Amperes.
                        If True, the input pixels (in range [0, 1] or [0, 255]) are mapped to stimulation amplitudes
                        using the default stimulus scale parameter specified in the params configuration file.

        return: Stimulation tensor with the stimulation amplitudes for each phosphene. """

        if isinstance(activation_mask, np.ndarray):
            dtype = activation_mask.dtype
            activation_mask = self.to_tensor(activation_mask)
            if (dtype == np.dtype('uint8')) or (activation_mask.max() > 1):
                activation_mask = scale_image(activation_mask, 1 / 255)
        if self._sampling_method == 'receptive_fields':
            electrode_activation = self.sample_receptive_fields(activation_mask)
        elif self._sampling_method == 'center':
            electrode_activation = self.sample_centers(activation_mask)  # electrode activations between 0 and 1
        else:
            raise NotImplementedError
        if rescale:
            electrode_activation = torch.mul(electrode_activation, self.params['sampling']['stimulus_scale'])
        elif electrode_activation.max() >= 1e-3:
            warnings.warn("High values detected! Activation mask not longer rescaled as default behaviour. Please set "
                          "rescale=True to map pixels in range [0, 1] or [0, 255] to the default stimulus scale.",
                          category=DeprecationWarning, stacklevel=2)
        return electrode_activation
    
    def get_raster_info(self) -> dict:
        """
        Get information about the current raster configuration.
        
        Returns
        -------
        info : dict
            Dictionary containing raster pattern information.
        """
        if not self.raster_enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'pattern': self.raster_pattern,
            'num_groups': self.raster_num_groups,
            'raster_rate_hz': self.raster_rate_hz,
            'group_interval_s': self.raster_group_interval_s,
            'current_group': self.current_raster_group,
            'array_shape': self.electrode_array_shape,
        }
    
    def _update_raster_state(self, dt: Optional[float] = None):
            """
            Update the current active raster group based on elapsed time.
            
            Parameters
            ----------
            dt : float, optional
                Time elapsed since last update in seconds. If None, uses
                1/fps from params.
            """
            if not self.raster_enabled:
                return
            
            if dt is None:
                dt = 1.0 / self.params['run']['fps']
            
            # Accumulate time
            self.raster_time_accumulator += dt
            
            # Check if we should advance to next group
            while self.raster_time_accumulator >= self.raster_group_interval_s:
                self.raster_time_accumulator -= self.raster_group_interval_s
                self.current_raster_group = (self.current_raster_group + 1) % self.raster_num_groups
                
                # Handle random pattern reshuffling
                if self.raster_pattern == 'random':
                    self.raster_frame_counter += 1
                    if self.raster_frame_counter >= self.raster_reshuffle_interval:
                        self.raster_frame_counter = 0
                        # Reshuffle groups
                        self._reshuffle_random_pattern()
    
    def _reshuffle_random_pattern(self):
        """Reshuffle using pure PyTorch."""
        num_electrodes = len(self.raster_groups_flat)
        
        # Create new random assignment on device
        new_groups = torch.arange(num_electrodes, device=self.raster_groups_flat.device) % self.raster_num_groups
        perm = torch.randperm(num_electrodes, device=self.raster_groups_flat.device)
        new_groups = new_groups[perm]
        self.raster_groups_flat = new_groups
        
        # Update schedule masks (all on GPU)
        for group_idx in range(self.raster_num_groups):
            mask = (self.raster_groups_flat == group_idx).float()
            mask = mask.reshape(self.shape[-3:])
            self.raster_schedule[group_idx] = mask

    def get_current_raster_mask(self) -> torch.Tensor:
        """
        Get the current active electrode mask based on raster state.
        
        Returns
        -------
        mask : torch.Tensor
            Binary mask indicating which electrodes are currently active.
            Shape matches self.shape.
        """
        if not self.raster_enabled:
            # All electrodes active
            return torch.ones(self.shape, **self.data_kwargs)
        
        # Return mask for current group
        mask = self.raster_schedule[self.current_raster_group]
        
        # Broadcast to full shape if needed
        if len(self.shape) == 4:  # With batch dimension
            mask = mask.unsqueeze(0).expand(self.shape)
        else:
            mask = mask.expand(self.shape)
        
        return mask
            
            

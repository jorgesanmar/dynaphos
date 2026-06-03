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


_PSEUDO_RANDOM_RASTER_ALIASES = {
    "random",
    "pseudo_random",
    "pseudo-random",
    "pseudorandom",
}


def _normalize_raster_pattern(pattern: str) -> str:
    normalized = str(pattern).strip().lower().replace(" ", "_")
    if normalized in _PSEUDO_RANDOM_RASTER_ALIASES:
        return "random"
    return normalized


def normalize_raster_mode(mode: str) -> str:
    """Normalize user-facing raster mode aliases."""
    normalized = str(mode).strip().lower().replace(" ", "_")
    if normalized in _PSEUDO_RANDOM_RASTER_ALIASES:
        return "random"
    return normalized


def apply_appearance_threshold(stim_raw: torch.Tensor, threshold_a: float) -> torch.Tensor:
    """Gate stimulation amplitudes below the appearance threshold."""
    return torch.where(stim_raw >= float(threshold_a), stim_raw, torch.zeros_like(stim_raw))


def compute_raster_timing(video_fps: float, groups: int, raster_enabled: bool) -> dict[str, float]:
    """Return raster cycle and group stepping rates synchronized to video FPS."""
    if float(video_fps) <= 0:
        raise ValueError(f"Video FPS must be > 0, got {video_fps}.")
    if int(groups) <= 0:
        raise ValueError(f"Raster groups must be > 0, got {groups}.")

    if not raster_enabled:
        return {
            "video_fps": float(video_fps),
            "cycle_rate_hz": 0.0,
            "group_step_rate_hz": 0.0,
            "group_interval_s": 0.0,
        }
    return {
        "video_fps": float(video_fps),
        "cycle_rate_hz": float(video_fps) / float(groups),
        "group_step_rate_hz": float(video_fps),
        "group_interval_s": 1.0 / float(video_fps),
    }


def _rng_permutation(size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        return np.random.permutation(size)
    return rng.permutation(size)


def _rng_random(rng: Optional[np.random.Generator] = None) -> float:
    if rng is None:
        return float(np.random.random())
    return float(rng.random())


def _balanced_group_targets(num_points: int, num_groups: int,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    target_counts = np.full(num_groups, num_points // num_groups, dtype=np.int64)
    remainder = num_points % num_groups
    if remainder > 0:
        target_counts[_rng_permutation(num_groups, rng)[:remainder]] += 1
    return target_counts


def _normalize_raster_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    spans = np.ptp(points, axis=0)
    spans[spans <= 1e-12] = 1.0
    return (points - mins) / spans


def _estimate_local_spacing(points: np.ndarray) -> float:
    spacings = []
    for dim in range(points.shape[1]):
        levels = np.unique(np.round(points[:, dim], decimals=12))
        diffs = np.diff(np.sort(levels))
        positive_diffs = diffs[diffs > 1e-12]
        if positive_diffs.size > 0:
            spacings.append(float(np.median(positive_diffs)))

    if spacings:
        return max(float(np.median(spacings)), 1e-12)
    if len(points) <= 1:
        return 1.0

    max_sample = 1024
    if len(points) > max_sample:
        sample_idx = np.linspace(0, len(points) - 1, max_sample, dtype=np.int64)
        sample = points[sample_idx]
    else:
        sample = points
    delta = sample[:, None, :] - points[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    dist2[dist2 <= 1e-24] = np.inf
    nearest = np.sqrt(np.min(dist2, axis=1))
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 1.0
    return max(float(np.median(nearest)), 1e-12)


def _pseudo_random_spaced_groups(points: np.ndarray, num_groups: int,
                                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Balanced pseudo-random grouping with a penalty for local same-group clusters."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be a 2D array with x/y coordinates.")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")

    num_points = len(points)
    if num_points == 0:
        raise ValueError("At least one coordinate is required for raster grouping.")

    points_norm = _normalize_raster_points(points)
    spacing = _estimate_local_spacing(points_norm)
    hard_radius2 = (spacing * 1.05) ** 2
    soft_radius2 = max((spacing * 2.0) ** 2, 1e-12)

    groups = np.full(num_points, -1, dtype=np.int64)
    target_counts = _balanced_group_targets(num_points, num_groups, rng)
    counts = np.zeros(num_groups, dtype=np.int64)
    assigned_indices = [[] for _ in range(num_groups)]

    for point_idx in _rng_permutation(num_points, rng):
        available_groups = np.flatnonzero(counts < target_counts)
        if available_groups.size == 0:
            available_groups = np.arange(num_groups, dtype=np.int64)

        point = points_norm[point_idx]
        group_costs = []
        for group_idx in available_groups:
            if assigned_indices[group_idx]:
                same_group = points_norm[np.asarray(assigned_indices[group_idx], dtype=np.int64)]
                delta = same_group - point
                dist2 = np.sum(delta * delta, axis=1)
                hard_conflicts = np.count_nonzero(dist2 <= hard_radius2)
                local_heat = hard_conflicts * 1000.0 + float(np.exp(-dist2 / soft_radius2).sum())
            else:
                local_heat = 0.0

            target = max(int(target_counts[group_idx]), 1)
            fill_ratio = float(counts[group_idx]) / target
            group_costs.append(local_heat + fill_ratio * 1e-3 + _rng_random(rng) * 1e-9)

        chosen_group = int(available_groups[int(np.argmin(group_costs))])
        groups[point_idx] = chosen_group
        counts[chosen_group] += 1
        assigned_indices[chosen_group].append(int(point_idx))

    return groups


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

    pattern = _normalize_raster_pattern(pattern)
    rng = np.random.default_rng(seed) if seed is not None else None

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
        yy, xx = np.indices(array_shape)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        flat_groups = _pseudo_random_spaced_groups(points, num_groups, rng=rng)
        raster_groups = torch.as_tensor(flat_groups, dtype=torch.long, device=device).reshape(array_shape)

    else:
        raise ValueError(f"Unknown pattern type: {pattern}")

    return raster_groups


def _cluster_coordinate_levels(values: np.ndarray) -> np.ndarray:
    """Map 1D coordinates to discrete level indices using data-driven clustering."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Expected a 1D array of coordinates.")
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    if values.size == 1:
        return np.zeros(1, dtype=np.int64)

    order = np.argsort(values)
    sorted_values = values[order]
    diffs = np.diff(sorted_values)
    positive_diffs = diffs[diffs > 1e-9]

    if positive_diffs.size == 0:
        labels_sorted = np.zeros(values.size, dtype=np.int64)
    else:
        tol = max(float(np.median(positive_diffs)) * 0.5, 1e-9)
        labels_sorted = np.zeros(values.size, dtype=np.int64)
        level = 0
        for idx, diff in enumerate(diffs, start=1):
            if diff > tol:
                level += 1
            labels_sorted[idx] = level

    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def _coordinate_based_raster_groups(x_coords: np.ndarray, y_coords: np.ndarray,
                                    pattern: str, num_groups: int,
                                    device: Union[str, torch.device],
                                    rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Assign raster groups from Cartesian electrode coordinates."""
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    pattern = _normalize_raster_pattern(pattern)

    if x_coords.shape != y_coords.shape:
        raise ValueError("x_coords and y_coords must have the same shape.")
    if x_coords.ndim != 1:
        raise ValueError("Raster coordinates must be 1D.")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")

    num_points = x_coords.size
    if num_points == 0:
        raise ValueError("At least one coordinate is required for raster grouping.")

    col_idx = _cluster_coordinate_levels(x_coords)
    row_idx = _cluster_coordinate_levels(-y_coords)  # highest y is top row

    if pattern == 'horizontal':
        n_rows = int(row_idx.max()) + 1
        rows_per_group = max(n_rows / num_groups, 1.0)
        groups = np.floor(row_idx / rows_per_group).astype(np.int64)
    elif pattern == 'vertical':
        n_cols = int(col_idx.max()) + 1
        cols_per_group = max(n_cols / num_groups, 1.0)
        groups = np.floor(col_idx / cols_per_group).astype(np.int64)
    elif pattern == 'checkerboard':
        if num_groups >= 4:
            offset = (row_idx % 2) * max(num_groups // 2, 1)
            groups = (col_idx + offset) % num_groups
        else:
            groups = (row_idx + col_idx) % num_groups
    elif pattern == 'random':
        points = np.column_stack([x_coords, y_coords])
        groups = _pseudo_random_spaced_groups(points, num_groups, rng=rng)
    else:
        raise ValueError(f"Unknown pattern type: {pattern}")

    groups = np.clip(groups, 0, num_groups - 1)
    return torch.as_tensor(groups, dtype=torch.long, device=device)








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



import numpy as np
import torch






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
                 raster_coordinates: Optional[Map] = None,
                 raster_enabled: bool = False,
                 raster_pattern: str = 'checkerboard',
                 raster_num_groups: int = 5,
                 raster_rate_hz: float = 4.5,
                 phosphene_mode: str = 'visual'):
        """
        Initialize a simulator
        :param params: Dictionary of simulation parameters.
        :param coordinates: Coordinates of phosphenes.
        :param rng: Random number generator.
        :param theta: Orientations for gabor filtering (if 'gabor_filtering' set to True)
        :param raster_coordinates: Electrode-space coordinates used for raster grouping.
            If None, the simulator falls back to `coordinates`.
        :param raster_enabled: Whether to enable raster pattern stimulation.
        :param raster_pattern: Raster pattern type ('horizontal', 'vertical', 'checkerboard', 'random').
        :param raster_num_groups: Number of raster groups.
        :param raster_rate_hz: Overall raster rate in Hz.
        :param phosphene_mode: 'visual' builds full phosphene distance maps for
            rendering. 'safety_centers' skips those maps and samples only at
            phosphene center pixels for lower-memory safety analysis.
        """

        self.params = params
        self.data_kwargs = get_data_kwargs(self.params)
        self.phosphene_mode = str(phosphene_mode).strip().lower()
        if self.phosphene_mode not in {'visual', 'safety_centers'}:
            raise ValueError("phosphene_mode must be 'visual' or 'safety_centers'.")

        rng = np.random.default_rng() if rng is None else rng
        set_deterministic(self.params['run']['seed'])

        self.deg2pix_coeff = get_deg2pix_coeff(self.params['run'])

        if self.phosphene_mode == 'visual':
            self.phosphene_maps = self.generate_phosphene_maps(
                coordinates,
                theta=theta,
                raster_coordinates=raster_coordinates,
            )
        else:
            self.phosphene_maps = None
            self._initialize_lightweight_phosphene_geometry(
                coordinates,
                raster_coordinates=raster_coordinates,
            )
        self._num_phosphenes = int(len(self.electrode_tags))

        batch_size = self.params['run']['batch_size']
        if batch_size != 0:
            self.shape = (batch_size, self.num_phosphenes, 1, 1)
            self._electrode_dimension = 1
        else:
            self.shape = (self.num_phosphenes, 1, 1)
            self._electrode_dimension = 0

        x_sorted, y_sorted = self._sorted_cartesian_coords
        r, _ = Map(x=x_sorted, y=y_sorted).polar
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
        self.delivered_amplitude = torch.zeros(self.shape, **self.data_kwargs)
        self.delivered_charge_per_second = torch.zeros(self.shape, **self.data_kwargs)
        self.current_pulse_width = torch.zeros(self.shape, **self.data_kwargs)
        self.current_frequency = torch.zeros(self.shape, **self.data_kwargs)

        # Pre-allocate some helper variables
        self._sampling_mask = None
        self._phosphene_centers = None
        self._warned_lightweight_receptive_fields = False
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
        self.raster_pattern = _normalize_raster_pattern(raster_pattern)
        self.raster_num_groups = raster_num_groups
        self.raster_rate_hz = raster_rate_hz

        # Calculate per-group activation rate
        fps = self.params['run']['fps']
        if self.raster_enabled:
        # Only calculate interval if we are actually rastering
            if raster_rate_hz > 0:
                self.raster_group_interval_s = 1.0 / (raster_num_groups * raster_rate_hz)
            else:
                self.raster_group_interval_s = float('inf') # Prevent division by zero if rate is 0
        else:
            self.raster_group_interval_s = None # Or 0.0, since it won't be used

        self.raster_last_update_time = 0.0

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


        self._sorted_cartesian_coords = getattr(self, "_sorted_cartesian_coords", None)
        self._sorted_raster_coords = getattr(self, "_sorted_raster_coords", None)

        # Create raster groups based on physical coordinates
        if self.raster_enabled:
            if self._sorted_raster_coords is not None:
                x_np, y_np = self._sorted_raster_coords
            elif self._sorted_cartesian_coords is not None:
                x_np, y_np = self._sorted_cartesian_coords
            else:
                x_np, y_np = coordinates.cartesian

            groups_tensor = _coordinate_based_raster_groups(
                x_np,
                y_np,
                pattern=self.raster_pattern,
                num_groups=self.raster_num_groups,
                device=self.data_kwargs['device'],
            )

            # Assign to self
            self.raster_groups_flat = groups_tensor

            self._raster_group_pool = None
            self._raster_group_pool_index = 0
            if self.raster_pattern == 'random':
                raster_cfg = self.params.get('raster', {}) or {}
                pool_size = max(1, int(raster_cfg.get('pseudo_random_pool_size', 4)))
                self._raster_group_pool = [groups_tensor.detach().cpu()]
                for _ in range(pool_size - 1):
                    pooled_groups = _coordinate_based_raster_groups(
                        x_np,
                        y_np,
                        pattern=self.raster_pattern,
                        num_groups=self.raster_num_groups,
                        device='cpu',
                    )
                    self._raster_group_pool.append(pooled_groups.detach().cpu())

            # Create schedule masks (one per group) - operations on GPU
            self.raster_schedule = []
            for group_idx in range(self.raster_num_groups):
                mask = (self.raster_groups_flat == group_idx).float()
                # Reshape to match the spatial dimensions (Channels, Height, Width)
                mask = mask.reshape(self.shape[-3:])
                self.raster_schedule.append(mask)

            # Initialize raster state
            self.current_raster_group = 0
            self.raster_time_accumulator = 0.0

            # For pseudo-random pattern, store when to reshuffle.
            if self.raster_pattern == 'random':
                self.raster_reshuffle_interval_s = max(
                    1e-9,
                    float(raster_cfg.get('reshuffle_interval_s', raster_cfg.get('reshuffle_interval', 5.0))),
                )
                self.raster_reshuffle_time_accumulator = 0.0
                self.raster_frame_counter = 0
        else:
            self.raster_groups = None
            self.raster_groups_flat = None
            self.raster_schedule = None
            self._raster_group_pool = None
            self._raster_group_pool_index = 0
            self.current_raster_group = 0
            self.raster_time_accumulator = 0.0

        self.reset()

    @property
    def num_phosphenes(self):
        if hasattr(self, "_num_phosphenes"):
            return self._num_phosphenes
        return len(self.phosphene_maps)

    def to_tensor(self, x: Union[int, float, np.ndarray]) -> torch.Tensor:
        return to_tensor(x, **self.data_kwargs)

    def _initialize_lightweight_phosphene_geometry(
            self,
            coordinates: Map,
            raster_coordinates: Optional[Map] = None,
    ) -> None:
        """Initialize sorted coordinates without allocating full distance maps."""
        x_coords, y_coords = coordinates.cartesian
        x_coords = np.asarray(x_coords)
        y_coords = np.asarray(y_coords)
        if raster_coordinates is None:
            raster_x, raster_y = x_coords.copy(), y_coords.copy()
        else:
            raster_x, raster_y = raster_coordinates.cartesian
            raster_x = np.asarray(raster_x)
            raster_y = np.asarray(raster_y)
            if len(raster_x) != len(x_coords) or len(raster_y) != len(y_coords):
                raise ValueError("raster_coordinates must match the number of phosphene coordinates.")

        indices = np.arange(len(x_coords))
        sort_idx = np.lexsort((x_coords, -y_coords))
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        raster_x = raster_x[sort_idx]
        raster_y = raster_y[sort_idx]
        self._sorted_cartesian_coords = (x_coords.copy(), y_coords.copy())
        self._sorted_raster_coords = (raster_x.copy(), raster_y.copy())
        self.electrode_tags = indices[sort_idx]

    def reset(self):
        """Reset memory of previous timestep and raster state."""
        self.activation.reset()
        self.trace.reset()
        self.sigma.reset()
        self.delivered_amplitude = torch.zeros(self.shape, **self.data_kwargs)
        self.delivered_charge_per_second = torch.zeros(self.shape, **self.data_kwargs)
        self.current_pulse_width = torch.zeros(self.shape, **self.data_kwargs)
        self.current_frequency = torch.zeros(self.shape, **self.data_kwargs)

        # Reset raster state
        if self.raster_enabled:
            self.current_raster_group = 0
            self.raster_time_accumulator = 0.0
            if self.raster_pattern == 'random':
                self.raster_reshuffle_time_accumulator = 0.0
                self.raster_frame_counter = 0

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
                                remove_invalid: Optional[bool] = False,
                                theta: Optional[np.ndarray] = None,
                                raster_coordinates: Optional[Map] = None,
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
        if raster_coordinates is None:
            raster_x, raster_y = x_coords.copy(), y_coords.copy()
        else:
            raster_x, raster_y = raster_coordinates.cartesian
            raster_x = np.asarray(raster_x)
            raster_y = np.asarray(raster_y)
            if len(raster_x) != len(x_coords) or len(raster_y) != len(y_coords):
                raise ValueError("raster_coordinates must match the number of phosphene coordinates.")

        # Implementation of spatial tagging and sorting
        indices = np.arange(len(x_coords))
        sort_idx = np.lexsort((x_coords, -y_coords))
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        raster_x = raster_x[sort_idx]
        raster_y = raster_y[sort_idx]
        self._sorted_cartesian_coords = (x_coords.copy(), y_coords.copy())
        self._sorted_raster_coords = (raster_x.copy(), raster_y.copy())
        self.electrode_tags = indices[sort_idx]

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
            raster_x = raster_x[to_numpy(valid)]
            raster_y = raster_y[to_numpy(valid)]
            self.electrode_tags = self.electrode_tags[to_numpy(valid)]
            self._sorted_raster_coords = (raster_x.copy(), raster_y.copy())
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

    def _init_power_heatmap_weights(self):
        """
        Pre-calculates the weights and indices for the spatial power heatmap.
        Finds the 4 nearest electrodes for every pixel and computes 1/dist^2 weights.
        """
        # self.phosphene_maps contains distances from each electrode (N) to every pixel (H, W)
        # Shape: (N_phosphenes, H, W)
        dists = self.phosphene_maps

        # Find the 4 closest electrodes for every pixel
        # values: the distances, indices: the electrode IDs
        # We use largest=False to get the smallest distances
        k = 4
        nearest_dists, self.power_indices = torch.topk(dists, k=k, dim=0, largest=False)

        # Calculate Weights: 1 / (distance^2)
        # Add a small epsilon to avoid division by zero at the exact electrode center
        epsilon = 1e-6
        dist_sq = nearest_dists ** 2
        self.power_weights = 1.0 / (dist_sq + epsilon)


    def update(self, amplitude: torch.Tensor,
               pulse_width: Optional[torch.Tensor] = None,
               frequency: Optional[torch.Tensor] = None,
               dt: Optional[float] = None,
               temperature_increase: Optional[Union[float, torch.Tensor]] = None):
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

        # Update raster state (timing)
        self._update_raster_state(dt)

        # Get current raster mask
        raster_mask = self.get_current_raster_mask()

        # Apply raster mask to amplitude
        # Electrodes not in the current active group get 0 amplitude (0 Current).
        masked_amplitude = amplitude.view(self.shape) * raster_mask

        # Handle defaults

        if pulse_width is None:
            pulse_width = self._pulse_width
        if frequency is None:
            frequency = self._frequency

        current_frequency = frequency.view(self.shape)
        current_pulse_width = pulse_width.view(self.shape)

        charge_per_s = self.get_current(masked_amplitude,
                                        current_frequency,
                                        current_pulse_width)
        delivered_charge_per_s = masked_amplitude * current_pulse_width * current_frequency
        self.delivered_amplitude = masked_amplitude
        self.delivered_charge_per_second = delivered_charge_per_s
        self.current_pulse_width = current_pulse_width
        self.current_frequency = current_frequency

        # Determine time step (Frame Duration) for Energy calculation
        if dt is None:
            dt = 1.0 / self.params['run']['fps']

        self.activation.update(charge_per_s)
        self.trace.update(charge_per_s)
        self.sigma.update(masked_amplitude)
        self.brightness.update(self.activation.get())



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
        if self.phosphene_maps is None:
            raise RuntimeError(
                "Gaussian phosphene rendering requires phosphene_mode='visual'."
            )

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
            'effective_charge_per_second':self.effective_charge_per_second,
            'delivered_amplitude': self.delivered_amplitude,
            'delivered_charge_per_second': self.delivered_charge_per_second,
            'pulse_width': self.current_pulse_width,
            'frequency': self.current_frequency,
            }

        return state

    def __call__(self, amplitude, pulse_width=None, frequency=None, temperature_increase=None):
        self.update(amplitude, pulse_width, frequency, temperature_increase=temperature_increase)
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
            if self.phosphene_maps is not None:
                self._phosphene_centers = self.phosphene_maps.flatten(start_dim=1).argmin(dim=-1)
            else:
                x_coords, y_coords = self._sorted_cartesian_coords
                res_x, res_y = self.params['run']['resolution']
                x_org, y_org = self.params['run']['origin']
                hemi_fov = self.params['run']['view_angle'] / 2
                x_min, x_max = x_org - hemi_fov, x_org + hemi_fov
                y_min, y_max = y_org - hemi_fov, y_org + hemi_fov
                x_idx = np.rint((x_coords - x_min) / (x_max - x_min) * (res_x - 1))
                y_idx = np.rint((y_coords - y_min) / (y_max - y_min) * (res_y - 1))
                x_idx = np.clip(x_idx, 0, res_x - 1).astype(np.int64)
                y_idx = np.clip(y_idx, 0, res_y - 1).astype(np.int64)
                flat_idx = y_idx * int(res_x) + x_idx
                self._phosphene_centers = torch.tensor(
                    flat_idx,
                    dtype=torch.long,
                    device=self.data_kwargs['device'],
                )
        return self._phosphene_centers

    def sample_centers(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the value of the activation mask at the center pixel of each phosphene"""
        return x.flatten(-2)[..., self.phosphene_centers]

    def sample_receptive_fields(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the maximum value of activation mask x within the 'receptive field' of each phosphene"""
        if self.phosphene_maps is None:
            if not self._warned_lightweight_receptive_fields:
                warnings.warn(
                    "phosphene_mode='safety_centers' does not allocate receptive-field masks; "
                    "sampling falls back to phosphene center pixels.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_lightweight_receptive_fields = True
            return self.sample_centers(x)
        return torch.amax(self.sampling_mask * x, dim=(-2,-1))

    @property
    def sampling_mask(self):
        """Boolean mask (tensor) that defines which pixels are inside the receptive field / center of each phosphene"""
        if self.phosphene_maps is None:
            raise RuntimeError(
                "sampling_mask requires full phosphene maps. Use phosphene_mode='visual' "
                "or sample center pixels in safety_centers mode."
            )
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
            'reshuffle_interval_s': getattr(self, 'raster_reshuffle_interval_s', 0.0),
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
            if self.raster_pattern == 'random':
                self.raster_reshuffle_time_accumulator += dt

            # Check if we should advance to next group
            while self.raster_time_accumulator >= self.raster_group_interval_s:
                self.raster_time_accumulator -= self.raster_group_interval_s
                self.current_raster_group = (self.current_raster_group + 1) % self.raster_num_groups

            # Handle pseudo-random pattern reshuffling on a seconds-based
            # cadence, independent of video FPS.
            if self.raster_pattern == 'random':
                self.raster_frame_counter += 1
                while self.raster_reshuffle_time_accumulator >= self.raster_reshuffle_interval_s:
                    self.raster_reshuffle_time_accumulator -= self.raster_reshuffle_interval_s
                    self._reshuffle_random_pattern()

    def _reshuffle_random_pattern(self):
        """Create a new spaced pseudo-random assignment for random rastering."""
        num_electrodes = len(self.raster_groups_flat)
        group_pool = getattr(self, "_raster_group_pool", None)

        if group_pool:
            self._raster_group_pool_index = (self._raster_group_pool_index + 1) % len(group_pool)
            new_groups = group_pool[self._raster_group_pool_index].to(self.raster_groups_flat.device)

        elif self._sorted_raster_coords is not None:
            x_np, y_np = self._sorted_raster_coords
            new_groups = _coordinate_based_raster_groups(
                x_np,
                y_np,
                pattern=self.raster_pattern,
                num_groups=self.raster_num_groups,
                device=self.raster_groups_flat.device,
            )
        else:
            raster_groups = create_raster_groups(
                self.electrode_array_shape,
                self.raster_num_groups,
                pattern=self.raster_pattern,
                device=self.raster_groups_flat.device,
            )
            new_groups = raster_groups.reshape(-1)[:num_electrodes]
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

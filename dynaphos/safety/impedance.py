"""Electrode impedance and stimulation power helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from dynaphos.simulation.simulator import State
from dynaphos.simulation.utils import print_stats


class Impedance(State):
    """
    Stores the uniform tissue resistance (Ohms) used for electrode load power:
        P = I^2 * Rtis
    """
    def __init__(self, params: dict, shape: Tuple[int, ...],
                 rng: Optional[np.random.Generator] = None,
                 verbose: Optional[bool] = False):
        super().__init__(params, shape, verbose)

        imp = self.params.get('impedance', {})
        self.Rtis = float(imp.get('Rtis', 9.9e3))
        if not np.isfinite(self.Rtis) or self.Rtis <= 0.0:
            raise ValueError("impedance.Rtis must be a finite positive resistance.")

        # Keep rng in the public signature for compatibility with existing callers.
        _ = rng
        self.state = torch.full(self.shape, self.Rtis, **self.data_kwargs)

        print_stats("Uniform Rtis (Ohms)", self.state, self.verbose)

    def update(self, x: torch.Tensor):
        # Tissue resistance is static and identical for every electrode.
        pass


def compute_frame_power(
    amplitude: torch.Tensor,
    impedance_ohm: torch.Tensor,
    pulse_width_s: torch.Tensor,
    frequency_hz: torch.Tensor,
    relative_stim_duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return peak and frame-average electrode load power in watts."""
    instant_power = (amplitude ** 2) * impedance_ohm
    duty_cycle = 2.0 * pulse_width_s * frequency_hz
    frame_power = instant_power * duty_cycle * float(relative_stim_duration)
    return instant_power, frame_power


def compute_device_power(
    amplitude: torch.Tensor,
    impedance_ohm: torch.Tensor,
    pulse_width_s: torch.Tensor,
    frequency_hz: torch.Tensor,
    relative_stim_duration: float,
    constant_power_W: float,
    driver_efficiency: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return stimulation-dependent IC heat and total IC heat power in watts.

    The stimulation-dependent IC component has been removed from the thermal
    model. Electrode load power is handled directly by Bioheat2D at electrode
    locations, while IC heat is the configured constant power only.
    """
    reference = torch.as_tensor(
        amplitude,
        dtype=torch.float32,
        device=amplitude.device if torch.is_tensor(amplitude) else None,
    )
    stim_ic_power = torch.zeros((), dtype=reference.dtype, device=reference.device)
    total_power = torch.as_tensor(
        max(float(constant_power_W), 0.0),
        dtype=stim_ic_power.dtype,
        device=stim_ic_power.device,
    )
    return stim_ic_power, total_power

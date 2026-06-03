"""Electrode impedance and stimulation power helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from dynaphos.simulator import State
from dynaphos.utils import print_stats


class Impedance(State):
    """
    Stores per-electrode REAL impedance (Ohms) used for load-power diagnostics:
        P = I^2 * Re(Z)
    """
    def __init__(self, params: dict, shape: Tuple[int, ...],
                 rng: Optional[np.random.Generator] = None,
                 verbose: Optional[bool] = False):
        super().__init__(params, shape, verbose)

        imp = self.params.get('impedance', {})
        self.Rtis = float(imp.get('Rtis', 9.9e3))
        self.Cdl  = float(imp.get('Cdl', 113.4e-9))
        self.Rct  = float(imp.get('Rct', 2.2e6))
        self.sigma_w = float(imp.get('sigma_w', 2.5e6))

        self.cv = float(imp.get('real_impedance_cv', 0.10))  # coefficient of variation
        seed = int(imp.get('seed', self.params['run']['seed']))
        self.rng = np.random.default_rng(seed) if rng is None else rng

        freq = float(self.params['default_stim']['freq_default'])

        # Base real impedance (scalar) from Randles model
        z_real = self.calculate_randles_real(freq)

        # Per-electrode variability (multiplicative), truncated to avoid extreme outliers
        # factor ~ N(1, cv) truncated to [1-2cv, 1+2cv]
        lo = 1.0 - 2.0 * self.cv
        hi = 1.0 + 2.0 * self.cv
        factors = self.rng.normal(loc=1.0, scale=self.cv, size=shape)
        factors = np.clip(factors, lo, hi)

        z_vals = z_real * factors  # per-electrode real impedances
        self.state = self.to_tensor(z_vals).clip(1.0, None)  # keep >= 1 ohm for safety

        print_stats(f"Re(Z) @ {freq}Hz (Ohms)", self.state, self.verbose)

    def calculate_randles_real(self, freq: float) -> float:
        """
        Randles model:
            Z = Rtis + Rct/(1 + j*omega*Rct*Cdl) + sigma_w / sqrt(j*omega)
        Return real part: Re(Z)
        """
        omega = 2.0 * np.pi * freq
        j = 1j

        z_faradaic = self.Rct / (1.0 + j * omega * self.Rct * self.Cdl)
        z_warburg  = self.sigma_w / (np.sqrt(j * omega))
        z_total = self.Rtis + z_faradaic + z_warburg

        return float(np.real(z_total))

    def update(self, x: torch.Tensor):
        # Static in this version; could be made frequency-dependent later.
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

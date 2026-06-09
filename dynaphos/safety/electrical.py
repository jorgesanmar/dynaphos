"""Electrical safety primitives.

This module is the stable import location for electrical calculations while
the legacy implementation is migrated in smaller steps.
"""

from dynaphos.safety.impedance import Impedance, compute_device_power, compute_frame_power
from dynaphos.safety.tracking import SafetyTracker

__all__ = [
    "Impedance",
    "SafetyTracker",
    "compute_device_power",
    "compute_frame_power",
]

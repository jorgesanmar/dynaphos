from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dynaphos.paths import package_file
from dynaphos.safety.io import load_safety_limits as _load_legacy_limits


def load_safety_limits(path: str | Path | None = None) -> dict[str, float]:
    resolved = package_file("safety") if path is None else Path(path)
    limits = _load_legacy_limits(resolved)
    with open(resolved, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    thresholds = config.get("thresholds", {}) or {}
    limits.update(
        {
            "amplitude_uA": float(thresholds.get("amplitude_max_ua", np.inf)),
            "pulse_width_us": float(thresholds.get("pulse_width_max_us", np.inf)),
            "frequency_hz": float(thresholds.get("frequency_max_hz", np.inf)),
            "charge_density_uc_cm2": float(
                thresholds.get("charge_density_max_uc_per_cm2", np.inf)
            ),
            "shannon_k": float(thresholds.get("shannon_k_limit", np.inf)),
            "current_density_A_cm2": float(
                thresholds.get("current_density_max_a_per_cm2", np.inf)
            ),
            "power_per_electrode_mW": float(
                thresholds.get("thermal_power_limit_per_electrode_mw", np.inf)
            ),
            "power_total_mW": float(
                thresholds.get("thermal_power_limit_total_mw", np.inf)
            ),
        }
    )
    return limits

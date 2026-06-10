from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dynaphos.paths import package_file
def _safety_value_to_nC(section: dict, *, default_factor: float) -> float:
    value = float((section or {}).get("value", np.inf))
    unit = str((section or {}).get("unit", "")).strip().lower().replace("µ", "u")
    if unit.startswith("uc"):
        return value * 1e3
    if unit.startswith("nc"):
        return value
    if unit.startswith("c"):
        return value * 1e9
    return value * float(default_factor)


def load_safety_limits(path: str | Path | None = None) -> dict[str, float]:
    resolved = package_file("safety") if path is None else Path(path)
    with open(resolved, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    thresholds = config.get("thresholds", {}) or {}
    thermal = config.get("thermal", {}) or {}
    guidelines = config.get("stimulation_safety_guidelines", {}) or {}
    accumulated = guidelines.get("accumulated_charge", {}) or {}
    accumulated_limits = accumulated.get("limits", {}) or {}
    temperature = guidelines.get("temperature", {}) or {}
    temperature_increase = temperature.get("temperature_increase", {}) or {}
    chronic = config.get("chronic", {}) or {}
    total_charge_limit_nC = (
        float(thresholds.get("accumulated_charge_limit_mc_per_second", np.inf))
        * 1e6
    )
    if not np.isfinite(total_charge_limit_nC):
        total_charge_limit_nC = _safety_value_to_nC(
            accumulated_limits.get("total_all_electrodes", {}) or {},
            default_factor=1e3,
        )
    return {
            "charge_per_phase_nC": float(
                thresholds.get(
                    "charge_per_phase_max_nc",
                    (guidelines.get("charge_per_phase", {}) or {}).get(
                        "value", np.inf
                    ),
                )
            ),
            "window_charge_per_electrode_nC": float(
                thresholds.get(
                    "accumulated_charge_per_electrode_nc_per_s",
                    _safety_value_to_nC(
                        accumulated_limits.get("per_electrode", {}) or {},
                        default_factor=1.0,
                    ),
                )
            ),
            "window_charge_total_nC": total_charge_limit_nC,
            "simultaneous_activation_pct": float(
                thresholds.get(
                    "simultaneous_activation_max_percentage",
                    (
                        (
                            guidelines.get("simultaneous_activation", {}) or {}
                        ).get("max_percentage_of_electrodes", {})
                        or {}
                    ).get("value", np.inf),
                )
            ),
            "temperature_increase_C": float(
                thermal.get(
                    "max_temp_rise",
                    (
                        temperature_increase.get(
                            "absolute_max_temperature_increase", {}
                        )
                        or {}
                    ).get("value", np.inf),
                )
            ),
            "cem43_min": float(
                thermal.get(
                    "cem43_limit_min",
                    (temperature.get("cem43", {}) or {}).get("value", np.inf),
                )
            ),
            "session_charge_limit_mC": float(
                chronic.get("session_charge_limit_c", np.inf)
            )
            * 1e3,
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

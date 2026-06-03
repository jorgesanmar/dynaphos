"""Electrical safety metric tracking for stimulation amplitudes."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional, Union

import logging
import warnings

import numpy as np
import torch
import yaml


class SafetyTracker:
    """Tracks and enforces stimulation safety constraints from config/safety.yaml."""

    def __init__(self, params: dict, num_electrodes: int, data_kwargs: dict):
        self.params = params
        self.num_electrodes = num_electrodes
        self.data_kwargs = data_kwargs
        self.enabled = bool(params.get("safety", {}).get("enable_charge_guard", True))
        self.warn_only = bool(params.get("safety", {}).get("charge_warn_only", True))
        self.log_every = int(params.get("safety", {}).get("charge_log_every", 0))
        self._step = 0

        self.default_dt_s = 1.0 / float(self.params["run"]["fps"])
        self.rel_stim_duration = float(self.params["default_stim"]["relative_stim_duration"])

        safety_config = self._load_safety_config()
        self._parse_limits(safety_config)
        self.reset()

    @staticmethod
    def _to_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _load_safety_config(self) -> dict:
        safety_params = self.params.get("safety", {}) or {}
        explicit = safety_params.get("guidelines_path", None)
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        candidates.append(Path(__file__).resolve().parents[2] / "config" / "safety.yaml")

        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        warnings.warn("Safety guidelines file not found; safety tracker will run with permissive defaults.")
        return {}

    @staticmethod
    def _unit_to_nC_factor(unit: object, default: float = 1.0) -> float:
        text = str(unit or "").strip().lower().replace("u", "u")
        if not text:
            return float(default)
        if text.startswith("uc"):
            return 1e3
        if text.startswith("nc"):
            return 1.0
        if text.startswith("c"):
            return 1e9
        return float(default)

    def _parse_limits(self, safety_config: dict):
        safety_params = self.params.get("safety", {}) or {}
        guidelines = safety_config.get("stimulation_safety_guidelines", safety_config) or {}
        geometry = safety_config.get("geometry", guidelines.get("geometry", {})) or {}
        thresholds = safety_config.get("thresholds", guidelines.get("thresholds", {})) or {}

        charge_per_phase = guidelines.get("charge_per_phase", {}) or {}
        self.charge_per_phase_limit_nC = self._to_float(charge_per_phase.get("value"), float("inf"))
        self.charge_per_phase_limit_nC = self._to_float(
            safety_params.get(
                "charge_per_phase_limit_nC",
                thresholds.get("charge_per_phase_max_nc", self.charge_per_phase_limit_nC),
            ),
            self.charge_per_phase_limit_nC,
        )

        accumulated_charge = guidelines.get("accumulated_charge", {}) or {}
        tw = accumulated_charge.get("time_window", {}) or {}
        window_from_thresholds = self._to_float(
            thresholds.get("accumulated_charge_window_ms", 1000.0),
            1000.0,
        ) / 1000.0
        self.charge_window_s = self._to_float(
            safety_params.get("charge_window_s", tw.get("value")),
            window_from_thresholds,
        )

        acc_limits = accumulated_charge.get("limits", {}) or {}
        per_e = acc_limits.get("per_electrode", {}) or {}
        total = acc_limits.get("total_all_electrodes", {}) or {}
        self.acc_limit_per_electrode_nC = (
            self._to_float(per_e.get("value"), thresholds.get("accumulated_charge_per_electrode_nc_per_s", float("inf")))
            * self._unit_to_nC_factor(per_e.get("unit"), default=1.0)
        )
        self.acc_limit_total_nC = (
            self._to_float(total.get("value"), thresholds.get("accumulated_charge_limit_uc_per_second", float("inf")))
            * self._unit_to_nC_factor(total.get("unit"), default=1e3)
        )
        self.acc_limit_per_electrode_nC = self._to_float(
            safety_params.get("window_charge_limit_per_electrode_nC", self.acc_limit_per_electrode_nC),
            self.acc_limit_per_electrode_nC,
        )
        self.acc_limit_total_nC = self._to_float(
            safety_params.get("window_charge_limit_total_nC", self.acc_limit_total_nC),
            self.acc_limit_total_nC,
        )

        self.electrode_surface_area_cm2 = self._to_float(
            safety_params.get("electrode_surface_area_cm2", geometry.get("electrode_surface_area_cm2")),
            float("nan"),
        )
        self.charge_density_limit_uc_cm2 = self._to_float(
            safety_params.get(
                "charge_density_limit_uc_per_cm2",
                (guidelines.get("charge_density", {}) or {}).get(
                    "value",
                    thresholds.get("charge_density_max_uc_per_cm2"),
                ),
            ),
            float("inf"),
        )
        self.shannon_k_limit = self._to_float(
            safety_params.get(
                "shannon_k_limit",
                (guidelines.get("shannon_k", {}) or {}).get("value", thresholds.get("shannon_k_limit")),
            ),
            float("inf"),
        )

        chronic = safety_config.get("chronic", guidelines.get("chronic", {})) or {}
        self.protocol_charge_limit_total_nC = self._to_float(
            safety_params.get("protocol_charge_limit_total_c", chronic.get("session_charge_limit_c")),
            float("inf"),
        ) * 1e9
        self.protocol_charge_limit_per_electrode_nC = self._to_float(
            safety_params.get("protocol_charge_limit_per_electrode_c"),
            float("inf"),
        ) * 1e9

        sim_activation = guidelines.get("simultaneous_activation", {}) or {}
        pct = sim_activation.get("max_percentage_of_electrodes", {}) or {}
        self.max_active_pct = self._to_float(pct.get("value"), float("inf"))

        temp = guidelines.get("temperature", {}) or {}
        temp_inc = temp.get("temperature_increase", guidelines.get("temperature_increase", {})) or {}
        max_dur = temp.get("max_continuous_duration_above_1_c", {}) or {}
        if not max_dur:
            max_dur = temp_inc.get("max_continuous_duration_above_1_c", {}) or {}
        abs_max = temp_inc.get("absolute_max_temperature_increase", {}) or {}
        self.max_duration_above_1c_s = self._to_float(max_dur.get("value"), float("inf"))
        self.abs_max_temp_increase_c = self._to_float(abs_max.get("value"), float("inf"))

    def _ensure_device(self, ref: torch.Tensor):
        if self.window_charge_per_electrode_nC.device == ref.device and self.window_charge_per_electrode_nC.dtype == ref.dtype:
            return

        self.window_charge_per_electrode_nC = self.window_charge_per_electrode_nC.to(device=ref.device, dtype=ref.dtype)
        updated_entries = deque()
        for duration_s, q_nC in self._window_entries:
            updated_entries.append((duration_s, q_nC.to(device=ref.device, dtype=ref.dtype)))
        self._window_entries = updated_entries

    def reset(self):
        self._window_entries = deque()
        self._window_time_s = 0.0
        self.window_charge_per_electrode_nC = torch.zeros(self.num_electrodes, **self.data_kwargs)
        self.protocol_charge_per_electrode_nC = torch.zeros(self.num_electrodes, **self.data_kwargs)
        self.active_above_1c_duration_s = 0.0
        self.last_temperature_increase_c = 0.0
        self.last_temperature_mean_increase_c = 0.0
        self.last_area_above_1c_mm2 = 0.0
        self.last_area_above_2c_mm2 = 0.0
        self.last_active_pct = 0.0
        self.last_charge_per_phase_nC = torch.zeros(self.num_electrodes, **self.data_kwargs)
        self.last_charge_density_uc_cm2 = torch.zeros(self.num_electrodes, **self.data_kwargs)
        self.last_shannon_k = torch.full((self.num_electrodes,), -torch.inf, **self.data_kwargs)
        self._step = 0

    def _raise_or_warn(self, message: str):
        if not self.enabled:
            return
        if self.warn_only:
            warnings.warn(message)
        else:
            raise RuntimeError(message)

    @staticmethod
    def _reduce_per_electrode(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x.sum(dim=(0, 2, 3))
        if x.dim() == 3:
            return x.sum(dim=(1, 2))
        return x.reshape(-1)

    def _push_charge_window(self, dt_s: float, delta_charge_nC: torch.Tensor):
        self._window_entries.append((dt_s, delta_charge_nC))
        self._window_time_s += dt_s
        self.window_charge_per_electrode_nC = self.window_charge_per_electrode_nC + delta_charge_nC

        while self._window_time_s > self.charge_window_s + 1e-12 and len(self._window_entries) > 0:
            excess = self._window_time_s - self.charge_window_s
            head_dt, head_q = self._window_entries[0]
            if head_dt <= excess + 1e-12:
                self._window_entries.popleft()
                self._window_time_s -= head_dt
                self.window_charge_per_electrode_nC = self.window_charge_per_electrode_nC - head_q
            else:
                frac = excess / head_dt
                trim_q = head_q * frac
                self._window_entries[0] = (head_dt - excess, head_q - trim_q)
                self._window_time_s -= excess
                self.window_charge_per_electrode_nC = self.window_charge_per_electrode_nC - trim_q

    def _check_temperature(self, temperature_increase: Optional[Union[float, torch.Tensor]],
                           dt_s: float, pixel_area_mm2: Optional[float] = None):
        if temperature_increase is None:
            return

        if torch.is_tensor(temperature_increase):
            temp_map = torch.as_tensor(temperature_increase)
            temp_delta_c = float(temp_map.max().item())
            self.last_temperature_mean_increase_c = float(temp_map.mean().item())
            if pixel_area_mm2 is not None:
                self.last_area_above_1c_mm2 = float((temp_map > 1.0).sum().item()) * pixel_area_mm2
                self.last_area_above_2c_mm2 = float((temp_map > 2.0).sum().item()) * pixel_area_mm2
        else:
            temp_delta_c = float(temperature_increase)
            self.last_temperature_mean_increase_c = temp_delta_c
            self.last_area_above_1c_mm2 = 0.0
            self.last_area_above_2c_mm2 = 0.0

        self.last_temperature_increase_c = temp_delta_c

        if temp_delta_c >= self.abs_max_temp_increase_c:
            self._raise_or_warn(
                f"[SafetyTracker] Temperature increase reached {temp_delta_c:.3f} C "
                f"(absolute max {self.abs_max_temp_increase_c:.3f} C)."
            )

        if temp_delta_c > 1.0:
            self.active_above_1c_duration_s += dt_s
        else:
            self.active_above_1c_duration_s = 0.0

        if self.active_above_1c_duration_s > self.max_duration_above_1c_s:
            self._raise_or_warn(
                f"[SafetyTracker] Temperature increase >1 C for {self.active_above_1c_duration_s:.2f}s "
                f"(limit {self.max_duration_above_1c_s:.2f}s)."
            )

    def update(self, charge_per_s: torch.Tensor, frequency: torch.Tensor,
               dt_s: Optional[float] = None,
               temperature_increase: Optional[Union[float, torch.Tensor]] = None):
        if dt_s is None:
            dt_s = self.default_dt_s

        per_elec_charge_rate = self._reduce_per_electrode(charge_per_s.detach())
        per_elec_frequency = self._reduce_per_electrode(frequency.detach())
        self._ensure_device(per_elec_charge_rate)

        safe_freq = torch.where(per_elec_frequency > 0, per_elec_frequency, torch.ones_like(per_elec_frequency))
        charge_per_phase_nC = torch.where(
            per_elec_frequency > 0,
            (per_elec_charge_rate / safe_freq) * 1e9,
            torch.zeros_like(per_elec_charge_rate),
        )
        self.last_charge_per_phase_nC = charge_per_phase_nC

        charge_per_phase_uC = charge_per_phase_nC / 1e3
        if np.isfinite(self.electrode_surface_area_cm2) and self.electrode_surface_area_cm2 > 0:
            charge_density_uc_cm2 = charge_per_phase_uC / self.electrode_surface_area_cm2
        else:
            charge_density_uc_cm2 = torch.zeros_like(charge_per_phase_uC)
        self.last_charge_density_uc_cm2 = charge_density_uc_cm2

        positive_mask = (charge_per_phase_uC > 0) & (charge_density_uc_cm2 > 0)
        self.last_shannon_k = torch.full_like(charge_density_uc_cm2, -torch.inf)
        self.last_shannon_k[positive_mask] = (
            torch.log10(charge_per_phase_uC[positive_mask]) +
            torch.log10(charge_density_uc_cm2[positive_mask])
        )

        over_phase = charge_per_phase_nC > self.charge_per_phase_limit_nC
        if torch.any(over_phase):
            idx = torch.nonzero(over_phase, as_tuple=False).view(-1).tolist()
            self._raise_or_warn(
                f"[SafetyTracker] Charge/phase exceeded on electrodes {idx}; "
                f"limit={self.charge_per_phase_limit_nC:.3f} nC."
            )

        over_density = charge_density_uc_cm2 > self.charge_density_limit_uc_cm2
        if torch.any(over_density):
            idx = torch.nonzero(over_density, as_tuple=False).view(-1).tolist()
            self._raise_or_warn(
                f"[SafetyTracker] Charge density exceeded on electrodes {idx}; "
                f"limit={self.charge_density_limit_uc_cm2:.3f} uC/cm^2."
            )

        over_shannon = self.last_shannon_k > self.shannon_k_limit
        if torch.any(over_shannon):
            idx = torch.nonzero(over_shannon, as_tuple=False).view(-1).tolist()
            self._raise_or_warn(
                f"[SafetyTracker] Shannon k exceeded on electrodes {idx}; "
                f"limit={self.shannon_k_limit:.3f}."
            )

        active_count = int((per_elec_charge_rate > 0).sum().item())
        self.last_active_pct = 100.0 * active_count / max(self.num_electrodes, 1)
        if self.last_active_pct > self.max_active_pct:
            self._raise_or_warn(
                f"[SafetyTracker] Active electrode percentage {self.last_active_pct:.2f}% "
                f"exceeds limit {self.max_active_pct:.2f}%."
            )

        # Charge-per-phase is reported above as a single phase. Accumulated
        # charge tracks delivered absolute charge for biphasic pulses.
        delta_charge_nC = 2.0 * per_elec_charge_rate * (dt_s * self.rel_stim_duration * 1e9)
        self._push_charge_window(dt_s, delta_charge_nC)
        self.protocol_charge_per_electrode_nC = self.protocol_charge_per_electrode_nC + delta_charge_nC

        over_window_per_elec = self.window_charge_per_electrode_nC > self.acc_limit_per_electrode_nC
        if torch.any(over_window_per_elec):
            idx = torch.nonzero(over_window_per_elec, as_tuple=False).view(-1).tolist()
            self._raise_or_warn(
                f"[SafetyTracker] {self.charge_window_s:.2f}s accumulated charge per electrode exceeded on electrodes {idx}; "
                f"limit={self.acc_limit_per_electrode_nC:.3f} nC."
            )

        total_window_nC = float(self.window_charge_per_electrode_nC.sum().item())
        if total_window_nC > self.acc_limit_total_nC:
            self._raise_or_warn(
                f"[SafetyTracker] {self.charge_window_s:.2f}s total accumulated charge exceeded; "
                f"total={total_window_nC:.3f} nC, limit={self.acc_limit_total_nC:.3f} nC."
            )

        over_protocol_per_elec = self.protocol_charge_per_electrode_nC > self.protocol_charge_limit_per_electrode_nC
        if torch.any(over_protocol_per_elec):
            idx = torch.nonzero(over_protocol_per_elec, as_tuple=False).view(-1).tolist()
            self._raise_or_warn(
                f"[SafetyTracker] Protocol cumulative charge exceeded on electrodes {idx}; "
                f"limit={self.protocol_charge_limit_per_electrode_nC:.3f} nC."
            )

        protocol_total_nC = float(self.protocol_charge_per_electrode_nC.sum().item())
        if protocol_total_nC > self.protocol_charge_limit_total_nC:
            self._raise_or_warn(
                f"[SafetyTracker] Protocol total accumulated charge exceeded; "
                f"total={protocol_total_nC:.3f} nC, limit={self.protocol_charge_limit_total_nC:.3f} nC."
            )

        self._check_temperature(temperature_increase, dt_s)

        self._step += 1
        if self.log_every > 0 and self._step % self.log_every == 0:
            logging.info(
                "[SafetyTracker] phase_max=%.3f nC, window_max=%.3f nC, window_total=%.3f nC, active=%.2f%%, dT=%.3f C",
                float(self.last_charge_per_phase_nC.max().item()),
                float(self.window_charge_per_electrode_nC.max().item()),
                total_window_nC,
                self.last_active_pct,
                self.last_temperature_increase_c,
            )

    def update_temperature_metrics(self, temperature_increase: Optional[Union[float, torch.Tensor]],
                                   dt_s: Optional[float] = None,
                                   pixel_area_mm2: Optional[float] = None):
        if dt_s is None:
            dt_s = self.default_dt_s
        self._check_temperature(temperature_increase, dt_s, pixel_area_mm2=pixel_area_mm2)

    def get_status(self) -> dict:
        return {
            "charge_per_phase_nC": self.last_charge_per_phase_nC,
            "charge_per_phase_limit_nC": self.charge_per_phase_limit_nC,
            "charge_density_uc_cm2": self.last_charge_density_uc_cm2,
            "charge_density_limit_uc_cm2": self.charge_density_limit_uc_cm2,
            "shannon_k": self.last_shannon_k,
            "shannon_k_limit": self.shannon_k_limit,
            "electrode_surface_area_cm2": self.electrode_surface_area_cm2,
            "window_s": self.charge_window_s,
            "window_charge_per_electrode_nC": self.window_charge_per_electrode_nC,
            "window_charge_total_nC": float(self.window_charge_per_electrode_nC.sum().item()),
            "window_charge_limit_per_electrode_nC": self.acc_limit_per_electrode_nC,
            "window_charge_limit_total_nC": self.acc_limit_total_nC,
            "protocol_charge_per_electrode_nC": self.protocol_charge_per_electrode_nC,
            "protocol_charge_total_nC": float(self.protocol_charge_per_electrode_nC.sum().item()),
            "protocol_charge_limit_per_electrode_nC": self.protocol_charge_limit_per_electrode_nC,
            "protocol_charge_limit_total_nC": self.protocol_charge_limit_total_nC,
            "active_percentage": self.last_active_pct,
            "active_percentage_limit": self.max_active_pct,
            "temperature_increase_c": self.last_temperature_increase_c,
            "temperature_mean_increase_c": self.last_temperature_mean_increase_c,
            "temperature_area_above_1c_mm2": self.last_area_above_1c_mm2,
            "temperature_area_above_2c_mm2": self.last_area_above_2c_mm2,
            "temperature_absolute_limit_c": self.abs_max_temp_increase_c,
            "duration_above_1c_s": self.active_above_1c_duration_s,
            "duration_above_1c_limit_s": self.max_duration_above_1c_s,
        }

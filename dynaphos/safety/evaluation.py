from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dynaphos.safety.limits import load_safety_limits


WITHIN_LIMIT = "within_limit"
EXCEEDS_LIMIT = "exceeds_limit"
NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class MetricEvaluation:
    name: str
    label: str
    observed: float | None
    limit: float | None
    ratio_to_limit: float | None
    unit: str
    status: str
    peak_time_s: float | None = None
    electrode_id: int | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyReport:
    status: str
    status_label: str
    metrics: tuple[MetricEvaluation, ...]
    limitations: tuple[str, ...]
    metrics_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_label": self.status_label,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "limitations": list(self.limitations),
            "metrics_path": self.metrics_path,
        }


def _finite_max(values: np.ndarray) -> tuple[float | None, tuple[int, ...] | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None, None
    finite = np.isfinite(array)
    if not finite.any():
        return None, None
    masked = np.where(finite, array, -np.inf)
    flat_index = int(np.argmax(masked))
    return float(masked.reshape(-1)[flat_index]), np.unravel_index(flat_index, array.shape)


def _evaluate(
    *,
    name: str,
    label: str,
    values: np.ndarray | None,
    limit: float | None,
    unit: str,
    time_s: np.ndarray | None,
    electrode_ids: np.ndarray | None,
    rationale: str,
    transform=None,
) -> MetricEvaluation:
    if values is None or limit is None or not np.isfinite(limit) or limit <= 0.0:
        return MetricEvaluation(
            name=name,
            label=label,
            observed=None,
            limit=None if limit is None or not np.isfinite(limit) else float(limit),
            ratio_to_limit=None,
            unit=unit,
            status=NOT_EVALUATED,
            rationale=rationale,
        )

    observed, index = _finite_max(values)
    if observed is None or index is None:
        return MetricEvaluation(
            name=name,
            label=label,
            observed=None,
            limit=float(limit),
            ratio_to_limit=None,
            unit=unit,
            status=NOT_EVALUATED,
            rationale=rationale,
        )
    if transform is not None:
        observed = float(transform(observed))
    ratio = observed / float(limit)
    time_index = index[0] if len(index) >= 1 else None
    electrode_index = index[1] if len(index) >= 2 else None
    peak_time = (
        float(time_s[time_index])
        if time_s is not None and time_index is not None and time_index < len(time_s)
        else None
    )
    electrode_id = (
        int(electrode_ids[electrode_index])
        if electrode_ids is not None
        and electrode_index is not None
        and electrode_index < len(electrode_ids)
        else None
    )
    return MetricEvaluation(
        name=name,
        label=label,
        observed=observed,
        limit=float(limit),
        ratio_to_limit=ratio,
        unit=unit,
        status=EXCEEDS_LIMIT if ratio > 1.0 else WITHIN_LIMIT,
        peak_time_s=peak_time,
        electrode_id=electrode_id,
        rationale=rationale,
    )


def evaluate_metrics(
    metrics_path: str | Path,
    *,
    safety_limits_path: str | Path | None = None,
) -> SafetyReport:
    path = Path(metrics_path).resolve()
    limits = load_safety_limits(safety_limits_path)
    with np.load(path, allow_pickle=True) as bundle:
        data = {key: np.asarray(bundle[key]) for key in bundle.files}

    time_s = data.get("time_s")
    thermal_time_s = data.get("thermal_time_s")
    electrode_ids = data.get("electrode_ids")
    amplitude = data.get("amplitude_per_electrode_uA")
    if amplitude is not None and amplitude.ndim == 2:
        active_percentage = (
            np.count_nonzero(amplitude > 0.0, axis=1)
            * 100.0
            / max(amplitude.shape[1], 1)
        )
    else:
        active_percentage = None
    electrode_area_cm2 = data.get("electrode_surface_area_cm2")
    if amplitude is not None and electrode_area_cm2 is not None:
        area = float(np.asarray(electrode_area_cm2).reshape(-1)[0])
        current_density = amplitude * 1e-6 / area if area > 0.0 else None
    else:
        current_density = None
    pulse_width = data.get("pulse_width_per_electrode_s", data.get("pulse_width_s"))
    pulse_frequency = data.get(
        "pulse_frequency_per_electrode_hz",
        data.get("pulse_frequency_hz"),
    )
    protocol_charge = data.get("protocol_charge_per_electrode_nC")
    protocol_total_mC = (
        np.asarray([float(np.nansum(protocol_charge)) / 1e6], dtype=np.float64)
        if protocol_charge is not None
        else None
    )
    power_per_electrode = data.get("power_per_electrode_W")
    power_per_electrode_mW = (
        np.asarray(power_per_electrode, dtype=np.float64) * 1e3
        if power_per_electrode is not None
        else None
    )
    if power_per_electrode_mW is not None and power_per_electrode_mW.ndim == 2:
        total_power_mW = np.sum(power_per_electrode_mW, axis=1)
        internal_power = data.get("internal_circuit_power_total_mW")
        if internal_power is not None:
            total_power_mW = total_power_mW + float(np.asarray(internal_power).reshape(-1)[0])
    else:
        total_power_mW = None

    metric_specs = (
        (
            "amplitude",
            "Current amplitude",
            amplitude,
            limits.get("amplitude_uA"),
            "uA",
            time_s,
            electrode_ids,
            "Configured maximum stimulation amplitude.",
            None,
        ),
        (
            "pulse_width",
            "Pulse width",
            pulse_width,
            limits.get("pulse_width_us"),
            "us",
            time_s if pulse_width is not None and np.asarray(pulse_width).ndim == 2 else None,
            electrode_ids,
            "Configured maximum phase pulse width.",
            lambda value: value * 1e6,
        ),
        (
            "frequency",
            "Pulse frequency",
            pulse_frequency,
            limits.get("frequency_hz"),
            "Hz",
            time_s if pulse_frequency is not None and np.asarray(pulse_frequency).ndim == 2 else None,
            electrode_ids,
            "Configured maximum stimulation frequency.",
            None,
        ),
        (
            "charge_per_phase",
            "Charge per phase",
            data.get("charge_per_phase_per_electrode_nC"),
            limits.get("charge_per_phase_nC"),
            "nC",
            time_s,
            electrode_ids,
            "Configured per-electrode charge-per-phase boundary.",
            None,
        ),
        (
            "charge_density",
            "Charge density",
            data.get("charge_density_per_electrode_uc_cm2"),
            limits.get("charge_density_uc_cm2"),
            "uC/cm2",
            time_s,
            electrode_ids,
            "Configured per-electrode charge-density boundary.",
            None,
        ),
        (
            "shannon_k",
            "Shannon k",
            data.get("shannon_k_per_electrode"),
            limits.get("shannon_k"),
            "",
            time_s,
            electrode_ids,
            "Configured Shannon-model reference boundary.",
            None,
        ),
        (
            "current_density",
            "Current density",
            current_density,
            limits.get("current_density_A_cm2"),
            "A/cm2",
            time_s,
            electrode_ids,
            "Configured current-density boundary.",
            None,
        ),
        (
            "window_charge_per_electrode",
            "Rolling charge per electrode",
            data.get("window_charge_per_electrode_nC"),
            limits.get("window_charge_per_electrode_nC"),
            "nC",
            time_s,
            electrode_ids,
            "Configured rolling-window charge boundary for one electrode.",
            None,
        ),
        (
            "window_charge_total",
            "Rolling total-array charge",
            data.get("window_charge_total_nC"),
            limits.get("window_charge_total_nC"),
            "nC",
            time_s,
            None,
            "Configured rolling-window charge boundary for the full array.",
            None,
        ),
        (
            "session_charge",
            "Protocol total charge",
            protocol_total_mC,
            limits.get("session_charge_limit_mC"),
            "mC",
            None,
            None,
            "Configured total session charge boundary.",
            None,
        ),
        (
            "active_percentage",
            "Simultaneously active electrodes",
            active_percentage,
            limits.get("simultaneous_activation_pct"),
            "%",
            time_s,
            None,
            "Configured maximum fraction of electrodes active simultaneously.",
            None,
        ),
        (
            "power_per_electrode",
            "Electrode load power",
            power_per_electrode_mW,
            limits.get("power_per_electrode_mW"),
            "mW",
            time_s,
            electrode_ids,
            "Configured thermal power boundary for one electrode.",
            None,
        ),
        (
            "power_total",
            "Total modeled device power",
            total_power_mW,
            limits.get("power_total_mW"),
            "mW",
            time_s,
            None,
            "Configured total thermal power budget.",
            None,
        ),
        (
            "temperature",
            "Focal temperature rise",
            data.get("max_dT"),
            limits.get("temperature_increase_C"),
            "degC",
            thermal_time_s,
            None,
            "Configured maximum modeled tissue temperature increase.",
            None,
        ),
        (
            "cem43",
            "Thermal dose",
            data.get("max_cem43"),
            limits.get("cem43_min"),
            "min",
            thermal_time_s,
            None,
            "Configured cumulative equivalent minutes at 43 degC.",
            None,
        ),
    )
    evaluations = tuple(
        _evaluate(
            name=name,
            label=label,
            values=values,
            limit=limit,
            unit=unit,
            time_s=metric_time,
            electrode_ids=metric_electrodes,
            rationale=rationale,
            transform=transform,
        )
        for (
            name,
            label,
            values,
            limit,
            unit,
            metric_time,
            metric_electrodes,
            rationale,
            transform,
        ) in metric_specs
    )

    evaluated = [metric for metric in evaluations if metric.status != NOT_EVALUATED]
    if any(metric.status == EXCEEDS_LIMIT for metric in evaluations):
        status = "configured_limits_exceeded"
        status_label = "Configured limits exceeded"
    elif len(evaluated) != len(evaluations):
        status = "incomplete_evaluation"
        status_label = "Incomplete evaluation"
    else:
        status = "within_configured_limits"
        status_label = "Within configured limits"

    return SafetyReport(
        status=status,
        status_label=status_label,
        metrics=evaluations,
        limitations=(
            "Research-use simulation only; this report is not a clinical safety determination.",
            "Results depend on the configured electrode geometry, tissue model, and safety limits.",
            "Model validation and regulatory review remain the responsibility of the researcher.",
        ),
        metrics_path=str(path),
    )

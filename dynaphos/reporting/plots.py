from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_standard_plots(metrics_path: str | Path, output_dir: str | Path) -> list[Path]:
    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(metrics_path, allow_pickle=True) as bundle:
        data = {key: np.asarray(bundle[key]) for key in bundle.files}

    written: list[Path] = []
    time_s = np.asarray(data.get("time_s", []), dtype=np.float64)
    amplitude = data.get("amplitude_per_electrode_uA")
    if amplitude is not None and amplitude.ndim == 2 and len(time_s) == amplitude.shape[0]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(time_s, np.max(amplitude, axis=1), label="maximum")
        ax.plot(time_s, np.mean(amplitude, axis=1), label="mean")
        ax.set(xlabel="Time (s)", ylabel="Current (uA)", title="Delivered stimulation")
        ax.legend()
        ax.grid(alpha=0.25)
        path = output_dir / "delivered_stimulation.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    thermal_time = np.asarray(data.get("thermal_time_s", []), dtype=np.float64)
    max_dt = np.asarray(data.get("max_dT", []), dtype=np.float64)
    mean_dt = np.asarray(data.get("mean_dT", []), dtype=np.float64)
    if len(thermal_time) and len(thermal_time) == len(max_dt):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(thermal_time, max_dt, label="focal maximum")
        if len(mean_dt) == len(thermal_time):
            ax.plot(thermal_time, mean_dt, label="spatial mean")
        ax.set(xlabel="Time (s)", ylabel="Temperature rise (degC)", title="Thermal response")
        ax.legend()
        ax.grid(alpha=0.25)
        path = output_dir / "thermal_response.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written

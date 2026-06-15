import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dynaphos.reporting import write_report_bundle
from dynaphos.safety.evaluation import EXCEEDS_LIMIT


def test_report_bundle_records_status_and_peak_location(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.npz"
    np.savez(
        metrics,
        time_s=np.asarray([0.1, 0.2], dtype=np.float32),
        thermal_time_s=np.asarray([0.1, 0.2], dtype=np.float32),
        electrode_ids=np.asarray([10, 11], dtype=np.int64),
        amplitude_per_electrode_uA=np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        charge_per_phase_per_electrode_nC=np.asarray(
            [[10.0, 0.0], [10.0, 101.234]], dtype=np.float32
        ),
        window_charge_per_electrode_nC=np.zeros((2, 2), dtype=np.float32),
        window_charge_total_nC=np.zeros(2, dtype=np.float32),
        charge_window_s=np.asarray(5.0, dtype=np.float32),
        protocol_charge_per_electrode_nC=np.asarray(
            [1_000_000.0, 500_000.0],
            dtype=np.float32,
        ),
        electrode_xy_mm=np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32),
        max_dT=np.asarray([0.1, 0.234], dtype=np.float32),
        mean_dT=np.asarray([0.05, 0.123], dtype=np.float32),
        max_cem43=np.asarray([0.0, 0.0123], dtype=np.float32),
        peak_heatmap_grid_names=np.asarray(["800um"]),
        peak_focal_time_s=np.asarray(0.2, dtype=np.float32),
        peak_mean_time_s=np.asarray(0.2, dtype=np.float32),
        dT_peak_focal_800um=np.asarray(
            [[0.1, 0.15], [0.2, 0.234]],
            dtype=np.float32,
        ),
        dT_peak_mean_800um=np.asarray(
            [[0.11, 0.12], [0.13, 0.14]],
            dtype=np.float32,
        ),
        extent_mm_800um=np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
    )

    report = write_report_bundle(tmp_path, metrics_path=metrics, write_figures=True)

    charge = next(metric for metric in report.metrics if metric.name == "charge_per_phase")
    window_charge = next(
        metric
        for metric in report.metrics
        if metric.name == "window_charge_per_electrode"
    )
    session_charge = next(
        metric for metric in report.metrics if metric.name == "session_charge"
    )
    mean_temperature = next(
        metric for metric in report.metrics if metric.name == "mean_temperature"
    )
    cem43 = next(metric for metric in report.metrics if metric.name == "cem43")
    assert charge.status == EXCEEDS_LIMIT
    assert charge.peak_time_s == pytest.approx(0.2)
    assert charge.electrode_id == 11
    assert window_charge.window_duration_s == pytest.approx(5.0)
    assert session_charge.observed == pytest.approx(1.5)
    assert mean_temperature.observed == pytest.approx(0.123)
    assert cem43.label == "CEM43"
    assert len(report.visualizations) == 5

    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "configured_limits_exceeded"
    payload_charge = next(
        metric for metric in payload["metrics"] if metric["name"] == "charge_per_phase"
    )
    payload_window = next(
        metric
        for metric in payload["metrics"]
        if metric["name"] == "window_charge_per_electrode"
    )
    payload_mean = next(
        metric for metric in payload["metrics"] if metric["name"] == "mean_temperature"
    )
    assert payload_charge["observed"] == 101.23
    assert payload_charge["observed_with_unit"] == "101.23 nC"
    assert payload_charge["ratio_band"] == "exceeds"
    assert payload_window["window_duration_s"] == 5.0
    assert payload_mean["observed"] == 0.12
    assert {
        visualization["name"] for visualization in payload["visualizations"]
    } >= {
        "total_charge_heatmap",
        "max_focal_temperature_heatmap",
        "max_mean_temperature_heatmap",
    }

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "µA" in html
    assert "µC/cm²" in html
    assert "°C" in html
    assert "CEM43" in html
    assert "5.00 s rolling window" in html
    assert "ratio exceeds" in html
    assert "101.23 nC" in html
    assert "figures/max_mean_temperature_heatmap.png" in html

    with open(tmp_path / "summary.csv", newline="", encoding="utf-8") as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert rows["charge_per_phase"]["observed"] == "101.23"
    assert rows["charge_per_phase"]["observed_with_unit"] == "101.23 nC"
    assert rows["window_charge_per_electrode"]["window_duration_s"] == "5.00"
    assert (
        rows["mean_temperature"]["visualization_paths"]
        == "figures/max_mean_temperature_heatmap.png"
    )

    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "figures" / "total_charge_heatmap.png").exists()
    assert (tmp_path / "figures" / "max_focal_temperature_heatmap.png").exists()
    assert (tmp_path / "figures" / "max_mean_temperature_heatmap.png").exists()

import json
from pathlib import Path

import numpy as np
import pytest

from dynaphos.reporting import write_report_bundle
from dynaphos.safety.evaluation import EXCEEDS_LIMIT, NOT_EVALUATED


def test_report_bundle_records_status_and_peak_location(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.npz"
    np.savez(
        metrics,
        time_s=np.asarray([0.1, 0.2], dtype=np.float32),
        thermal_time_s=np.asarray([0.1, 0.2], dtype=np.float32),
        electrode_ids=np.asarray([10, 11], dtype=np.int64),
        amplitude_per_electrode_uA=np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        charge_per_phase_per_electrode_nC=np.asarray(
            [[10.0, 0.0], [10.0, 101.0]], dtype=np.float32
        ),
        window_charge_per_electrode_nC=np.zeros((2, 2), dtype=np.float32),
        window_charge_total_nC=np.zeros(2, dtype=np.float32),
        max_dT=np.asarray([0.1, 0.2], dtype=np.float32),
    )

    report = write_report_bundle(tmp_path, metrics_path=metrics, write_figures=False)

    charge = next(metric for metric in report.metrics if metric.name == "charge_per_phase")
    cem43 = next(metric for metric in report.metrics if metric.name == "cem43")
    assert charge.status == EXCEEDS_LIMIT
    assert charge.peak_time_s == pytest.approx(0.2)
    assert charge.electrode_id == 11
    assert cem43.status == NOT_EVALUATED
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "configured_limits_exceeded"
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "summary.csv").exists()

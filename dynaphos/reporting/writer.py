from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from dynaphos.reporting.html import write_html
from dynaphos.reporting.json import write_json
from dynaphos.reporting.plots import write_standard_plots
from dynaphos.safety.evaluation import SafetyReport, evaluate_metrics


def write_summary_csv(path: str | Path, report: SafetyReport) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "metric",
                "observed",
                "limit",
                "unit",
                "ratio_to_limit",
                "status",
                "peak_time_s",
                "electrode_id",
            ),
        )
        writer.writeheader()
        for metric in report.metrics:
            writer.writerow(
                {
                    "metric": metric.name,
                    "observed": metric.observed,
                    "limit": metric.limit,
                    "unit": metric.unit,
                    "ratio_to_limit": metric.ratio_to_limit,
                    "status": metric.status,
                    "peak_time_s": metric.peak_time_s,
                    "electrode_id": metric.electrode_id,
                }
            )
    return output


def write_report_bundle(
    run_dir: str | Path,
    *,
    metrics_path: str | Path | None = None,
    safety_limits_path: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
    write_figures: bool = True,
) -> SafetyReport:
    output_dir = Path(run_dir).resolve()
    metrics = Path(metrics_path or output_dir / "metrics.npz").resolve()
    report = evaluate_metrics(metrics, safety_limits_path=safety_limits_path)
    write_json(output_dir / "report.json", report.to_dict())
    write_html(output_dir / "report.html", report, manifest=manifest)
    write_summary_csv(output_dir / "summary.csv", report)
    if write_figures:
        write_standard_plots(metrics, output_dir / "figures")
    return report

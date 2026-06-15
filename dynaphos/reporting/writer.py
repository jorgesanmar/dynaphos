from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from dynaphos.reporting.html import write_html
from dynaphos.reporting.json import write_json
from dynaphos.reporting.plots import write_standard_plots
from dynaphos.safety.evaluation import (
    SafetyReport,
    evaluate_metrics,
    format_value_with_unit,
    ratio_band,
)


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def write_summary_csv(path: str | Path, report: SafetyReport) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "metric",
                "label",
                "observed",
                "observed_with_unit",
                "limit",
                "unit",
                "ratio_to_limit",
                "ratio_band",
                "status",
                "peak_time_s",
                "electrode_id",
                "window_duration_s",
                "visualization_paths",
            ),
        )
        writer.writeheader()
        for metric in report.metrics:
            visualization_paths = ";".join(
                visualization.path
                for visualization in report.visualizations
                if visualization.metric_name == metric.name
            )
            writer.writerow(
                {
                    "metric": metric.name,
                    "label": metric.label,
                    "observed": _format_number(metric.observed),
                    "observed_with_unit": format_value_with_unit(
                        metric.observed,
                        metric.unit,
                    ),
                    "limit": _format_number(metric.limit),
                    "unit": metric.unit,
                    "ratio_to_limit": _format_number(metric.ratio_to_limit),
                    "ratio_band": ratio_band(metric.ratio_to_limit),
                    "status": metric.status,
                    "peak_time_s": _format_number(metric.peak_time_s),
                    "electrode_id": metric.electrode_id,
                    "window_duration_s": _format_number(metric.window_duration_s),
                    "visualization_paths": visualization_paths,
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
    visualizations = (
        write_standard_plots(metrics, output_dir / "figures")
        if write_figures
        else []
    )
    report = replace(report, visualizations=tuple(visualizations))
    write_json(output_dir / "report.json", report.to_dict())
    write_html(output_dir / "report.html", report, manifest=manifest)
    write_summary_csv(output_dir / "summary.csv", report)
    return report

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from dynaphos.safety.evaluation import NOT_EVALUATED, SafetyReport


def _format(value: float | int | None) -> str:
    if value is None:
        return "Not evaluated"
    return f"{value:.6g}"


def write_html(
    path: str | Path,
    report: SafetyReport,
    *,
    manifest: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    rows = []
    for metric in report.metrics:
        location = []
        if metric.peak_time_s is not None:
            location.append(f"t={metric.peak_time_s:.3f} s")
        if metric.electrode_id is not None:
            location.append(f"electrode {metric.electrode_id}")
        rows.append(
            "<tr>"
            f"<td>{escape(metric.label)}</td>"
            f"<td>{escape(_format(metric.observed))}</td>"
            f"<td>{escape(_format(metric.limit))}</td>"
            f"<td>{escape(metric.unit)}</td>"
            f"<td>{escape(_format(metric.ratio_to_limit))}</td>"
            f"<td><span class='{escape(metric.status)}'>{escape(metric.status.replace('_', ' '))}</span></td>"
            f"<td>{escape(', '.join(location) or '-')}</td>"
            "</tr>"
        )
    limitations = "".join(f"<li>{escape(item)}</li>" for item in report.limitations)
    run_id = escape(str((manifest or {}).get("run_id", "DynaPhos experiment")))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DynaPhos safety report - {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #18212b; }}
    h1, h2 {{ color: #173f66; }}
    .status {{ border-left: 6px solid #173f66; background: #eef4f8; padding: 1rem; font-size: 1.2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid #d7dde3; padding: .65rem; text-align: left; }}
    th {{ background: #f4f6f8; }}
    .within_limit {{ color: #176b3a; font-weight: 600; }}
    .exceeds_limit {{ color: #9b2525; font-weight: 700; }}
    .not_evaluated {{ color: #6b7280; font-weight: 600; }}
    .notice {{ background: #fff7df; border: 1px solid #e8d28d; padding: 1rem; }}
    footer {{ margin-top: 2rem; color: #6b7280; }}
  </style>
</head>
<body>
  <h1>DynaPhos Safety Report</h1>
  <p><strong>Run:</strong> {run_id}</p>
  <div class="status"><strong>Overall result:</strong> {escape(report.status_label)}</div>
  <h2>Metric Evaluation</h2>
  <table>
    <thead><tr><th>Metric</th><th>Observed</th><th>Limit</th><th>Unit</th><th>Ratio</th><th>Status</th><th>Peak location</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Model Limitations</h2>
  <div class="notice"><ul>{limitations}</ul></div>
  <footer>Generated from {escape(report.metrics_path)}</footer>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output

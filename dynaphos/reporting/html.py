from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from dynaphos.safety.evaluation import (
    SafetyReport,
    format_value_with_unit,
    ratio_band,
)


def _format_value(value: float | None, unit: str = "") -> str:
    if value is None:
        return "Not evaluated"
    return format_value_with_unit(value, unit) or "Not evaluated"


def _format_ratio(value: float | None) -> str:
    return "Not evaluated" if value is None else f"{value:.2f}×"


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
            location.append(f"t={metric.peak_time_s:.2f} s")
        if metric.electrode_id is not None:
            location.append(f"electrode {metric.electrode_id}")
        context = (
            f"<div class='metric-context'>{metric.window_duration_s:.2f} s rolling window</div>"
            if metric.window_duration_s is not None
            else ""
        )
        metric_visualizations = [
            visualization
            for visualization in report.visualizations
            if visualization.metric_name == metric.name
        ]
        visualization_links = " ".join(
            f"<a href='{escape(visualization.path, quote=True)}'>View</a>"
            for visualization in metric_visualizations
        ) or "-"
        rows.append(
            "<tr>"
            f"<td><strong>{escape(metric.label)}</strong>{context}</td>"
            f"<td>{escape(_format_value(metric.observed, metric.unit))}</td>"
            f"<td>{escape(_format_value(metric.limit, metric.unit))}</td>"
            f"<td><span class='ratio {ratio_band(metric.ratio_to_limit)}'>"
            f"{escape(_format_ratio(metric.ratio_to_limit))}</span></td>"
            f"<td><span class='{escape(metric.status)}'>"
            f"{escape(metric.status.replace('_', ' '))}</span></td>"
            f"<td>{escape(', '.join(location) or '-')}</td>"
            f"<td>{visualization_links}</td>"
            "</tr>"
        )

    limitations = "".join(f"<li>{escape(item)}</li>" for item in report.limitations)
    run_id = escape(str((manifest or {}).get("run_id", "DynaPhos experiment")))
    visualization_cards = "".join(
        (
            "<figure>"
            f"<a href='{escape(visualization.path, quote=True)}'>"
            f"<img src='{escape(visualization.path, quote=True)}' "
            f"alt='{escape(visualization.label, quote=True)}' loading='lazy'></a>"
            f"<figcaption><strong>{escape(visualization.label)}</strong>"
            f"<br>{escape(visualization.description)}</figcaption>"
            "</figure>"
        )
        for visualization in report.visualizations
    )
    visualization_section = (
        f"<h2>Visualizations</h2><div class='visualizations'>{visualization_cards}</div>"
        if visualization_cards
        else ""
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DynaPhos safety report - {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; color: #18212b; }}
    h1, h2 {{ color: #173f66; }}
    .status {{ border-left: 6px solid #173f66; background: #eef4f8; padding: 1rem; font-size: 1.2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid #d7dde3; padding: .65rem; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f8; }}
    .within_limit {{ color: #176b3a; font-weight: 600; }}
    .exceeds_limit {{ color: #9b2525; font-weight: 700; }}
    .not_evaluated {{ color: #6b7280; font-weight: 600; }}
    .ratio {{ border-radius: 999px; display: inline-block; min-width: 5.8rem; padding: .2rem .55rem; text-align: center; font-weight: 700; }}
    .ratio.within {{ background: #dcfce7; color: #166534; }}
    .ratio.approaching {{ background: #fef3c7; color: #92400e; }}
    .ratio.exceeds {{ background: #fee2e2; color: #991b1b; }}
    .ratio.not_evaluated {{ background: #e5e7eb; color: #4b5563; }}
    .metric-context {{ color: #5f6b76; font-size: .85rem; margin-top: .15rem; }}
    .notice {{ background: #fff7df; border: 1px solid #e8d28d; padding: 1rem; }}
    .visualizations {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
    figure {{ border: 1px solid #d7dde3; border-radius: .35rem; margin: 0; padding: .75rem; background: #fff; }}
    figure img {{ display: block; height: auto; max-width: 100%; }}
    figcaption {{ color: #4b5563; line-height: 1.35; padding-top: .55rem; }}
    footer {{ margin-top: 2rem; color: #6b7280; }}
  </style>
</head>
<body>
  <h1>DynaPhos Safety Report</h1>
  <p><strong>Run:</strong> {run_id}</p>
  <div class="status"><strong>Overall result:</strong> {escape(report.status_label)}</div>
  <h2>Metric Evaluation</h2>
  <table>
    <thead><tr><th>Metric</th><th>Observed</th><th>Limit</th><th>Ratio</th><th>Status</th><th>Peak location</th><th>Visualization</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {visualization_section}
  <h2>Model Limitations</h2>
  <div class="notice"><ul>{limitations}</ul></div>
  <footer>Generated from {escape(report.metrics_path)}</footer>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output

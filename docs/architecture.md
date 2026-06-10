# Package Organization

DynaPhos 0.2 groups code by responsibility.

- `dynaphos.experiment`: configuration-driven execution, manifests, metrics,
  completed-run discovery, and sweeps.
- `dynaphos.simulation`: phosphene simulation, cortical mapping, coordinates,
  and raster grouping.
- `dynaphos.media`: input resolution, preprocessing, comparison, and rendering.
- `dynaphos.electrodes`: array geometry, loading, and visualization.
- `dynaphos.safety`: impedance, electrical and thermal tracking, limits, and
  evaluation. It does not orchestrate experiments or studies.
- `dynaphos.reporting`: generic per-run reports and dashboards.
- `dynaphos.studies`: phase 1-3 cross-run analysis.
- `dynaphos.strategies`: the stimulation strategy API and built-ins.

Root modules no longer duplicate these implementations. This keeps dependency
direction clear: experiments compose simulation, media, strategies, safety,
and reporting; studies consume completed experiment artifacts.

## Tools

`tools/` is not an alternate application package. It contains standalone,
maintainer-oriented research experiments. The retained Bioheat3D scripts live
under `tools/experiments/bioheat3d/` and deliberately remain outside pytest
discovery and the supported Python API.

Reusable commands belong to the installed `dynaphos` CLI. Rendering,
preprocessing comparison, electrode visualization, sweeps, reporting, and
phase analysis can all be launched there.

## Run Contract

A run is available for analysis only when:

1. `manifest.yaml` has `schema_version: 1`;
2. its status is `completed`; and
3. `metrics.npz` exists.

Failed and incomplete runs are visible warnings during discovery. Missing
metrics for a completed run indicate corruption and raise an error.

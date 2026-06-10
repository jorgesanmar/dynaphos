# Migrating to DynaPhos 0.2.0

Version 0.2.0 removes the compatibility layer used while phase results were
being regenerated.

## Artifact Names

Use `manifest.yaml` and `metrics.npz`. The following files are no longer
written or read by supported workflows:

- `run_manifest.yaml`
- `safety_metrics.npz`
- per-run `summary.txt`

Report artifacts remain `report.html`, `report.json`, `summary.csv`, and
`figures/`.

## Python Imports

Replace removed root imports as follows:

- `dynaphos.simulator` -> `dynaphos.simulation`
- `dynaphos.cortex_models` -> `dynaphos.simulation.cortex`
- `dynaphos.utils` -> `dynaphos.simulation.utils`
- `dynaphos.image_processing` -> `dynaphos.media.preprocessing`
- `dynaphos.pipeline` -> `dynaphos.media.rendering`
- `dynaphos.electrode_grid` -> `dynaphos.electrodes`
- `dynaphos.safety.runner` -> `dynaphos.experiment.execution`
- safety phase visualizers -> `dynaphos.studies`
- `dynaphos.safety.single_case` -> `dynaphos.reporting.dashboard`

The old safety orchestration and compatibility I/O modules have been removed.

## Scripts

Deprecated phase runners and visualizers under `tools/safety/`, all
`tools/legacy/` scripts, and `run_study_sweeps_task.py` have been removed.
Use:

```text
dynaphos sweep SWEEP.yaml --resume
dynaphos study phase1 RESULTS_ROOT
dynaphos study phase2 RESULTS_ROOT --phase1-root PHASE1_ROOT
dynaphos study phase3 RESULTS_ROOT --phase1-root PHASE1_ROOT --phase2-root PHASE2_ROOT
```

Standalone Bioheat3D experiments moved to `tools/experiments/bioheat3d/`; their
filenames no longer begin with `test_`.

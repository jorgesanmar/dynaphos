# Reproducing the Phase Studies

Study definitions live under `examples/studies/`. Comparative analysis is
implemented in `dynaphos.studies` and reads only completed canonical runs:
`manifest.yaml` plus `metrics.npz`.

## Phase 1

```bash
dynaphos sweep examples/studies/phase1/sweep.yaml
dynaphos study phase1 \
  results/safety/simulation_pipeline/amplitude_grid_preprocessing
```

This expands to 27 amplitude, electrode-array, and preprocessing combinations.

## Phase 2

```bash
dynaphos sweep examples/studies/phase2/ic_only_sweep.yaml
dynaphos sweep examples/studies/phase2/additivity_sweep.yaml
dynaphos study phase2 \
  results/safety/simulation_pipeline/ic_power \
  --phase1-root results/safety/simulation_pipeline/amplitude_grid_preprocessing
```

These expand to 21 IC-only cases and 6 additivity cases.

## Phase 3

```bash
dynaphos sweep examples/studies/phase3/screen_sweep.yaml
dynaphos sweep examples/studies/phase3/transfer_sweep.yaml
dynaphos study phase3 \
  results/safety/simulation_pipeline/raster_protocols \
  --phase1-root results/safety/simulation_pipeline/amplitude_grid_preprocessing \
  --phase2-root results/safety/simulation_pipeline/ic_power
```

These expand to 6 screening and 20 transfer cases.

Each analysis writes CSV summaries and comparative figures to
`<results-root>/comparative_visuals` by default. Use `--output-root` to keep
generated analysis outside an existing results tree.

Directories whose manifest status is not `completed` are skipped with a
warning. A completed manifest without `metrics.npz` is treated as corrupt and
stops the analysis.

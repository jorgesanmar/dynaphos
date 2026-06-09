# Reproducing the Existing Studies

The original phase-specific configurations and scripts remain available during
the compatibility period. General experiment equivalents live under
`examples/studies/`.

## Phase 1

```bash
dynaphos sweep examples/studies/phase1/sweep.yaml
```

This expands to 27 amplitude, electrode-array, and preprocessing combinations.

## Phase 2

```bash
dynaphos sweep examples/studies/phase2/ic_only_sweep.yaml
dynaphos sweep examples/studies/phase2/additivity_sweep.yaml
```

These expand to 21 IC-only cases and 6 additivity cases.

## Phase 3

```bash
dynaphos sweep examples/studies/phase3/screen_sweep.yaml
dynaphos sweep examples/studies/phase3/transfer_sweep.yaml
```

These expand to 6 screening and 20 transfer cases.

Study-specific comparative plots in `ic_power_visualize.py` and
`raster_effects.py` continue to consume the compatibility
`safety_metrics.npz` files.

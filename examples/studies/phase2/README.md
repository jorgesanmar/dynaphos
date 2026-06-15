# Phase 2: Internal-Circuit Power

Run the IC-only linearity cases and additivity cases separately:

```bash
dynaphos sweep examples/studies/phase2/ic_only_sweep.yaml
dynaphos sweep examples/studies/phase2/additivity_sweep.yaml
```

IC-only cases automatically skip video decoding and phosphene stimulation.
They advance the Pennes model using only the configured constant IC power.
The IC-only sweep uses the base `coords_800um` grid because all three electrode
grids have the same device footprint and therefore produce identical IC-only
thermal results. The sweep contains seven runs, one per IC power level.
Video-driven additivity cases can reuse the shared stimulation cache.

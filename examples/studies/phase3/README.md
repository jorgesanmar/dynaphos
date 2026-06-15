# Phase 3: Raster Strategies

```bash
dynaphos sweep examples/studies/phase3/screen_sweep.yaml
dynaphos sweep examples/studies/phase3/transfer_sweep.yaml
dynaphos sweep examples/studies/phase3/transfer_groups3_sweep.yaml --resume
```

The general strategy interface replaces phase-specific raster orchestration.

Phase 3 reuses amplitude-independent stimulation traces from phase 1 whenever
the video, preprocessing, electrode grid, resolution, and sampling setup match.
Raster masks are still applied per phase 3 run, so checkerboard and
pseudo-random outputs remain protocol-specific.

The focused `transfer_groups3_sweep.yaml` adds the three-group protocol to all
transfer-panel conditions without changing the run identities of the completed
four- and five-group transfer sweeps.

# Phase 1: Amplitude, Grid, and Preprocessing

This is the general-config equivalent of
`config/safety_experiments_phase1_amp_grid_prep.yaml`.

```bash
dynaphos sweep examples/studies/phase1/sweep.yaml
```

It expands to the same 27 amplitude, grid, and preprocessing combinations.

The first run for each video, preprocessing, and electrode-grid combination
stores its amplitude-independent stimulation trace under
`results/safety/simulation_pipeline/.stimulation_cache`. Later amplitude runs,
and matching phase 3 runs, reuse that trace while recomputing their electrical
and thermal results.

Delete `.stimulation_cache` to force all stimulation traces to be rebuilt.

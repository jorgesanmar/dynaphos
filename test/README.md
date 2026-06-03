# Test Guide

The focused tests cover the package-owned pipeline and the thin tool wrappers.

Run them from the repository root with:

```bash
pytest
```

Canonical entry points now live under:

- `tools/preprocessing/visualize_preprocessing.py`
- `tools/electrode_grid/visualize_electrode_grid.py`
- `tools/phosphenes/render_image.py`
- `tools/phosphenes/render_video.py`
- `tools/safety/run_phase1.py`
- `tools/safety/visualize.py`

Reusable implementation code lives in `dynaphos/`, especially
`dynaphos.pipeline`, `dynaphos.electrode_grid`, `dynaphos.simulator`, and
`dynaphos.safety`.

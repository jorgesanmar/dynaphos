# Test Guide

The focused tests cover the package-owned pipeline and the thin tool wrappers.

Run them from the repository root with:

```bash
pytest
```

Canonical entry points now live under:

- `dynaphos validate <experiment.yaml>`
- `dynaphos run <experiment.yaml>`
- `dynaphos sweep <sweep.yaml> --resume`
- `dynaphos report <run-directory>`
- `dynaphos strategies list`
- `dynaphos render image|video`
- `dynaphos preprocess compare`
- `dynaphos electrodes plot`
- `dynaphos study phase1|phase2|phase3`

Reusable implementation code lives in `dynaphos/`, especially
`dynaphos.config`, `dynaphos.experiment`, `dynaphos.strategies`,
`dynaphos.simulation`, `dynaphos.media`, `dynaphos.electrodes`,
`dynaphos.safety`, `dynaphos.reporting`, and `dynaphos.studies`.

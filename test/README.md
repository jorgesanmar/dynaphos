# Test Guide

The focused tests cover the package-owned pipeline and the thin tool wrappers.

Run them from the repository root with:

```bash
pytest
```

Canonical entry points now live under:

- `dynaphos validate <experiment.yaml>`
- `dynaphos run <experiment.yaml>`
- `dynaphos sweep <sweep.yaml>`
- `dynaphos report <run-directory>`
- `dynaphos strategies list`

Reusable implementation code lives in `dynaphos/`, especially
`dynaphos.config`, `dynaphos.experiment`, `dynaphos.strategies`,
`dynaphos.simulation`, `dynaphos.safety`, and `dynaphos.reporting`.

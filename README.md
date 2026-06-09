# DynaPhos

DynaPhos simulates cortical visual prosthesis stimulation and produces
electrical and thermal safety reports for research protocols.

The repository now has one config-first interface for individual experiments,
parameter sweeps, custom stimulation strategies, and report generation.

## Install

```bash
pip install -e .
```

Development dependencies:

```bash
pip install -e ".[dev]"
```

## Five-Minute Example

The minimal example uses a packaged image, a built-in electrode array, and a
checkerboard strategy:

```bash
dynaphos validate examples/experiments/minimal.yaml
dynaphos run examples/experiments/minimal.yaml
```

The run writes:

```text
results/minimal_example/
  manifest.yaml
  report.html
  report.json
  summary.csv
  metrics.npz
  safety_metrics.npz
  figures/
```

`report.html` is the main human-readable result. `report.json` and
`summary.csv` provide machine-readable evaluations, while `metrics.npz`
contains detailed arrays. `safety_metrics.npz` is retained for compatibility
with existing analysis scripts.

Reports use the terms **within configured limits**, **configured limits
exceeded**, and **incomplete evaluation**. They are research simulation
outputs, not clinical safety determinations.

## CLI

```text
dynaphos validate experiment.yaml
dynaphos run experiment.yaml
dynaphos sweep sweep.yaml
dynaphos report results/my_run
dynaphos strategies list
```

## Python API

```python
import dynaphos

config = dynaphos.load_experiment("examples/experiments/minimal.yaml")
result = dynaphos.run_experiment(config)
print(result.report.status)
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Experiment configuration](docs/configuration.md)
- [Writing stimulation strategies](docs/strategies.md)
- [Understanding safety reports](docs/reports.md)
- [Python API](docs/python-api.md)
- [Reproducing the phase 1-3 studies](docs/studies.md)
- [Scientific methodology and assumptions](methodology.md)

## Existing Workflows

The phase-specific scripts under `tools/safety/` remain available for one
compatibility release, but are deprecated in favor of the study definitions
under `examples/studies/`. Existing numerical modules and raw NPZ outputs remain
supported during this migration.

## Tests

```bash
pytest
```

## Citation

van der Grinten, M. et al. (2024). *Towards biologically plausible phosphene
simulation for the differentiable optimization of visual cortical prostheses*.
eLife, 13, e85812. https://doi.org/10.7554/eLife.85812

## License

GNU General Public License v3. See [LICENSE](LICENSE).

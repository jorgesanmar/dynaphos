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
  metrics.npz
  report.html
  report.json
  summary.csv
  figures/
```

`report.html` is the main human-readable result. `report.json` and
`summary.csv` provide machine-readable evaluations, while `metrics.npz`
contains the canonical detailed arrays. Completed runs use only
`manifest.yaml` and `metrics.npz` as their persisted analysis contract.

Reports use the terms **within configured limits**, **configured limits
exceeded**, and **incomplete evaluation**. They are research simulation
outputs, not clinical safety determinations.

## CLI

```text
dynaphos validate experiment.yaml
dynaphos run experiment.yaml
dynaphos sweep sweep.yaml --resume
dynaphos report results/my_run
dynaphos strategies list
dynaphos render image --input stimulus.png
dynaphos render video --input stimulus.mp4
dynaphos preprocess compare --input stimulus.mp4
dynaphos electrodes plot
dynaphos study phase1 results/safety/simulation_pipeline/amplitude_grid_preprocessing
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
- [Package organization](docs/architecture.md)
- [Migrating to 0.2.0](docs/migration-0.2.md)
- [Scientific methodology and assumptions](methodology.md)

## Repository Organization

Reusable code is organized by domain under `dynaphos/`: experiment execution,
simulation, media, electrodes, safety evaluation, reporting, studies, and
stimulation strategies. The `tools/` directory contains standalone research
experiments that are not part of the supported Python API. Phase comparison
workflows live in `dynaphos.studies`, not in `tools/`.

## Tests

```bash
pytest
```


## License

GNU General Public License v3. See [LICENSE](LICENSE).

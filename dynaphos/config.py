from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from itertools import product
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


T = TypeVar("T", bound="StrictConfig")


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    origin = get_origin(annotation)
    if origin not in (Union, UnionType):
        return False, annotation
    args = get_args(annotation)
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) != len(args) and len(non_none) == 1:
        return True, non_none[0]
    return False, annotation


def _convert_value(annotation: Any, value: Any, path: str) -> Any:
    optional, inner = _is_optional(annotation)
    if value is None:
        if optional:
            return None
        raise ValueError(f"{path} may not be null.")
    annotation = inner if optional else annotation
    origin = get_origin(annotation)

    if isinstance(annotation, type) and issubclass(annotation, StrictConfig):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping.")
        return annotation.from_dict(value, path=path)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list.")
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [_convert_value(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping.")
        return dict(value)
    if annotation is Path:
        return Path(value)
    if annotation is Any:
        return value
    if annotation in (str, int, float, bool):
        if annotation is bool and not isinstance(value, bool):
            raise ValueError(f"{path} must be true or false.")
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must be a valid {annotation.__name__}.") from exc
    return value


@dataclass
class StrictConfig:
    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for config_field in fields(self):
            value = getattr(self, config_field.name)
            if value is None:
                continue
            optional, annotation = _is_optional(hints.get(config_field.name, Any))
            _ = optional
            if annotation is Path and not isinstance(value, Path):
                setattr(self, config_field.name, Path(value))
        self.validate(type(self).__name__)

    @classmethod
    def from_dict(cls: type[T], values: Mapping[str, Any], *, path: str = "config") -> T:
        if not isinstance(values, Mapping):
            raise ValueError(f"{path} must be a mapping.")
        config_fields = {item.name: item for item in fields(cls) if item.init}
        unknown = sorted(set(values) - set(config_fields))
        if unknown:
            raise ValueError(f"{path} contains unknown field(s): {', '.join(unknown)}.")

        hints = get_type_hints(cls)
        converted: dict[str, Any] = {}
        for name, value in values.items():
            converted[name] = _convert_value(hints.get(name, Any), value, f"{path}.{name}")
        instance = cls(**converted)
        instance.validate(path)
        return instance

    def validate(self, path: str = "config") -> None:
        _ = path

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value):
                return {key: convert(item) for key, item in asdict(value).items()}
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(self)


@dataclass
class InputConfig(StrictConfig):
    path: Path | None = None
    builtin: str | None = None
    stage: str = "preprocessed"
    preprocessing_method: str = "groundtruth"
    preprocessing_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], *, path: str = "input"):
        normalized = dict(values)
        if normalized.get("path") and "builtin" not in normalized:
            normalized["builtin"] = None
        return super().from_dict(normalized, path=path)

    def validate(self, path: str = "input") -> None:
        if bool(self.path) == bool(self.builtin):
            raise ValueError(f"{path} must define exactly one of 'path' or 'builtin'.")
        if self.stage not in {"original", "preprocessed"}:
            raise ValueError(f"{path}.stage must be 'original' or 'preprocessed'.")
        if not self.preprocessing_method.strip():
            raise ValueError(f"{path}.preprocessing_method may not be empty.")


@dataclass
class ElectrodeArrayConfig(StrictConfig):
    coordinates: Path | None = None
    builtin: str | None = "coords_800um"

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], *, path: str = "electrode_array"):
        normalized = dict(values)
        if normalized.get("coordinates") and "builtin" not in normalized:
            normalized["builtin"] = None
        return super().from_dict(normalized, path=path)

    def validate(self, path: str = "electrode_array") -> None:
        if bool(self.coordinates) == bool(self.builtin):
            raise ValueError(
                f"{path} must define exactly one of 'coordinates' or 'builtin'."
            )


@dataclass
class ProtocolConfig(StrictConfig):
    amplitude_uA: float = 60.0
    appearance_threshold_uA: float = 30.0
    pulse_width_us: float = 170.0
    frequency_hz: float = 300.0
    relative_stim_duration: float = 1.0
    internal_circuit_power_mW: float = 0.0

    def validate(self, path: str = "protocol") -> None:
        for name in (
            "amplitude_uA",
            "appearance_threshold_uA",
            "pulse_width_us",
            "frequency_hz",
            "internal_circuit_power_mW",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{path}.{name} must be non-negative.")
        if not 0.0 <= float(self.relative_stim_duration) <= 1.0:
            raise ValueError(f"{path}.relative_stim_duration must be between 0 and 1.")


@dataclass
class StrategyConfig(StrictConfig):
    name: str | None = "direct"
    import_path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], *, path: str = "strategy"):
        normalized = dict(values)
        if normalized.get("import_path") and "name" not in normalized:
            normalized["name"] = None
        return super().from_dict(normalized, path=path)

    def validate(self, path: str = "strategy") -> None:
        if bool(self.name) == bool(self.import_path):
            raise ValueError(f"{path} must define exactly one of 'name' or 'import_path'.")


@dataclass
class SimulationConfig(StrictConfig):
    params: Path | None = None
    seed: int = 42
    force_cpu: bool = False
    resolution: int | None = None
    max_frames: int = 0
    preview_seconds: float = 0.0
    phosphene_mode: str = "safety_centers"
    cooldown_seconds: float = 300.0
    cooldown_baseline_tolerance_C: float = 1e-3
    enable_cem43: bool = False

    def validate(self, path: str = "simulation") -> None:
        if self.resolution is not None and self.resolution <= 0:
            raise ValueError(f"{path}.resolution must be greater than zero.")
        if self.max_frames < 0:
            raise ValueError(f"{path}.max_frames must be non-negative.")
        if self.preview_seconds < 0.0 or self.cooldown_seconds < 0.0:
            raise ValueError(f"{path} durations must be non-negative.")
        if self.cooldown_baseline_tolerance_C < 0.0:
            raise ValueError(f"{path}.cooldown_baseline_tolerance_C must be non-negative.")
        if self.phosphene_mode not in {"safety_centers", "visual"}:
            raise ValueError(
                f"{path}.phosphene_mode must be 'safety_centers' or 'visual'."
            )


@dataclass
class SafetyConfig(StrictConfig):
    limits: Path | None = None
    track_electrical: bool = True
    track_thermal: bool = True
    electrode_heat_enabled: bool = True

    def validate(self, path: str = "safety") -> None:
        if not self.track_thermal:
            raise ValueError(
                f"{path}.track_thermal must remain true because the standard "
                "safety report requires thermal evaluation."
            )


@dataclass
class OutputConfig(StrictConfig):
    root: Path = Path("results")
    run_id: str = "experiment"
    overwrite: bool = False
    write_figures: bool = True

    def validate(self, path: str = "output") -> None:
        if not self.run_id.strip():
            raise ValueError(f"{path}.run_id may not be empty.")
        if any(part in {".", ".."} for part in Path(self.run_id).parts):
            raise ValueError(f"{path}.run_id must be a simple relative name.")


@dataclass
class ExperimentConfig(StrictConfig):
    input: InputConfig
    electrode_array: ElectrodeArrayConfig = field(default_factory=ElectrodeArrayConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = field(default=None, repr=False, init=False)

    def resolve_paths(self, source_path: Path) -> "ExperimentConfig":
        resolved = copy.deepcopy(self)
        base = source_path.resolve().parent

        def resolve(value: Path | None) -> Path | None:
            if value is None or value.is_absolute():
                return value
            return (base / value).resolve()

        resolved.input.path = resolve(resolved.input.path)
        resolved.electrode_array.coordinates = resolve(resolved.electrode_array.coordinates)
        resolved.simulation.params = resolve(resolved.simulation.params)
        resolved.safety.limits = resolve(resolved.safety.limits)
        resolved.output.root = resolve(resolved.output.root) or resolved.output.root
        resolved.source_path = source_path.resolve()
        return resolved

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values.pop("source_path", None)
        return values


@dataclass
class SweepConfig(StrictConfig):
    experiment: Path
    matrix: dict[str, list[Any]]
    output_root: Path | None = None
    stop_on_error: bool = True

    def validate(self, path: str = "sweep") -> None:
        if not self.matrix:
            raise ValueError(f"{path}.matrix must define at least one axis.")
        for key, values in self.matrix.items():
            if not key or not isinstance(values, list) or not values:
                raise ValueError(f"{path}.matrix.{key} must be a non-empty list.")

    def variants(self) -> list[dict[str, Any]]:
        keys = list(self.matrix)
        return [
            dict(zip(keys, values))
            for values in product(*(self.matrix[key] for key in keys))
        ]


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with open(source, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a YAML mapping.")
    return data


def load_experiment(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    config = ExperimentConfig.from_dict(load_yaml(source), path="experiment")
    return config.resolve_paths(source)


def load_sweep(path: str | Path) -> SweepConfig:
    source = Path(path).resolve()
    sweep = SweepConfig.from_dict(load_yaml(source), path="sweep")
    if not sweep.experiment.is_absolute():
        sweep.experiment = (source.parent / sweep.experiment).resolve()
    if sweep.output_root is not None and not sweep.output_root.is_absolute():
        sweep.output_root = (source.parent / sweep.output_root).resolve()
    return sweep


def set_config_value(values: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid sweep field path: {dotted_path!r}")
    target = values
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Sweep field path does not reference a mapping: {dotted_path}")
        target = child
    if parts[-1] not in target:
        raise ValueError(f"Sweep field path does not exist: {dotted_path}")
    target[parts[-1]] = value

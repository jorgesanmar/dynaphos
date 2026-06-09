from __future__ import annotations

import importlib
import inspect
from typing import Any

from dynaphos.config import StrategyConfig
from dynaphos.strategies.builtins import BUILTIN_STRATEGIES


def available_strategies() -> tuple[str, ...]:
    return tuple(sorted(name for name in BUILTIN_STRATEGIES if "-" not in name))


def _load_object(import_path: str) -> Any:
    module_name, separator, attribute = import_path.partition(":")
    if not separator:
        module_name, separator, attribute = import_path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(
            "Custom strategy import paths must use 'package.module:ClassName'."
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"Custom strategy object not found: {import_path}") from exc


def load_strategy(config: StrategyConfig):
    if config.import_path:
        factory = _load_object(config.import_path)
    else:
        name = str(config.name).strip().lower().replace(" ", "_")
        try:
            factory = BUILTIN_STRATEGIES[name]
        except KeyError as exc:
            choices = ", ".join(available_strategies())
            raise ValueError(f"Unknown strategy '{config.name}'. Available: {choices}.") from exc

    if inspect.isclass(factory):
        try:
            strategy = factory(options=dict(config.options))
        except TypeError:
            strategy = factory(**dict(config.options))
    elif callable(factory):
        strategy = factory(dict(config.options))
    else:
        strategy = factory
    if not callable(getattr(strategy, "initialize", None)) or not callable(
        getattr(strategy, "step", None)
    ):
        raise TypeError("A stimulation strategy must define initialize(context) and step(frame).")
    return strategy

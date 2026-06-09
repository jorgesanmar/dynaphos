import numpy as np
import pytest

from dynaphos.config import StrategyConfig
from dynaphos.strategies import StrategyContext, StrategyInput, load_strategy
from dynaphos.strategies.base import StimulusCommand


def context() -> StrategyContext:
    return StrategyContext(
        electrode_ids=np.arange(4),
        electrode_xy_mm=np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float),
        default_pulse_width_s=np.full(4, 170e-6),
        default_frequency_hz=np.full(4, 300.0),
        fps=20.0,
        seed=42,
    )


def frame(index: int = 1) -> StrategyInput:
    return StrategyInput(
        time_s=(index - 1) / 20.0,
        frame_index=index,
        stimulus_image=np.zeros((2, 2), dtype=float),
        baseline_amplitude_A=np.full(4, 60e-6),
        electrode_ids=np.arange(4),
        electrode_xy_mm=context().electrode_xy_mm,
        rng=np.random.default_rng(42),
    )


def test_checkerboard_strategy_activates_one_group() -> None:
    strategy = load_strategy(StrategyConfig(name="checkerboard", options={"groups": 4}))
    strategy.initialize(context())

    command = strategy.step(frame()).validated(context())

    assert np.count_nonzero(command.activation_mask) == 1
    assert np.count_nonzero(command.amplitude_A) == 1


def test_strategy_command_rejects_wrong_shape_and_negative_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        StimulusCommand(
            amplitude_A=np.ones(3),
            pulse_width_s=np.ones(4),
            frequency_hz=np.ones(4),
            activation_mask=np.ones(4, dtype=bool),
        ).validated(context())

    with pytest.raises(ValueError, match="negative"):
        StimulusCommand(
            amplitude_A=np.asarray([-1.0, 0.0, 0.0, 0.0]),
            pulse_width_s=np.ones(4),
            frequency_hz=np.ones(4),
            activation_mask=np.ones(4, dtype=bool),
        ).validated(context())


def test_pseudo_random_strategy_is_deterministic() -> None:
    first = load_strategy(
        StrategyConfig(name="pseudo_random", options={"groups": 2, "reshuffle_interval_s": 5})
    )
    second = load_strategy(
        StrategyConfig(name="pseudo_random", options={"groups": 2, "reshuffle_interval_s": 5})
    )
    first.initialize(context())
    second.initialize(context())

    assert np.array_equal(
        first.step(frame()).activation_mask,
        second.step(frame()).activation_mask,
    )

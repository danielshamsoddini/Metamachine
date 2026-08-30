from types import SimpleNamespace

import numpy as np

from metamachine.environments.components.reward import (
    RollingRMSProjectedGravityLeanPenaltyComponent,
)


def _pitch_quat_from_lean(lean: float) -> np.ndarray:
    angle = float(np.arcsin(lean))
    return np.asarray([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])


def test_rolling_rms_catches_periodic_lean_and_resets() -> None:
    component = RollingRMSProjectedGravityLeanPenaltyComponent(
        "rolling_lean",
        window_seconds=0.4,
        free_lean=0.08,
        tracking_sigma=0.12,
        require_full_window=True,
    )
    calculator = SimpleNamespace(dt=0.04, gravity_vec=np.asarray([0.0, 0.0, -1.0]))
    state = SimpleNamespace(accurate_quat=_pitch_quat_from_lean(0.0))

    values = []
    for index in range(10):
        state.accurate_quat = _pitch_quat_from_lean(0.18 if index % 2 == 0 else 0.0)
        values.append(component.calculate(state, calculator))
    assert values[-1] < 0.0

    component.reset()
    for _ in range(10):
        state.accurate_quat = _pitch_quat_from_lean(0.04)
        value = component.calculate(state, calculator)
    assert value == 0.0

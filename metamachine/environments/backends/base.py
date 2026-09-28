"""Small stateful physics boundary used by the CPU task implementation."""

from typing import Protocol

import numpy as np


class PhysicsBackend(Protocol):
    """Expose task state in the compiled morphology's coordinate order.

    ``model`` is immutable morphology metadata; ``data`` is the legacy task
    state view. Only the adapter advances dynamics. Controls are actuator
    inputs (after policy clipping/scaling/PD), not raw policy actions.
    """

    name: str
    model: object
    data: object

    def step(self, ctrl: np.ndarray, n_frames: int) -> None: ...

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None: ...

    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None: ...

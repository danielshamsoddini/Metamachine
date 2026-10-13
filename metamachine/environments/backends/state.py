"""Small device-state contract shared by task kernels and physics adapters.

Arrays are borrowed, not snapshots. No per-body Python getters or host copies.
Engine-native generalized coordinates have an explicit topology/layout; body
transforms use XYZ + XYZW and spatial velocities use (v_COM_world, omega_world).
"""

from dataclasses import dataclass
from typing import Any


class BackendCapabilityError(ValueError):
    """A task requested a feature this backend does not implement."""


@dataclass(frozen=True)
class DeviceStateView:
    joint_q: Any
    joint_qd: Any
    body_transform: Any
    body_velocity: Any
    body_com: Any  # Static local COM offsets; shared between state buffers.
    num_worlds: int
    q_stride: int
    qd_stride: int
    body_stride: int

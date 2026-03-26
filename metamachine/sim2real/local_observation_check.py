from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SIM2REAL_CHECK_FILENAME = "sim2real_check.json"
DEFAULT_SIM2REAL_TOLERANCES = {
    "projected_gravity": 0.20,
    "gyro": 1.50,
    "dof_pos": 0.15,
    "dof_vel": 2.00,
    "latent": 0.25,
}


def _broadcast_param(
    value: Any,
    *,
    num_modules: int,
    name: str,
    default: float = 0.0,
) -> np.ndarray:
    if value is None:
        return np.full(num_modules, float(default), dtype=np.float32)
    if np.isscalar(value):
        return np.full(num_modules, float(value), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).flatten()
    if arr.size != num_modules:
        raise ValueError(f"{name} must have length {num_modules}, got {arr.size}")
    return arr.astype(np.float32, copy=False)


@dataclass(frozen=True)
class ActionSequenceStep:
    action: np.ndarray
    step_index: int
    elapsed_sec: float
    segment_index: int
    segment_step_index: int
    segment_elapsed_sec: float
    segment_type: str


@dataclass(frozen=True)
class ActionSequenceSegment:
    kind: str
    duration_sec: float
    action: np.ndarray | None = None
    offset: np.ndarray | None = None
    amplitude: np.ndarray | None = None
    frequency_hz: np.ndarray | None = None
    phase: np.ndarray | None = None

    @classmethod
    def from_config(cls, data: dict[str, Any], *, num_modules: int) -> "ActionSequenceSegment":
        kind = str(data.get("type", "hold")).strip().lower()
        duration_sec = float(data.get("duration_sec", 0.0))
        if duration_sec <= 0.0:
            raise ValueError(f"sequence segment duration_sec must be > 0, got {duration_sec}")

        if kind == "hold":
            return cls(
                kind=kind,
                duration_sec=duration_sec,
                action=_broadcast_param(
                    data.get("action", None),
                    num_modules=num_modules,
                    name="hold.action",
                    default=0.0,
                ),
            )

        if kind == "sine":
            return cls(
                kind=kind,
                duration_sec=duration_sec,
                offset=_broadcast_param(
                    data.get("offset", data.get("action", None)),
                    num_modules=num_modules,
                    name="sine.offset",
                    default=0.0,
                ),
                amplitude=_broadcast_param(
                    data.get("amplitude", None),
                    num_modules=num_modules,
                    name="sine.amplitude",
                    default=0.0,
                ),
                frequency_hz=_broadcast_param(
                    data.get("frequency_hz", None),
                    num_modules=num_modules,
                    name="sine.frequency_hz",
                    default=1.0,
                ),
                phase=_broadcast_param(
                    data.get("phase", None),
                    num_modules=num_modules,
                    name="sine.phase",
                    default=0.0,
                ),
            )

        raise ValueError(f"Unsupported sim-to-real sequence segment type: {kind}")

    def num_steps(self, local_dt: float) -> int:
        return max(1, int(round(float(self.duration_sec) / float(local_dt))))

    def action_at(self, t_sec: float) -> np.ndarray:
        if self.kind == "hold":
            return np.asarray(self.action, dtype=np.float32).copy()
        if self.kind == "sine":
            assert self.offset is not None
            assert self.amplitude is not None
            assert self.frequency_hz is not None
            assert self.phase is not None
            theta = 2.0 * np.pi * self.frequency_hz * float(t_sec) + self.phase
            return (self.offset + self.amplitude * np.sin(theta)).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported segment kind: {self.kind}")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "type": self.kind,
            "duration_sec": float(self.duration_sec),
        }
        if self.kind == "hold":
            payload["action"] = np.asarray(self.action, dtype=np.float32).tolist()
        elif self.kind == "sine":
            payload["offset"] = np.asarray(self.offset, dtype=np.float32).tolist()
            payload["amplitude"] = np.asarray(self.amplitude, dtype=np.float32).tolist()
            payload["frequency_hz"] = np.asarray(self.frequency_hz, dtype=np.float32).tolist()
            payload["phase"] = np.asarray(self.phase, dtype=np.float32).tolist()
        return payload


@dataclass(frozen=True)
class ActionSequenceCase:
    name: str
    description: str
    require_stationary: bool
    sequence: tuple[ActionSequenceSegment, ...]

    @classmethod
    def from_config(cls, data: dict[str, Any], *, num_modules: int) -> "ActionSequenceCase":
        sequence_cfg = data.get("sequence", None)
        if not sequence_cfg:
            if "action" not in data:
                raise ValueError(
                    f"sim_to_real_check case '{data.get('name', 'unnamed')}' "
                    "must define either 'sequence' or legacy 'action'"
                )
            duration_sec = float(data.get("duration_sec", 0.0))
            if duration_sec <= 0.0:
                raise ValueError(
                    f"Legacy sim_to_real_check case '{data.get('name', 'unnamed')}' "
                    "must define duration_sec when using top-level action"
                )
            sequence_cfg = [
                {
                    "type": "hold",
                    "duration_sec": duration_sec,
                    "action": data.get("action", None),
                }
            ]

        sequence = tuple(
            ActionSequenceSegment.from_config(dict(segment), num_modules=num_modules)
            for segment in sequence_cfg
        )
        if not sequence:
            raise ValueError(f"sim_to_real_check case '{data.get('name', 'unnamed')}' has empty sequence")

        return cls(
            name=str(data.get("name", "case")).strip(),
            description=str(data.get("description", "")).strip(),
            require_stationary=bool(data.get("require_stationary", False)),
            sequence=sequence,
        )

    @classmethod
    def from_payload(
        cls,
        data: dict[str, Any],
        *,
        num_modules: int,
        local_dt: float,
        legacy_settle_steps: int | None = None,
    ) -> "ActionSequenceCase":
        if data.get("sequence"):
            return cls.from_config(data, num_modules=num_modules)

        if "action" in data:
            duration_sec = float(data.get("duration_sec", 0.0))
            if duration_sec <= 0.0:
                duration_sec = float(local_dt) * float(legacy_settle_steps or 0)
            return cls.from_config(
                {
                    "name": data.get("name", "case"),
                    "description": data.get("description", ""),
                    "require_stationary": data.get("require_stationary", False),
                    "sequence": [
                        {
                            "type": "hold",
                            "duration_sec": duration_sec,
                            "action": data.get("action", None),
                        }
                    ],
                },
                num_modules=num_modules,
            )

        raise ValueError(f"Unsupported sim-to-real case payload: {data}")

    def total_steps(self, local_dt: float) -> int:
        return sum(segment.num_steps(local_dt) for segment in self.sequence)

    def total_duration_sec(self, local_dt: float) -> float:
        return float(self.total_steps(local_dt)) * float(local_dt)

    def iter_steps(self, local_dt: float):
        global_step = 0
        for segment_index, segment in enumerate(self.sequence):
            step_count = segment.num_steps(local_dt)
            for segment_step_index in range(step_count):
                segment_elapsed_sec = float(segment_step_index) * float(local_dt)
                yield ActionSequenceStep(
                    action=segment.action_at(segment_elapsed_sec),
                    step_index=global_step,
                    elapsed_sec=float(global_step) * float(local_dt),
                    segment_index=segment_index,
                    segment_step_index=segment_step_index,
                    segment_elapsed_sec=segment_elapsed_sec,
                    segment_type=segment.kind,
                )
                global_step += 1

    def final_action(self, local_dt: float) -> np.ndarray:
        last = None
        for last in self.iter_steps(local_dt):
            pass
        if last is None:
            raise ValueError(f"Case '{self.name}' has no steps")
        return np.asarray(last.action, dtype=np.float32).copy()

    def to_payload(self, local_dt: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "require_stationary": self.require_stationary,
            "duration_sec": self.total_duration_sec(local_dt),
            "num_steps": self.total_steps(local_dt),
            "sequence": [segment.to_payload() for segment in self.sequence],
            "final_action": self.final_action(local_dt).tolist(),
        }


def build_default_sim2real_cases(num_modules: int) -> list[ActionSequenceCase]:
    return [
        ActionSequenceCase.from_config(
            {
                "name": "zero_action",
                "description": "Zero policy action held long enough to refresh local observation history",
                "require_stationary": True,
                "sequence": [
                    {
                        "type": "hold",
                        "duration_sec": 5.0,
                        "action": [0.0] * num_modules,
                    }
                ],
            },
            num_modules=num_modules,
        )
    ]


def normalize_suite_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    data["local_dt"] = float(data.get("local_dt", 0.01))
    data["settle_steps"] = int(data.get("settle_steps", 15))
    data["local_obs_dim"] = int(data.get("local_obs_dim", 0))
    data["num_modules"] = int(data.get("num_modules", 0))
    data["obs_tolerance"] = np.asarray(data.get("obs_tolerance", []), dtype=np.float32)
    data["obs_labels"] = list(data.get("obs_labels", []))
    return data

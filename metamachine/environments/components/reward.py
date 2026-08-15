"""
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import mujoco
from omegaconf import OmegaConf

from ...utils.curves import isaac_reward, plateau
from ...utils.math_utils import normalize_angle, quat_apply, quat_rotate_inverse


class RewardComponent(ABC):
    """Base class for reward components."""

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        self.name = name
        self.weight = weight
        self.params = kwargs

    @abstractmethod
    def calculate(self, state, calculator) -> float:
        """Calculate the reward component value."""
        pass

    def reset(self) -> None:
        """Reset component state if needed.

        Default implementation does nothing. Override in subclasses that need to reset state.
        """
        return  # Default implementation does nothing


class LinearVelocityTrackingComponent(RewardComponent):
    """Tracks linear velocity in forward direction."""

    def calculate(self, state, calculator) -> float:
        target_vel = self.params.get("target_velocity", 1.0)
        if isinstance(target_vel, str) and target_vel.startswith("cmd:"):
            target_vel = state.get_command_by_name(target_vel[4:])
            # print(f"Using command value for target velocity: {target_vel}")
        tracking_sigma = self.params.get("tracking_sigma", 0.15)

        projected_forward_vel = np.dot(
            state.accurate_vel_body, calculator.projected_forward_vec
        )
        lin_vel_error = np.sum(np.square(target_vel - projected_forward_vel))
        return np.exp(-lin_vel_error / tracking_sigma)


def _phase_fade_scale(params, calculator) -> float:
    """Return a linear phase fade without changing pre-fade reward semantics."""
    fade_start = params.get("fade_start_time")
    end_time = params.get("end_time")
    if fade_start is None or end_time is None:
        return 1.0
    fade_start = float(fade_start)
    end_time = float(end_time)
    elapsed = calculator.step_counter * calculator.dt
    if end_time <= fade_start:
        return float(elapsed < end_time)
    return float(np.clip((end_time - elapsed) / (end_time - fade_start), 0.0, 1.0))


class CommandedRollFlipCompletionComponent(RewardComponent):
    """Reward a commanded positive body-x rotation and a stable target orientation.

    The component integrates positive body-frame x angular velocity until
    ``target_turns * 2π``.  After completion it rewards a low-rate settle at
    ``landing_alignment_target``: ``+1`` is upright and ``-1`` is inverted.
    This makes a 180 degree X-axis flip distinguishable from a 360 degree
    somersault that merely returns upright.
    """

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.reset()

    def reset(self) -> None:
        self.roll_progress = 0.0
        self.completed = False
        self.success = False
        self.command_seen = False
        self._success_bonus_paid = False
        self._missed_completion_penalty_paid = False

    @staticmethod
    def _upright_alignment(state, calculator) -> float:
        projected_gravity = quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)
        return float(np.clip(np.dot(calculator.projected_upward_vec, -projected_gravity), -1.0, 1.0))

    def calculate(self, state, calculator) -> float:
        command_name = str(self.params.get("command_name", "cmd_flip_x"))
        command = float(state.get_command_by_name(command_name))
        active_threshold = float(self.params.get("active_threshold", 0.5))
        angular_velocity = np.asarray(state.accurate_ang_vel_body, dtype=np.float64)
        x_rate = float(angular_velocity[0])
        off_axis_rate = float(np.linalg.norm(angular_velocity[1:]))
        upright_raw = self._upright_alignment(state, calculator)
        upright = max(upright_raw, 0.0)
        landing_target = float(np.clip(self.params.get("landing_alignment_target", 1.0), -1.0, 1.0))
        landing_alignment = max(landing_target * upright_raw, 0.0)
        total_rate = float(np.linalg.norm(angular_velocity))

        stationary_sigma = max(float(self.params.get("stationary_sigma", 1.5)), 1e-6)
        stationary = float(np.exp(-np.square(total_rate) / stationary_sigma))
        idle_scale = float(self.params.get("idle_upright_scale", 0.5))
        settle_scale = float(self.params.get("settle_upright_scale", 0.5))

        if command >= active_threshold:
            self.command_seen = True

        target_rate = max(float(self.params.get("target_roll_rate", 5.0)), 1e-6)
        if self.completed:
            # A discrete flip must settle once its single turn is complete.  The
            # scheduled command pulse is normally off by this point; while it
            # remains on, actively charge continued roll to prevent multi-turn
            # spinning from replacing the landing phase.
            if command >= active_threshold:
                rate_cost = min(abs(x_rate) / target_rate, 1.0)
                return -float(self.params.get("post_completion_rate_penalty", 1.0)) * rate_cost
            settled = settle_scale * landing_alignment * stationary
            if settled >= float(self.params.get("success_settle_threshold", 0.75)):
                self.success = True
                if not self._success_bonus_paid:
                    settled += float(self.params.get("success_bonus", 20.0))
                    self._success_bonus_paid = True
            return settled

        if command < active_threshold and not self.command_seen:
            return idle_scale * upright * stationary

        # A discrete command releases the initiation request; it must not erase
        # the opportunity to finish the turn using the generated momentum.
        deadline = float(self.params.get("completion_deadline", np.inf))
        elapsed = calculator.step_counter * calculator.dt
        if elapsed > deadline:
            if not self._missed_completion_penalty_paid:
                self._missed_completion_penalty_paid = True
                return -float(self.params.get("missed_completion_penalty", 0.5))
            return 0.0

        off_axis_sigma = max(float(self.params.get("off_axis_sigma", 4.0)), 1e-6)
        roll_rate_score = float(np.clip(max(x_rate, 0.0) / target_rate, 0.0, 1.0))
        axis_purity = float(np.exp(-np.square(off_axis_rate) / off_axis_sigma))
        progress_scale = float(self.params.get("progress_rate_scale", 0.5))
        if command < active_threshold:
            progress_scale *= float(self.params.get("coast_progress_scale", 0.5))
        reward = progress_scale * roll_rate_score * axis_purity

        target_angle = 2.0 * np.pi * max(float(self.params.get("target_turns", 1.0)), 1e-6)
        self.roll_progress = min(target_angle, self.roll_progress + max(x_rate, 0.0) * calculator.dt)
        if self.roll_progress >= target_angle:
            self.completed = True
            if elapsed <= deadline:
                reward += float(self.params.get("completion_bonus", 8.0))

        return reward


class CommandedAxisAngularVelocityComponent(RewardComponent):
    """Track a command-scaled body-frame angular velocity about one axis.

    A zero-valued command rewards stopping the chosen rotation. A nonzero
    command requests ``command_scale * command`` radians/s about ``axis`` and
    independently suppresses rotation about the other two body axes.
    """

    def calculate(self, state, calculator) -> float:
        command_name = str(self.params.get("command_name", "cmd_flip_x"))
        command = float(state.get_command_by_name(command_name))
        zero_command_start_time = self.params.get("zero_command_start_time")
        if (
            zero_command_start_time is not None
            and command <= 0.0
            and calculator.step_counter * calculator.dt < float(zero_command_start_time)
        ):
            return 0.0
        command_scale = float(self.params.get("command_scale", 1.0))
        target_rate = command_scale * command

        axis = np.asarray(self.params.get("axis", [1.0, 0.0, 0.0]), dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-8:
            raise ValueError("commanded_axis_angular_velocity requires a nonzero axis")
        axis /= axis_norm

        angular_velocity = np.asarray(state.accurate_ang_vel_body, dtype=np.float64)
        axis_rate = float(np.dot(angular_velocity, axis))
        off_axis_rate = angular_velocity - axis_rate * axis

        tracking_sigma = max(float(self.params.get("tracking_sigma", 1.0)), 1e-6)
        off_axis_sigma = max(float(self.params.get("off_axis_sigma", 1.0)), 1e-6)
        rate_reward = np.exp(-np.square(target_rate - axis_rate) / tracking_sigma)
        axis_reward = np.exp(-np.dot(off_axis_rate, off_axis_rate) / off_axis_sigma)
        return float(rate_reward * axis_reward)


class AngularVelocityTrackingComponent(RewardComponent):
    """Tracks angular velocity around gravity axis."""

    def calculate(self, state, calculator) -> float:
        end_time = self.params.get("end_time")
        if (
            end_time is not None
            and calculator.step_counter * calculator.dt >= float(end_time)
        ):
            return 0.0

        target_ang_vel = self.params.get("target_angular_velocity", 0.0)
        if isinstance(target_ang_vel, str) and target_ang_vel.startswith("cmd:"):
            target_ang_vel = state.get_command_by_name(target_ang_vel[4:])
            # print(f"Using command value for target angular velocity: {target_ang_vel}")

        tracking_sigma = self.params.get("tracking_sigma", 0.15)

        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        # Match state.derived.yaw_rate: rotation about the gravity/up axis.
        projected_z_ang = np.dot(
            -accurate_projected_gravity, state.accurate_ang_vel_body
        )
        ang_vel_error = np.sum(np.square(target_ang_vel - projected_z_ang))
        return _phase_fade_scale(self.params, calculator) * np.exp(
            -ang_vel_error / tracking_sigma
        )


# class LinearVelocityTrackingCMDComponent(RewardComponent):
#     """Tracks linear velocity in forward direction."""

#     def calculate(self, state, calculator) -> float:
#         target_vel = state.get_command_by_name('forward_speed')
#         tracking_sigma = self.params.get('tracking_sigma', 0.15)

#         projected_forward_vel = np.dot(state.accurate_vel_body,
#                                      calculator.projected_forward_vec)
#         lin_vel_error = np.sum(np.square(target_vel - projected_forward_vel))
#         return np.exp(-lin_vel_error / tracking_sigma)


# class AngularVelocityTrackingCMDComponent(RewardComponent):
#     """Tracks angular velocity around gravity axis."""

#     def calculate(self, state, calculator) -> float:
#         target_ang_vel = state.get_command_by_name('turn_rate')
#         tracking_sigma = self.params.get('tracking_sigma', 0.15)

#         accurate_projected_gravity = quat_rotate_inverse(state.accurate_quat,
#                                                        calculator.gravity_vec)
#         projected_z_ang = np.dot(state.accurate_ang_vel_body,
#                                accurate_projected_gravity)
#         ang_vel_error = np.sum(np.square(target_ang_vel - projected_z_ang))
#         return np.exp(-ang_vel_error / tracking_sigma)


class ContactFlightTimeComponent(RewardComponent):
    """Rewards flight time between contacts."""

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.contact_counter = {}

    def _active_floor_contacts(self, state) -> list[int]:
        contact_source = self.params.get("contact_source", "socks")
        if contact_source == "geoms":
            contacts = list(getattr(state, "contact_floor_geoms", []))
        else:
            contacts = list(getattr(state, "contact_floor_socks", []))

        geom_filters = self.params.get("geom_name_contains")
        if geom_filters and state.mj_model is not None:
            contacts = [
                geom
                for geom in contacts
                if any(token in state.mj_model.geom(geom).name for token in geom_filters)
            ]
        return contacts

    def calculate(self, state, calculator) -> float:
        allowed_contacts = self.params.get("allowed_num_contacts", 1)
        active_contacts = self._active_floor_contacts(state)

        # Update contact counters
        for key in self.contact_counter:
            self.contact_counter[key] += 1
        for contact_geom in active_contacts:
            self.contact_counter[contact_geom] = 0

        if len(active_contacts) >= allowed_contacts + 1:
            self.contact_counter = dict.fromkeys(self.contact_counter, 0)

        feet_air_time = np.array(list(self.contact_counter.values())) * calculator.dt
        return np.sum(feet_air_time)

    def reset(self) -> None:
        self.contact_counter = {}


class PersistentFootAirTimePenaltyComponent(RewardComponent):
    """Penalize a foot only when its airborne interval becomes abnormally long.

    Ordinary swing phases are free for ``grace_time`` seconds.  Beyond that,
    each foot's cost ramps smoothly to one, independently, so a policy cannot
    obtain a cheap tripod gait by parking one particular leg in the air.
    """

    DEFAULT_FOOT_GEOM_NAMES = (
        "front_right_ankle_geom",
        "front_left_ankle_geom",
        "rear_left_ankle_geom",
        "rear_right_ankle_geom",
    )

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.foot_geom_ids: dict[str, int] = {}
        self.air_times: dict[str, float] = {}
        self._model_identity: int | None = None

    def _resolve_foot_geoms(self, model) -> None:
        requested_names = tuple(
            self.params.get("foot_geom_names", self.DEFAULT_FOOT_GEOM_NAMES)
        )
        available = {
            model.geom(geom_id).name: geom_id
            for geom_id in range(int(model.ngeom))
            if model.geom(geom_id).name
        }
        missing = [name for name in requested_names if name not in available]
        if missing:
            raise ValueError(
                "persistent_foot_air_time_penalty could not find foot geoms: "
                + ", ".join(missing)
            )
        self.foot_geom_ids = {name: available[name] for name in requested_names}
        self.air_times = dict.fromkeys(requested_names, 0.0)
        self._model_identity = id(model)

    def calculate(self, state, calculator) -> float:
        model = getattr(state, "mj_model", None)
        if model is None:
            return 0.0
        if not self.foot_geom_ids or self._model_identity != id(model):
            self._resolve_foot_geoms(model)

        active_contacts = set(getattr(state, "contact_floor_geoms", []))
        grace_time = max(float(self.params.get("grace_time", 0.35)), 0.0)
        ramp_time = max(float(self.params.get("ramp_time", 0.65)), 1e-6)
        power = max(float(self.params.get("power", 2.0)), 1.0)
        total_cost = 0.0

        for foot_name, geom_id in self.foot_geom_ids.items():
            if geom_id in active_contacts:
                self.air_times[foot_name] = 0.0
            else:
                self.air_times[foot_name] += calculator.dt
            normalized_excess = np.clip(
                (self.air_times[foot_name] - grace_time) / ramp_time,
                0.0,
                1.0,
            )
            total_cost += float(normalized_excess**power)

        return -total_cost

    def reset(self) -> None:
        self.air_times = dict.fromkeys(self.foot_geom_ids, 0.0)


class FootSlipPenaltyComponent(RewardComponent):
    """Penalize horizontal slip of ankle geoms while they contact the floor."""

    DEFAULT_FOOT_GEOM_NAMES = PersistentFootAirTimePenaltyComponent.DEFAULT_FOOT_GEOM_NAMES

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.foot_geom_ids: dict[str, int] = {}
        self._model_identity: int | None = None

    def _resolve_foot_geoms(self, model) -> None:
        requested_names = tuple(
            self.params.get("foot_geom_names", self.DEFAULT_FOOT_GEOM_NAMES)
        )
        available = {
            model.geom(geom_id).name: geom_id
            for geom_id in range(int(model.ngeom))
            if model.geom(geom_id).name
        }
        missing = [name for name in requested_names if name not in available]
        if missing:
            raise ValueError(
                "foot_slip_penalty could not find foot geoms: "
                + ", ".join(missing)
            )
        self.foot_geom_ids = {name: available[name] for name in requested_names}
        self._model_identity = id(model)

    def calculate(self, state, calculator) -> float:
        model = getattr(state, "mj_model", None)
        data = getattr(state, "mj_data", None)
        if model is None or data is None:
            return 0.0
        if not self.foot_geom_ids or self._model_identity != id(model):
            self._resolve_foot_geoms(model)

        active_contacts = set(getattr(state, "contact_floor_geoms", []))
        free_speed = max(float(self.params.get("free_speed", 0.08)), 0.0)
        speed_scale = max(float(self.params.get("speed_scale", 0.35)), 1e-6)
        power = max(float(self.params.get("power", 2.0)), 1.0)
        total_cost = 0.0
        spatial_velocity = np.empty(6, dtype=np.float64)

        for geom_id in self.foot_geom_ids.values():
            if geom_id not in active_contacts:
                continue
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, spatial_velocity, 0
            )
            planar_speed = float(np.linalg.norm(spatial_velocity[3:5]))
            normalized_excess = np.clip(
                (planar_speed - free_speed) / speed_scale, 0.0, 1.0
            )
            total_cost += float(normalized_excess**power)
        return -total_cost


class ExcessiveFootHeightPenaltyComponent(RewardComponent):
    """Penalize ankle geoms lifted far above ordinary swing clearance.

    ``reference_mode=median_other_feet`` makes the measurement invariant to
    whole-body crouching by comparing each ankle with the other three ankles.
    ``reference_mode=torso`` compares against torso-center height.  The default
    ``floor`` mode preserves existing configuration behavior.
    """

    DEFAULT_FOOT_GEOM_NAMES = PersistentFootAirTimePenaltyComponent.DEFAULT_FOOT_GEOM_NAMES

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.foot_geom_ids: dict[str, int] = {}
        self._model_identity: int | None = None

    def _resolve_foot_geoms(self, model) -> None:
        requested_names = tuple(
            self.params.get("foot_geom_names", self.DEFAULT_FOOT_GEOM_NAMES)
        )
        available = {
            model.geom(geom_id).name: geom_id
            for geom_id in range(int(model.ngeom))
            if model.geom(geom_id).name
        }
        missing = [name for name in requested_names if name not in available]
        if missing:
            raise ValueError(
                "excessive_foot_height_penalty could not find foot geoms: "
                + ", ".join(missing)
            )
        self.foot_geom_ids = {name: available[name] for name in requested_names}
        self._model_identity = id(model)

    def calculate(self, state, calculator) -> float:
        model = getattr(state, "mj_model", None)
        data = getattr(state, "mj_data", None)
        if model is None or data is None:
            return 0.0
        if not self.foot_geom_ids or self._model_identity != id(model):
            self._resolve_foot_geoms(model)

        start_time = max(float(self.params.get("start_time", 0.5)), 0.0)
        if calculator.step_counter * calculator.dt < start_time:
            return 0.0

        reference_mode = str(self.params.get("reference_mode", "floor"))
        free_height = float(
            self.params.get(
                "free_relative_height",
                self.params.get("free_height", 0.22),
            )
        )
        height_scale = max(float(self.params.get("height_scale", 0.18)), 1e-6)
        floor_height = float(self.params.get("floor_height", 0.0))
        power = max(float(self.params.get("power", 2.0)), 1.0)
        total_cost = 0.0
        foot_heights = {
            foot_name: float(data.geom_xpos[geom_id][2])
            for foot_name, geom_id in self.foot_geom_ids.items()
        }
        for foot_name, height_world in foot_heights.items():
            if reference_mode == "median_other_feet":
                other_heights = [
                    other_height
                    for other_name, other_height in foot_heights.items()
                    if other_name != foot_name
                ]
                height = height_world - float(np.median(other_heights))
            elif reference_mode == "torso":
                height = height_world - float(state.accurate_pos_world[2])
            elif reference_mode == "floor":
                height = height_world - floor_height
            else:
                raise ValueError(
                    "excessive_foot_height_penalty reference_mode must be "
                    "floor, torso, or median_other_feet"
                )
            normalized_excess = np.clip(
                (height - free_height) / height_scale,
                0.0,
                1.0,
            )
            total_cost += float(normalized_excess**power)
        return -total_cost


class DOFVelocityPenaltyComponent(RewardComponent):
    """Penalizes excessive DOF velocities."""

    def calculate(self, state, calculator) -> float:
        velocity_limit = self.params.get("velocity_limit", 10.0)
        return -np.sum((np.abs(state.dof_vel) - velocity_limit).clip(0, 1e5))


class TimedDOFVelocityPenaltyComponent(RewardComponent):
    """Applies a bounded joint-motion cost after a phase transition."""

    def calculate(self, state, calculator) -> float:
        if self.params.get("require_phase_clearance_latched", False) and not bool(
            getattr(state, "phase_clearance_latched", False)
        ):
            return 0.0
        start_time = float(self.params.get("start_time", 0.0))
        elapsed = calculator.step_counter * calculator.dt - start_time
        if elapsed < 0.0:
            return 0.0

        ramp_time = max(float(self.params.get("ramp_time", 0.0)), 0.0)
        ramp = (
            1.0
            if ramp_time == 0.0
            else float(np.clip(elapsed / ramp_time, 0.0, 1.0))
        )
        free_velocity = max(float(self.params.get("free_velocity", 1.0)), 0.0)
        velocity_scale = max(float(self.params.get("velocity_scale", 4.0)), 1e-6)
        max_penalty = max(float(self.params.get("max_penalty", 4.0)), 0.0)
        excess = np.maximum(
            np.abs(np.asarray(state.dof_vel)) - free_velocity,
            0.0,
        )
        penalty = float(np.mean(np.square(excess / velocity_scale)))
        return -ramp * min(penalty, max_penalty)


class DOFAccelerationPenaltyComponent(RewardComponent):
    """Penalizes DOF accelerations."""

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.last_dof_vel = None

    def calculate(self, state, calculator) -> float:
        if self.last_dof_vel is not None:
            dof_acc_penalty = np.sum(
                np.square((self.last_dof_vel - state.dof_vel) / calculator.dt)
            )
        else:
            dof_acc_penalty = 0
        self.last_dof_vel = state.dof_vel.copy()
        return -dof_acc_penalty

    def reset(self) -> None:
        self.last_dof_vel = None


class ContactPenaltyComponent(RewardComponent):
    """Penalizes unwanted contacts."""

    def _active_floor_contacts(self, state) -> list[int]:
        contact_source = self.params.get("contact_source", "balls")
        if contact_source == "geoms":
            contacts = list(getattr(state, "contact_floor_geoms", []))
        elif contact_source == "joint_floor":
            return []  # handled in calculate()
        else:
            contacts = list(getattr(state, "contact_floor_balls", []))

        geom_filters = self.params.get("geom_name_contains")
        if geom_filters and state.mj_model is not None:
            contacts = [
                geom
                for geom in contacts
                if any(token in state.mj_model.geom(geom).name for token in geom_filters)
            ]
        return contacts

    def calculate(self, state, calculator) -> float:
        if self.params.get("contact_source") == "joint_floor":
            return -float(getattr(state, "num_jointfloor_contact", 0))
        return -len(self._active_floor_contacts(state))


class TimedContactPenaltyComponent(ContactPenaltyComponent):
    """Penalize selected floor contacts only after an initial grace period."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        if calculator.step_counter * calculator.dt < start_time:
            return 0.0
        return super().calculate(state, calculator)


class ExponentialTimedContactPenaltyComponent(ContactPenaltyComponent):
    """Penalize contacts after a grace period with an exponential time ramp."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        elapsed = calculator.step_counter * calculator.dt - start_time
        if elapsed < 0.0:
            return 0.0

        exponent_rate = float(self.params.get("exponent_rate", 0.35))
        max_multiplier = max(float(self.params.get("max_multiplier", 32.0)), 1.0)
        multiplier = min(float(np.exp(exponent_rate * elapsed)), max_multiplier)
        return multiplier * super().calculate(state, calculator)


class RampedContactPenaltyComponent(ContactPenaltyComponent):
    """Bounded contact cost with a smooth phase-in period."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        elapsed = calculator.step_counter * calculator.dt - start_time
        if elapsed < 0.0:
            return 0.0

        ramp_time = max(float(self.params.get("ramp_time", 0.0)), 0.0)
        ramp = 1.0 if ramp_time == 0.0 else float(
            np.clip(elapsed / ramp_time, 0.0, 1.0)
        )
        max_contacts = max(float(self.params.get("max_contacts", 4.0)), 0.0)
        contact_cost = max(
            float(super().calculate(state, calculator)), -max_contacts
        )
        return ramp * contact_cost


class TimedAirborneSpinComponent(RewardComponent):
    """Reward commanded yaw only while airborne, after a launch period."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        if calculator.step_counter * calculator.dt < start_time:
            return 0.0

        contacts = list(getattr(state, "contact_floor_geoms", []))
        filters = self.params.get("geom_name_contains")
        if filters and state.mj_model is not None:
            contacts = [
                geom
                for geom in contacts
                if any(token in state.mj_model.geom(geom).name for token in filters)
            ]
        if contacts:
            return 0.0

        target = self.params.get("target_angular_velocity", 0.0)
        if isinstance(target, str) and target.startswith("cmd:"):
            target = state.get_command_by_name(target[4:])
        sigma = float(self.params.get("tracking_sigma", 0.01))
        projected_gravity = quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)
        yaw_rate = np.dot(-projected_gravity, state.accurate_ang_vel_body)
        return float(np.exp(-np.square(float(target) - yaw_rate) / sigma))


class TimedContactFreeSpinComponent(RewardComponent):
    """Dense signed yaw reward for a post-release, contact-free phase.

    Unlike ``timed_airborne_spin``, a stopped robot earns an explicit negative
    value instead of merely losing a small positive tracking reward. This
    prevents a quiet survival pose from replacing the desired passive coast.
    Omit ``geom_name_contains`` to require that *no* robot geom touches the
    floor; provide a filter only for tasks that intentionally permit a body
    contact.
    """

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        if calculator.step_counter * calculator.dt < start_time:
            return 0.0

        contacts = list(getattr(state, "contact_floor_geoms", []))
        filters = self.params.get("geom_name_contains")
        if filters and state.mj_model is not None:
            contacts = [
                geom
                for geom in contacts
                if any(token in state.mj_model.geom(geom).name for token in filters)
            ]
        if contacts:
            return float(self.params.get("contact_value", -1.0))

        target = self.params.get("target_angular_velocity", 0.0)
        if isinstance(target, str) and target.startswith("cmd:"):
            target = state.get_command_by_name(target[4:])
        target = float(target)
        direction = 1.0 if target >= 0.0 else -1.0
        target_speed = max(abs(target), 1e-6)
        min_speed = float(
            self.params.get("min_angular_velocity", 0.5 * target_speed)
        )
        min_speed = min(max(min_speed, 0.0), target_speed - 1e-6)

        projected_gravity = quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)
        yaw_rate = float(np.dot(-projected_gravity, state.accurate_ang_vel_body))
        directed_rate = direction * yaw_rate

        # The score is negative when stalled/reversed, rises continuously in
        # the useful coast range, and saturates at the target spin rate.
        stalled_cap = abs(float(self.params.get("stalled_penalty", 1.0)))
        score = (directed_rate - min_speed) / max(target_speed - min_speed, 1e-6)
        return float(np.clip(score, -stalled_cap, 1.0))


class TimedSpinRetentionComponent(RewardComponent):
    """Signed post-launch spin reward independent of limb contacts."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        elapsed = calculator.step_counter * calculator.dt - start_time
        if elapsed < 0.0:
            return 0.0

        target = self.params.get("target_angular_velocity", 0.0)
        if isinstance(target, str) and target.startswith("cmd:"):
            target = state.get_command_by_name(target[4:])
        target = float(target)
        direction = 1.0 if target >= 0.0 else -1.0
        target_speed = max(abs(target), 1e-6)
        min_speed = float(
            self.params.get("min_angular_velocity", 0.5 * target_speed)
        )
        min_speed = min(max(min_speed, 0.0), target_speed - 1e-6)

        projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        yaw_rate = float(
            np.dot(-projected_gravity, state.accurate_ang_vel_body)
        )
        directed_rate = direction * yaw_rate
        stalled_cap = abs(float(self.params.get("stalled_penalty", 1.0)))
        score = (directed_rate - min_speed) / max(
            target_speed - min_speed, 1e-6
        )

        ramp_time = max(float(self.params.get("ramp_time", 0.0)), 0.0)
        ramp = 1.0 if ramp_time == 0.0 else float(
            np.clip(elapsed / ramp_time, 0.0, 1.0)
        )
        return ramp * float(np.clip(score, -stalled_cap, 1.0))


class GatedPassiveSpinComponent(RewardComponent):
    """Reward post-launch torso spin only while legs are clear and nearly still."""

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.clear_steps = 0

    def _active_leg_contacts(self, state) -> list[int]:
        contacts = list(getattr(state, "contact_floor_geoms", []) or [])
        filters = self.params.get(
            "geom_name_contains",
            ["hip_geom", "upper_geom", "ankle_geom"],
        )
        if filters and state.mj_model is not None:
            contacts = [
                geom
                for geom in contacts
                if any(
                    token in (state.mj_model.geom(int(geom)).name or "")
                    for token in filters
                )
            ]
        return contacts

    def _has_required_support(self, state) -> bool:
        filters = self.params.get("required_support_geom_name_contains")
        if not filters:
            return True
        if state.mj_model is None:
            return False
        return any(
            any(
                token in (state.mj_model.geom(int(geom)).name or "")
                for token in filters
            )
            for geom in (getattr(state, "contact_floor_geoms", []) or [])
        )

    def _whole_body_momentum_and_inertia(self, state, calculator) -> tuple[float, float]:
        """Return signed axial angular momentum and effective system inertia."""
        if state.mj_model is None or state.mj_data is None:
            raise ValueError(
                "gated_passive_spin whole_body_effective_yaw requires MuJoCo state"
            )

        import mujoco

        model = state.mj_model
        data = state.mj_data
        masses = np.asarray(model.body_mass, dtype=np.float64)
        body_ids = np.flatnonzero(masses > 0.0)
        total_mass = float(np.sum(masses[body_ids]))
        if total_mass <= 0.0:
            return 0.0, 0.0

        com_positions = np.asarray(data.xipos, dtype=np.float64)
        system_com = np.sum(
            masses[body_ids, None] * com_positions[body_ids],
            axis=0,
        ) / total_mass

        up_axis = -np.asarray(calculator.gravity_vec, dtype=np.float64)
        up_norm = float(np.linalg.norm(up_axis))
        if up_norm <= 1e-9:
            up_axis = np.array([0.0, 0.0, 1.0])
        else:
            up_axis /= up_norm

        angular_momentum = np.zeros(3, dtype=np.float64)
        effective_inertia = 0.0
        spatial_velocity = np.zeros(6, dtype=np.float64)
        for body_id in body_ids:
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                spatial_velocity,
                0,
            )
            omega_world = spatial_velocity[:3].copy()
            com_velocity = spatial_velocity[3:].copy()

            inertia_rotation = np.asarray(
                data.ximat[body_id],
                dtype=np.float64,
            ).reshape(3, 3)
            inertia_world = (
                inertia_rotation
                @ np.diag(np.asarray(model.body_inertia[body_id], dtype=np.float64))
                @ inertia_rotation.T
            )
            radius = com_positions[body_id] - system_com
            angular_momentum += inertia_world @ omega_world
            angular_momentum += np.cross(
                radius,
                masses[body_id] * com_velocity,
            )
            effective_inertia += float(up_axis @ inertia_world @ up_axis)
            effective_inertia += float(
                masses[body_id]
                * (
                    np.dot(radius, radius)
                    - np.square(np.dot(up_axis, radius))
                )
            )

        if effective_inertia <= 1e-9:
            return 0.0, 0.0
        axial_momentum = float(np.dot(up_axis, angular_momentum))
        return axial_momentum, effective_inertia

    def _whole_body_effective_spin_rate(self, state, calculator) -> float:
        momentum, inertia = self._whole_body_momentum_and_inertia(state, calculator)
        return 0.0 if inertia <= 1e-9 else momentum / inertia

    def _spin_rate(self, state, calculator) -> float:
        source = self.params.get("spin_measure", "torso_yaw")
        if source == "whole_body_effective_yaw":
            return self._whole_body_effective_spin_rate(state, calculator)
        if source != "torso_yaw":
            raise ValueError(
                "gated_passive_spin spin_measure must be torso_yaw or "
                "whole_body_effective_yaw"
            )
        projected_gravity = quat_rotate_inverse(
            state.accurate_quat,
            calculator.gravity_vec,
        )
        return float(
            np.dot(-projected_gravity, state.accurate_ang_vel_body)
        )

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        if calculator.step_counter * calculator.dt < start_time:
            self.clear_steps = 0
            return 0.0

        if self._active_leg_contacts(state) or not self._has_required_support(state):
            self.clear_steps = 0
            return 0.0
        self.clear_steps += 1

        clearance_time = max(
            float(self.params.get("continuous_clearance_time", 0.25)),
            0.0,
        )
        clearance_gate = (
            1.0
            if clearance_time == 0.0
            else float(
                np.clip(
                    self.clear_steps * calculator.dt / clearance_time,
                    0.0,
                    1.0,
                )
            )
        )

        free_velocity = max(
            float(self.params.get("free_joint_velocity", 0.25)),
            0.0,
        )
        velocity_scale = max(
            float(self.params.get("joint_velocity_scale", 0.75)),
            1e-6,
        )
        dof_velocity = np.abs(np.asarray(state.dof_vel, dtype=np.float64))
        velocity_excess = np.maximum(dof_velocity - free_velocity, 0.0)
        motion_gate = float(
            np.exp(-np.mean(np.square(velocity_excess / velocity_scale)))
        )

        target = self.params.get("target_angular_velocity", 0.0)
        if isinstance(target, str) and target.startswith("cmd:"):
            target = state.get_command_by_name(target[4:])
        target = float(target)
        direction = 1.0 if target >= 0.0 else -1.0
        target_speed = max(abs(target), 1e-6)
        min_speed = float(
            self.params.get("min_angular_velocity", 0.5 * target_speed)
        )
        min_speed = min(max(min_speed, 0.0), target_speed - 1e-6)

        yaw_rate = self._spin_rate(state, calculator)
        directed_rate = direction * yaw_rate
        spin_score = float(
            np.clip(
                (directed_rate - min_speed)
                / max(target_speed - min_speed, 1e-6),
                0.0,
                1.0,
            )
        )
        return clearance_gate * motion_gate * spin_score

    def reset(self) -> None:
        self.clear_steps = 0


class LatchedMomentumRetentionComponent(GatedPassiveSpinComponent):
    """Retain real whole-body angular momentum after the achieved pose is latched."""

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.release_momentum = None
        self.release_quality = 0.0

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        if calculator.step_counter * calculator.dt < start_time:
            self.reset()
            return 0.0
        if not bool(getattr(state, "phase_clearance_latched", False)):
            self.clear_steps = 0
            return 0.0

        momentum, inertia = self._whole_body_momentum_and_inertia(state, calculator)
        # A passive coast should retain the momentum it actually earned, not
        # repeatedly chase a prescribed angular speed. New configs therefore
        # provide an explicit direction and an optional one-sided saturation
        # rate for release quality. Keep the old target form as a fallback so
        # existing experiments remain reproducible.
        target = self.params.get("target_angular_velocity")
        direction_param = self.params.get("spin_direction")
        if direction_param is None:
            if isinstance(target, str) and target.startswith("cmd:"):
                target = state.get_command_by_name(target[4:])
            target = float(0.0 if target is None else target)
            direction = 1.0 if target >= 0.0 else -1.0
        else:
            direction = 1.0 if float(direction_param) >= 0.0 else -1.0
        directed_momentum = direction * momentum

        if self.release_momentum is None:
            self.release_momentum = max(directed_momentum, 1e-9)
            release_rate = directed_momentum / max(inertia, 1e-9)
            min_release_rate = max(
                float(self.params.get("min_release_angular_velocity", 0.2)),
                0.0,
            )
            required_release_rate = max(
                float(
                    self.params.get(
                        "required_release_angular_velocity",
                        min_release_rate,
                    )
                ),
                min_release_rate,
            )
            # A weak release must never turn into a valid coast merely because
            # it retains a large fraction of a tiny initial momentum.
            if release_rate < required_release_rate:
                self.release_quality = 0.0
            else:
                saturation_rate = max(
                    float(
                        self.params.get(
                            "quality_saturation_rate",
                            abs(float(target)) if target is not None else required_release_rate,
                        )
                    ),
                    required_release_rate,
                )
                self.release_quality = float(
                    np.clip(release_rate / saturation_rate, 0.0, 1.0)
                )

        if self._active_leg_contacts(state) or not self._has_required_support(state):
            self.clear_steps = 0
            return 0.0
        self.clear_steps += 1

        clearance_time = max(
            float(self.params.get("continuous_clearance_time", 0.25)),
            0.0,
        )
        clearance_gate = (
            1.0
            if clearance_time == 0.0
            else float(
                np.clip(
                    self.clear_steps * calculator.dt / clearance_time,
                    0.0,
                    1.0,
                )
            )
        )
        free_velocity = max(
            float(self.params.get("free_joint_velocity", 0.25)),
            0.0,
        )
        velocity_scale = max(
            float(self.params.get("joint_velocity_scale", 0.75)),
            1e-6,
        )
        velocity_excess = np.maximum(
            np.abs(np.asarray(state.dof_vel, dtype=np.float64)) - free_velocity,
            0.0,
        )
        motion_gate = float(
            np.exp(-np.mean(np.square(velocity_excess / velocity_scale)))
        )
        max_retention = max(float(self.params.get("max_retention", 1.1)), 0.0)
        retention = float(
            np.clip(
                directed_momentum / max(self.release_momentum, 1e-9),
                0.0,
                max_retention,
            )
        )
        return self.release_quality * retention * clearance_gate * motion_gate

    def reset(self) -> None:
        super().reset()
        self.release_momentum = None
        self.release_quality = 0.0


class WholeBodyReleaseSpinComponent(GatedPassiveSpinComponent):
    """Signed absolute whole-body spin objective around the release window."""

    def calculate(self, state, calculator) -> float:
        time_s = calculator.step_counter * calculator.dt
        start_time = float(self.params.get("start_time", 0.0))
        end_time = float(self.params.get("end_time", np.inf))
        if time_s < start_time or time_s >= end_time:
            return 0.0

        target = self.params.get("target_angular_velocity")
        direction_param = self.params.get("spin_direction")
        if direction_param is None:
            if isinstance(target, str) and target.startswith("cmd:"):
                target = state.get_command_by_name(target[4:])
            target = float(0.0 if target is None else target)
            direction = 1.0 if target >= 0.0 else -1.0
            saturation_rate = max(abs(target), 1e-6)
        else:
            direction = 1.0 if float(direction_param) >= 0.0 else -1.0
            saturation_rate = max(
                float(self.params.get("saturation_angular_velocity", 1.0)),
                1e-6,
            )
        directed_rate = (
            direction * self._whole_body_effective_spin_rate(state, calculator)
        )
        stalled_cap = abs(float(self.params.get("stalled_penalty", 1.0)))
        # Keep this dense all the way down to zero. The hard minimum belongs
        # to the latch/retention gate, not to the launch-learning signal.
        score = directed_rate / saturation_rate
        return float(np.clip(score, -stalled_cap, 1.0))


class ClearanceLatchPreparationComponent(GatedPassiveSpinComponent):
    """Dense release preparation reward until the safe pose latch succeeds."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        end_time = float(self.params.get("end_time", np.inf))
        time_s = calculator.step_counter * calculator.dt
        if time_s < start_time or time_s >= end_time:
            return 0.0
        if bool(getattr(state, "phase_clearance_latched", False)):
            return 0.0

        max_contacts = max(float(self.params.get("max_contacts", 4.0)), 1.0)
        contact_score = 1.0 - min(len(self._active_leg_contacts(state)) / max_contacts, 1.0)
        support_score = 1.0 if self._has_required_support(state) else 0.0

        limits = np.asarray(
            self.params.get("joint_limits", np.ones_like(state.dof_pos)),
            dtype=np.float64,
        )
        positions = np.asarray(state.dof_pos, dtype=np.float64)
        margin = max(float(self.params.get("joint_limit_margin", 0.15)), 1e-6)
        margin_score = float(
            np.mean(np.clip((limits - np.abs(positions)) / margin, 0.0, 1.0))
        )

        clearance_time = max(
            float(self.params.get("continuous_clearance_time", 0.25)),
            calculator.dt,
        )
        progress = float(
            np.clip(
                float(getattr(state, "phase_clearance_steps", 0))
                * calculator.dt
                / clearance_time,
                0.0,
                1.0,
            )
        )
        return float(
            0.45 * contact_score
            + 0.20 * support_score
            + 0.20 * margin_score
            + 0.15 * progress
        )


class PhaseLatchSuccessComponent(RewardComponent):
    """Post-boundary success signal; failed release remains explicitly negative."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        if calculator.step_counter * calculator.dt < start_time:
            return 0.0
        return (
            1.0
            if bool(getattr(state, "phase_clearance_latched", False))
            else -abs(float(self.params.get("failure_value", 1.0)))
        )


class TimedHeightTrackingComponent(RewardComponent):
    """Apply height tracking only during a configured time interval."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        end_time = self.params.get("end_time")
        elapsed = calculator.step_counter * calculator.dt
        if elapsed < start_time or (end_time is not None and elapsed >= float(end_time)):
            return 0.0
        desired_height = self.params.get("desired_height")
        if desired_height is None or desired_height == -1:
            desired_height = state.sim_init_pos[2]
        tracking_sigma = self.params.get("tracking_sigma", 0.005)
        return isaac_reward(
            desired_height,
            state.accurate_pos_world[2],
            tracking_sigma,
        )


class SingleLegSupportPenaltyComponent(RewardComponent):
    """Penalizes the single-leg support exploit while spinning."""

    def _active_floor_contacts(self, state) -> list[int]:
        contact_source = self.params.get("contact_source", "geoms")
        if contact_source == "geoms":
            contacts = list(getattr(state, "contact_floor_geoms", []))
        elif contact_source == "balls":
            contacts = list(getattr(state, "contact_floor_balls", []))
        elif contact_source == "socks":
            contacts = list(getattr(state, "contact_floor_socks", []))
        else:
            contacts = list(getattr(state, "contact_floor_geoms", []))

        geom_filters = self.params.get("geom_name_contains")
        if geom_filters and state.mj_model is not None:
            contacts = [
                geom
                for geom in contacts
                if any(token in state.mj_model.geom(geom).name for token in geom_filters)
            ]
        return contacts

    def calculate(self, state, calculator) -> float:
        n_contacts = len(self._active_floor_contacts(state))
        # Main exploit guard: leaning or tapping with exactly one foot.
        if n_contacts == 1:
            return -1.0
        return 0.0


class JumpRewardComponent(RewardComponent):
    """Reward upward velocity only in the configured post-settle launch window."""

    def calculate(self, state, calculator) -> float:
        elapsed = calculator.step_counter * calculator.dt
        if elapsed < float(self.params.get("start_time", 0.0)):
            return 0.0
        end_time = self.params.get("end_time")
        if end_time is not None and elapsed >= float(end_time):
            return 0.0
        command_name = self.params.get("command_name")
        if command_name is not None:
            active_threshold = float(self.params.get("active_threshold", 0.5))
            if float(state.get_command_by_name(str(command_name))) < active_threshold:
                return 0.0
        if bool(self.params.get("require_ground_contact", False)) and not getattr(
            state, "contact_floor_geoms", []
        ):
            return 0.0
        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        upward_vel = np.dot(state.accurate_vel_body, -accurate_projected_gravity)
        max_vel = self.params.get("max_velocity", 1.0)
        return np.clip(upward_vel, 0, max_vel)


class OrientationRewardComponent(RewardComponent):
    """Rewards maintaining upright orientation."""

    def calculate(self, state, calculator) -> float:
        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        return np.dot(calculator.projected_upward_vec, -accurate_projected_gravity)


class HeightTrackingComponent(RewardComponent):
    """Tracks desired height."""

    def calculate(self, state, calculator) -> float:
        desired_height = self.params.get("desired_height")
        if desired_height is None or desired_height == -1:
            desired_height = state.sim_init_pos[2]
        tracking_sigma = self.params.get("tracking_sigma", 0.005)

        height = state.accurate_pos_world[2]
        return isaac_reward(desired_height, height, tracking_sigma)


class TorsoContactPenaltyComponent(RewardComponent):
    """Penalizes torso touching the ground."""

    def calculate(self, state, calculator) -> float:
        torso_geoms = self.params.get("torso_geoms", ["left0", "right0"])
        torso_touch_floor = np.any(
            [
                state.mj_model.geom(geom).name in torso_geoms
                for geom in state.contact_floor_balls
            ]
        )
        return -float(torso_touch_floor)


class LowHeightPenaltyComponent(RewardComponent):
    """Penalizes torso getting too close to the ground."""

    def calculate(self, state, calculator) -> float:
        min_height = float(self.params.get("min_height", 0.4))
        height = float(state.accurate_pos_world[2])
        deficit = max(min_height - height, 0.0)
        if deficit == 0.0:
            return 0.0

        # penalty_scale makes the hinge independent of raw meter units.  The
        # default 1.0 preserves the historical behavior for existing configs.
        penalty_scale = max(float(self.params.get("penalty_scale", 1.0)), 1e-6)
        penalty = np.square(deficit / penalty_scale)
        max_penalty = self.params.get("max_penalty")
        if max_penalty is not None:
            penalty = min(penalty, abs(float(max_penalty)))
        return -float(penalty)


class DOFPositionTrackingComponent(RewardComponent):
    """Tracks desired DOF positions."""

    def calculate(self, state, calculator) -> float:
        tracking_sigma = self.params.get("tracking_sigma", 10.0)
        target_positions = self.params.get("target_positions")
        if target_positions is None:
            target_positions = state.cfg.control.default_dof_pos

        return isaac_reward(
            normalize_angle(np.array(target_positions)),
            normalize_angle(state.accurate_dof_pos),
            tracking_sigma,
        )


class TimedDOFPositionTrackingComponent(DOFPositionTrackingComponent):
    """Dense bounded joint-pose score activated after a launch phase."""

    def calculate(self, state, calculator) -> float:
        start_time = float(self.params.get("start_time", 0.0))
        elapsed = calculator.step_counter * calculator.dt - start_time
        if elapsed < 0.0:
            return 0.0
        end_time = self.params.get("end_time")
        if (
            end_time is not None
            and calculator.step_counter * calculator.dt >= float(end_time)
        ):
            return 0.0

        target_positions = self.params.get("target_positions")
        if target_positions is None:
            target_positions = state.cfg.control.default_dof_pos
        error = normalize_angle(np.asarray(target_positions)) - normalize_angle(
            np.asarray(state.accurate_dof_pos)
        )
        tracking_scale = max(float(self.params.get("tracking_scale", 0.75)), 1e-6)
        max_penalty = max(float(self.params.get("max_penalty", 2.0)), 0.0)
        pose_score = 1.0 - float(np.mean(np.square(error / tracking_scale)))
        pose_score = float(np.clip(pose_score, -max_penalty, 1.0))

        ramp_time = max(float(self.params.get("ramp_time", 0.0)), 0.0)
        ramp = 1.0 if ramp_time == 0.0 else float(
            np.clip(elapsed / ramp_time, 0.0, 1.0)
        )
        return ramp * pose_score


class PlateauAngularVelocityComponent(RewardComponent):
    """Plateau-style reward for angular velocity using jing vector."""

    def calculate(self, state, calculator) -> float:
        from metamachine.utils.visual_utils import get_jing_vector

        ang_vel = state.accurate_ang_vel_body
        jing_vec = get_jing_vector(state.dof_pos[0], calculator.theta)
        ang_vel_forward = np.dot(jing_vec, ang_vel)

        target_velocity = self.params.get("target_velocity", 6.0)
        max_step_limit = self.params.get("max_step_velocity_limit", 2e5)
        velocity_cap = self.params.get("velocity_cap", 12.0)

        # Apply velocity cap if still in early training
        if target_velocity > velocity_cap and calculator.step_counter < max_step_limit:
            target_velocity = velocity_cap

        return plateau(ang_vel_forward, target_velocity)


class PlateauSpinComponent(RewardComponent):
    """Plateau-style reward for spinning around gravity axis."""

    def calculate(self, state, calculator) -> float:
        end_time = self.params.get("end_time")
        if (
            end_time is not None
            and calculator.step_counter * calculator.dt >= float(end_time)
        ):
            return 0.0

        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        spin_value = np.dot(-accurate_projected_gravity, state.accurate_ang_vel_body)

        target_spin = self.params.get("target_spin", 0.0)

        if target_spin > 0:
            score = plateau(spin_value, target_spin)
        elif target_spin < 0:
            score = plateau(-spin_value, -target_spin)
        else:
            score = -np.square(spin_value)
        return _phase_fade_scale(self.params, calculator) * score


class PlateauHeightComponent(RewardComponent):
    """Plateau-style reward for height tracking."""

    def calculate(self, state, calculator) -> float:
        height = state.accurate_pos_world[2]
        target_height = self.params.get("target_height", 0.0)
        return plateau(height, target_height)


class RecoveryRewardComponent(RewardComponent):
    """Combined DOF position tracking and orientation reward."""

    def calculate(self, state, calculator) -> float:
        tracking_sigma = self.params.get("tracking_sigma", 10.0)

        # DOF position tracking
        dof_reward = isaac_reward(
            normalize_angle(np.array(state.default_dof_pos)),
            normalize_angle(state.accurate_dof_pos),
            tracking_sigma,
        )

        # Orientation reward
        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        upward_reward = np.dot(
            calculator.projected_upward_vec, -accurate_projected_gravity
        )

        return dof_reward * upward_reward


class JumpTimerComponent(RewardComponent):
    """Manages jump timing without providing reward."""

    def __init__(self, name: str, weight: float = 0.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.jump_timer = 0

    def calculate(self, state, calculator) -> float:
        jump_time = self.params.get("jump_time", 50)
        jump_sig = state.commands[0]

        if jump_sig:
            self.jump_timer += 1
            if self.jump_timer > jump_time:
                state.commands[0] = 0
                self.jump_timer = 0

        return 0

    def reset(self) -> None:
        self.jump_timer = 0


class TripodJumpComponent(RewardComponent):
    """Complex tripod jumping behavior with state-dependent rewards."""

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.jump_timer = 0

    def calculate(self, state, calculator) -> float:
        jump_sig = state.commands[0]
        flying = len(state.contact_floor_geoms) == 0

        stationary_height = self.params.get("stationary_height", 0.5)
        jumping_height = self.params.get("jumping_height", 1.0)
        spinning_speed = self.params.get("spinning_speed", 5.0)
        weights = self.params.get("sub_weights", [1, 1, 100, 0, 0, 0])

        desired_height = jumping_height if jump_sig else stationary_height
        height = state.accurate_pos_world[2]

        # DOF tracking
        dof_reward = isaac_reward(
            normalize_angle(np.array(state.default_dof_pos)),
            normalize_angle(state.accurate_dof_pos),
            10.0,
        )

        # Orientation
        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        upward_reward = np.dot(
            calculator.projected_upward_vec, -accurate_projected_gravity
        )

        if not jump_sig:
            pos_reward = dof_reward * upward_reward
            height_track_reward = 0
            jump_bonus = 0
        else:
            pos_reward = 0
            height_track_reward = plateau(height, desired_height)
            self.jump_timer += 1

            if height > desired_height and flying:
                height_track_reward = 0
                jump_bonus = 1
                state.commands[0] = 0
                self.jump_timer = 0
            else:
                jump_bonus = 0

        # Spin rewards
        spin = np.dot(-accurate_projected_gravity, state.accurate_ang_vel_body)
        if jump_sig:
            spin_reward = plateau(spin, spinning_speed)
            spin_bonus = plateau(spin, spinning_speed)
        else:
            spin_reward = isaac_reward(0, spin, 0.1)
            spin_bonus = 0

        up_dir_dot = np.dot([0, 0, 1], -accurate_projected_gravity)

        reward_terms = np.array(
            [
                pos_reward,
                height_track_reward,
                jump_bonus,
                spin_reward,
                up_dir_dot,
                spin_bonus,
            ]
        )

        return np.sum(weights * reward_terms)

    def reset(self) -> None:
        self.jump_timer = 0


class ActionRateComponent(RewardComponent):
    """Rewards action rate."""

    def calculate(self, state, calculator) -> float:
        last_action = state.action_history.last_last_action
        current_action = state.action_history.last_action
        action_rate = np.sum(np.square(current_action - last_action)) / calculator.dt
        return action_rate


class TimedActionRatePenaltyComponent(RewardComponent):
    """Bounded post-phase cost for continually changing position targets."""

    def calculate(self, state, calculator) -> float:
        if self.params.get("require_phase_clearance_latched", False) and not bool(
            getattr(state, "phase_clearance_latched", False)
        ):
            return 0.0
        start_time = float(self.params.get("start_time", 0.0))
        elapsed = calculator.step_counter * calculator.dt - start_time
        if elapsed < 0.0:
            return 0.0

        ramp_time = max(float(self.params.get("ramp_time", 0.0)), 0.0)
        ramp = (
            1.0
            if ramp_time == 0.0
            else float(np.clip(elapsed / ramp_time, 0.0, 1.0))
        )
        free_delta = max(float(self.params.get("free_action_delta", 0.01)), 0.0)
        delta_scale = max(float(self.params.get("action_delta_scale", 0.10)), 1e-6)
        max_penalty = max(float(self.params.get("max_penalty", 4.0)), 0.0)
        current_action = np.asarray(
            state.action_history.last_action,
            dtype=np.float64,
        )
        last_action = np.asarray(
            state.action_history.last_last_action,
            dtype=np.float64,
        )
        excess = np.maximum(np.abs(current_action - last_action) - free_delta, 0.0)
        penalty = float(np.mean(np.square(excess / delta_scale)))
        return -ramp * min(penalty, max_penalty)


class ActionRateRateComponent(RewardComponent):
    """Penalizes changes in action rate (second-order action smoothness)."""

    def calculate(self, state, calculator) -> float:
        current_action = state.action_history.last_action
        last_action = state.action_history.last_last_action
        last_last_action = state.action_history.last_last_last_action
        second_diff = current_action - 2.0 * last_action + last_last_action
        action_rate_rate = np.sum(np.square(second_diff)) / calculator.dt
        return action_rate_rate


class ActionMagnitudePenaltyComponent(RewardComponent):
    """Ant-style L2 control cost on the instantaneous action magnitude."""

    def calculate(self, state, calculator) -> float:
        action = np.asarray(state.action_history.last_action, dtype=np.float64)
        action_limit = max(abs(float(self.params.get("action_limit", 1.0))), 1e-6)
        return -float(np.sum(np.square(action / action_limit)))


class WorldZVelocityPenaltyComponent(RewardComponent):
    """Penalize vertical torso bouncing above a small free-motion allowance."""

    def calculate(self, state, calculator) -> float:
        free_velocity = abs(float(self.params.get("free_velocity", 0.0)))
        tracking_sigma = max(float(self.params.get("tracking_sigma", 0.25)), 1e-6)
        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))
        excess = max(abs(float(np.asarray(vel_world)[2])) - free_velocity, 0.0)
        return -np.square(excess) / tracking_sigma


class RollPitchAngularVelocityPenaltyComponent(RewardComponent):
    """Penalize excessive roll/pitch rate while leaving yaw motion unpenalized."""

    def calculate(self, state, calculator) -> float:
        free_angular_velocity = abs(
            float(self.params.get("free_angular_velocity", 0.0))
        )
        tracking_sigma = max(float(self.params.get("tracking_sigma", 1.0)), 1e-6)
        up_body = -quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)
        up_body = up_body / (np.linalg.norm(up_body) + 1e-8)
        angular_velocity = np.asarray(state.accurate_ang_vel_body, dtype=np.float64)
        yaw_rate = float(np.dot(angular_velocity, up_body))
        roll_pitch_rate = float(
            np.sqrt(
                max(
                    float(np.dot(angular_velocity, angular_velocity)) - yaw_rate**2,
                    0.0,
                )
            )
        )
        excess = max(roll_pitch_rate - free_angular_velocity, 0.0)
        return -np.square(excess) / tracking_sigma


class YawAngularVelocityPenaltyComponent(RewardComponent):
    """Penalize excess turning about gravity while allowing small corrections."""

    def calculate(self, state, calculator) -> float:
        free_angular_velocity = abs(
            float(self.params.get("free_angular_velocity", 0.0))
        )
        tracking_sigma = max(float(self.params.get("tracking_sigma", 1.0)), 1e-6)
        up_body = -quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)
        up_body = up_body / (np.linalg.norm(up_body) + 1e-8)
        angular_velocity = np.asarray(state.accurate_ang_vel_body, dtype=np.float64)
        yaw_rate = abs(float(np.dot(angular_velocity, up_body)))
        excess = max(yaw_rate - free_angular_velocity, 0.0)
        return -np.square(excess) / tracking_sigma


class InitialHeadingStabilityComponent(RewardComponent):
    """Penalize deviation from the torso heading sampled at reset."""
    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.initial_heading = None
    def reset(self) -> None:
        self.initial_heading = None
    def calculate(self, state, calculator) -> float:
        heading = float(np.asarray(state.derived.heading).reshape(-1)[0])
        if self.initial_heading is None:
            self.initial_heading = heading
            return 0.0
        error = float(np.arctan2(np.sin(heading - self.initial_heading), np.cos(heading - self.initial_heading)))
        free = abs(float(self.params.get("free_deviation_radians", 0.0)))
        scale = max(float(self.params.get("tracking_sigma", 0.25)), 1e-6)
        cost = float(np.square(max(abs(error) - free, 0.0) / scale))
        return -min(cost, max(float(self.params.get("max_penalty", np.inf)), 0.0))


class UnwrappedAxisRotationComponent(RewardComponent):
    """Track signed unwrapped rotation about a body or gravity-aligned axis."""
    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.reset()
    def reset(self) -> None:
        self.accumulated_rotation = 0.0
        self.completed = False
        self.success = False
        self._completion_bonus_paid = False
        self._success_bonus_paid = False
    def _axis_rate(self, state, calculator):
        angular_velocity = np.asarray(state.accurate_ang_vel_body, dtype=np.float64)
        if str(self.params.get("axis_mode", "body")) == "gravity":
            axis = -quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)
        else:
            axis = np.asarray(self.params.get("axis", [1.0, 0.0, 0.0]), dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        rate = float(np.dot(angular_velocity, axis))
        return rate, float(np.linalg.norm(angular_velocity - rate * axis))
    def calculate(self, state, calculator) -> float:
        elapsed = calculator.step_counter * calculator.dt
        if elapsed < float(self.params.get("start_time", 0.0)):
            return 0.0
        end_time = self.params.get("end_time")
        if end_time is not None and elapsed >= float(end_time):
            return 0.0
        direction = 1.0 if float(self.params.get("direction", 1.0)) >= 0 else -1.0
        rate, off_axis = self._axis_rate(state, calculator)
        directed_rate = direction * rate
        self.accumulated_rotation += directed_rate * calculator.dt
        target_rate = max(float(self.params.get("target_rate", 4.0)), 1e-6)
        progress = float(np.clip(max(directed_rate, 0.0) / target_rate, 0.0, 1.0))
        off_axis_sigma = max(float(self.params.get("off_axis_sigma", 6.0)), 1e-6)
        axis_purity = float(np.exp(-np.square(off_axis) / off_axis_sigma))
        reward = progress * axis_purity
        # Net-progress mode charges reverse rotation, so alternating positive and
        # negative bursts cannot earn a large reward while accumulating zero turn.
        if str(self.params.get("progress_mode", "positive_rate")) == "signed_rate":
            reverse_scale = max(float(self.params.get("reverse_penalty_scale", 1.0)), 0.0)
            normalized_rate = directed_rate / target_rate
            reward = float(np.clip(normalized_rate, -reverse_scale, 1.0)) * axis_purity
        if bool(self.params.get("continuous", False)):
            return reward
        target = 2.0 * np.pi * max(float(self.params.get("target_turns", 1.0)), 1e-6)
        if self.accumulated_rotation >= target - abs(float(self.params.get("completion_tolerance_radians", 0.5))):
            self.completed = True
            if not self._completion_bonus_paid:
                reward += float(self.params.get("completion_bonus", 10.0))
                self._completion_bonus_paid = True
        if self.completed:
            upright = float(np.dot(calculator.projected_upward_vec, -quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)))
            residual = float(np.linalg.norm(state.accurate_ang_vel_body))
            settled = max(upright, 0.0) * float(np.exp(-np.square(residual) / max(float(self.params.get("settle_angular_sigma", 1.5)), 1e-6)))
            reward += float(self.params.get("settle_scale", 1.0)) * settled
            if settled >= float(self.params.get("success_settle_threshold", 0.8)):
                self.success = True
                if not self._success_bonus_paid:
                    reward += float(self.params.get("success_bonus", 20.0))
                    self._success_bonus_paid = True
        return reward


class JumpPeakRecoveryComponent(RewardComponent):
    """Record peak height, air time and takeoff, optionally rewarding recovery."""
    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.reset()
    def reset(self) -> None:
        self.initial_height = None
        self.peak_height = 0.0
        self.airborne_time = 0.0
        self.takeoff_vertical_velocity = 0.0
        self.was_airborne = False
        self.success = False
        self._success_bonus_paid = False
        self._grounded_baseline_steps = 0
        self.baseline_ready = False
    def calculate(self, state, calculator) -> float:
        height = float(state.accurate_pos_world[2])
        vel = np.asarray(state.accurate_vel_world, dtype=np.float64)
        settle_time = max(float(self.params.get("baseline_after_seconds", 0.0)), 0.0)
        min_grounded_steps = max(int(self.params.get("baseline_grounded_steps", 1)), 1)
        grounded = len(getattr(state, "contact_floor_geoms", [])) > 0
        if self.initial_height is None:
            if calculator.step_counter * calculator.dt >= settle_time and grounded:
                self._grounded_baseline_steps += 1
                if self._grounded_baseline_steps >= min_grounded_steps:
                    self.initial_height = height
                    self.baseline_ready = True
            else:
                self._grounded_baseline_steps = 0
            if self.initial_height is None:
                return 0.0
        relative_height = height - self.initial_height
        airborne = len(getattr(state, "contact_floor_geoms", [])) == 0
        if airborne:
            self.airborne_time += calculator.dt
            self.was_airborne = True
            self.takeoff_vertical_velocity = max(self.takeoff_vertical_velocity, float(vel[2]))

        # For jump tasks, do not count a tall supported stance as a jump.  Peak
        # height and its dense reward can be restricted to genuine flight.
        previous_peak = self.peak_height
        height_only_while_airborne = bool(self.params.get("height_only_while_airborne", False))
        if airborne or not height_only_while_airborne:
            self.peak_height = max(self.peak_height, relative_height)
        reward = max(self.peak_height - previous_peak, 0.0) / max(float(self.params.get("peak_height_scale", 0.25)), 1e-6)
        success_height = float(self.params.get("success_height", np.inf))
        require_recovery = bool(self.params.get("require_recovery", False))
        min_airborne_time = max(float(self.params.get("min_airborne_time", 0.0)), 0.0)
        enough_flight = self.airborne_time >= min_airborne_time
        if not require_recovery:
            # Maximum-height mode succeeds at the threshold immediately, but a
            # supported extension cannot qualify when flight is required.
            if self.peak_height >= success_height and enough_flight:
                self.success = True
                if not self._success_bonus_paid:
                    reward += float(self.params.get("success_bonus", 20.0))
                    self._success_bonus_paid = True
            return reward
        upright = float(np.dot(calculator.projected_upward_vec, -quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)))
        stable = (self.was_airborne and enough_flight and not airborne and relative_height >= -float(self.params.get("landing_height_tolerance", 0.12))
                  and upright >= float(self.params.get("upright_threshold", 0.8))
                  and float(np.linalg.norm(vel)) <= float(self.params.get("landing_velocity_threshold", 1.5))
                  and float(np.linalg.norm(state.accurate_ang_vel_body)) <= float(self.params.get("landing_angular_velocity_threshold", 2.0)))
        if stable and self.peak_height >= float(self.params.get("success_height", 0.20)):
            self.success = True
            if not self._success_bonus_paid:
                reward += float(self.params.get("success_bonus", 10.0))
                self._success_bonus_paid = True
        return reward

class ContactForcePenaltyComponent(RewardComponent):
    """Penalize excessive clipped floor-contact force without punishing stance."""

    def calculate(self, state, calculator) -> float:
        model = getattr(state, "mj_model", None)
        data = getattr(state, "mj_data", None)
        if model is None or data is None:
            return 0.0

        import mujoco

        threshold = max(float(self.params.get("force_threshold", 0.0)), 0.0)
        force_clip = max(
            float(self.params.get("force_clip", 250.0)), threshold + 1e-6
        )
        floor_tokens = self.params.get("floor_geom_name_contains", ["floor"])
        geom_filters = self.params.get("geom_name_contains")
        total_cost = 0.0

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1_name = model.geom(contact.geom1).name or ""
            geom2_name = model.geom(contact.geom2).name or ""
            geom1_is_floor = any(token in geom1_name for token in floor_tokens)
            geom2_is_floor = any(token in geom2_name for token in floor_tokens)
            if not (geom1_is_floor or geom2_is_floor):
                continue

            non_floor_name = geom2_name if geom1_is_floor else geom1_name
            if geom_filters and not any(token in non_floor_name for token in geom_filters):
                continue

            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            force_norm = float(np.linalg.norm(contact_force[:3]))
            normalized_excess = np.clip(
                (force_norm - threshold) / (force_clip - threshold), 0.0, 1.0
            )
            total_cost += float(np.square(normalized_excess))

        return -total_cost



def _get_valid_goal_distance(state) -> Optional[float]:
    """Read a usable goal distance from state, if available."""
    distance = float(getattr(state.raw, "goal_distance", -1.0))
    if not np.isfinite(distance) or distance < 0.0:
        return None
    return distance


class GoalDistancePenaltyComponent(RewardComponent):
    """Dense penalty on current goal distance."""

    def calculate(self, state, calculator) -> float:
        distance = _get_valid_goal_distance(state)
        if distance is None:
            return 0.0

        offset = float(self.params.get("offset", 0.0))
        clamp_max = self.params.get("clamp_max", None)

        distance = max(distance - offset, 0.0)
        if clamp_max is not None:
            distance = min(distance, float(clamp_max))

        return -distance


class GoalProgressComponent(RewardComponent):
    """Rewards reducing goal distance from one step to the next."""

    def calculate(self, state, calculator) -> float:
        delta = float(getattr(state.raw, "goal_distance_delta", 0.0))
        if not np.isfinite(delta):
            return 0.0

        clip_abs = self.params.get("clip_abs", None)
        if clip_abs is not None:
            clip_abs = abs(float(clip_abs))
            delta = float(np.clip(delta, -clip_abs, clip_abs))

        if bool(self.params.get("positive_only", False)):
            delta = max(delta, 0.0)

        return delta


class GoalSuccessBonusComponent(RewardComponent):
    """Bonus when the robot is within a success radius of the goal."""

    def calculate(self, state, calculator) -> float:
        distance = _get_valid_goal_distance(state)
        if distance is None:
            return 0.0

        success_distance = float(self.params.get("success_distance", 0.08))
        return float(distance <= success_distance)


class WindowedDisplacementEfficiencyComponent(RewardComponent):
    """
    Windowed displacement efficiency reward for locomotion.
    
    This reward component tracks position over a sliding window and computes:
    1. Speed: net displacement / time (how fast the robot moves toward its goal)
    2. Efficiency: net displacement / path length (how straight the path is)
    
    The final reward combines speed and efficiency, encouraging both fast and
    efficient locomotion without shaking or zigzagging.
    
    Parameters (in params dict):
        window_size: Number of steps to track (default: 100)
        speed_weight: Weight for speed component (default: 1.0)
        efficiency_weight: Weight for efficiency component (default: 0.5)
        use_weld_cluster: If True, use weld cluster average position; 
                         if False, use accurate_pos_world (default: True)
    
    Example YAML configuration:
        - name: windowed_efficiency
          type: windowed_displacement_efficiency
          weight: 1.0
          params:
            window_size: 100
            speed_weight: 1.0
            efficiency_weight: 0.5
            use_weld_cluster: true
    """

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.pos_history: list[np.ndarray] = []

    def calculate(self, state, calculator) -> float:
        window_size = self.params.get("window_size", 100)
        speed_weight = self.params.get("speed_weight", 1.0)
        efficiency_weight = self.params.get("efficiency_weight", 0.5)
        # print("efficiency_weight:", efficiency_weight)
        use_weld_cluster = self.params.get("use_weld_cluster", True)
        
        # Get current position
        if use_weld_cluster and state.mj_model is not None and state.mj_data is not None:
            from ...utils.mujoco_utils import get_largest_weld_cluster_average_pos
            
            result = get_largest_weld_cluster_average_pos(state.mj_model, state.mj_data)
            if result[1] is not None:
                curr_pos = result[1][:2]  # Use only x, y coordinates
            else:
                # Fallback to accurate position if no weld cluster found
                curr_pos = state.accurate_pos_world[:2].copy()
        else:
            # Use accurate position directly
            curr_pos = state.accurate_pos_world[:2].copy()
        
        self.pos_history.append(curr_pos)
        
        # Clamp window length
        if len(self.pos_history) > window_size:
            self.pos_history.pop(0)
        
        # Need at least 2 points to compute anything
        if len(self.pos_history) < 2:
            return 0.0
        
        # Get earliest position in window
        last_pos = self.pos_history[0]
        
        # Net displacement (straight-line distance from start to current)
        disp = np.linalg.norm(curr_pos - last_pos)
        
        # Total path length traveled (accumulated movement)
        path_len = sum(
            np.linalg.norm(self.pos_history[i + 1] - self.pos_history[i])
            for i in range(len(self.pos_history) - 1)
        )
        
        # Speed = net displacement / time
        time_elapsed = calculator.dt * len(self.pos_history)
        speed = disp / time_elapsed if time_elapsed > 0 else 0.0
        
        # Efficiency = how straight / non-shaky the path is (0 to 1)
        efficiency = disp / (path_len + 1e-6)
        
        # Final reward: weighted combination of speed and efficiency
        reward = speed_weight * speed + efficiency_weight * efficiency
        
        return reward

    def reset(self) -> None:
        self.pos_history = []


class WindowedTurningCurveTrackingComponent(RewardComponent):
    """
    Windowed turning reward based on net heading change over a sliding window.

    This is meant for commanded turning tasks where instantaneous torso yaw-rate
    is too noisy or oscillatory. The component tracks:
    1. Net heading change over the window, which suppresses fast left-right
       oscillations that produce little overall turning.
    2. Turning consistency, which penalizes oscillatory turning even when the
       average turn-rate looks reasonable.
    3. Optional curvature matching when both forward-speed and turn-rate
       commands are available.
    """

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.pos_history: list[np.ndarray] = []
        self.yaw_rate_history: list[float] = []

    def _get_yaw_rate(self, state, calculator) -> float:
        yaw_rate = getattr(state.derived, "yaw_rate", None)
        if yaw_rate is not None:
            return float(np.asarray(yaw_rate, dtype=np.float32).reshape(-1)[0])

        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        return float(np.dot(-accurate_projected_gravity, state.accurate_ang_vel_body))

    def _resolve_target(self, value: Any, state) -> float:
        if isinstance(value, str) and value.startswith("cmd:"):
            return float(state.get_command_by_name(value[4:]))
        return float(value)

    def _get_planar_position(self, state) -> np.ndarray:
        use_weld_cluster = self.params.get("use_weld_cluster", True)
        if use_weld_cluster and state.mj_model is not None and state.mj_data is not None:
            from ...utils.mujoco_utils import get_largest_weld_cluster_average_pos

            result = get_largest_weld_cluster_average_pos(state.mj_model, state.mj_data)
            if result[1] is not None:
                return np.asarray(result[1][:2], dtype=np.float32)

        if state.accurate.pos_world is not None:
            return np.asarray(state.accurate.pos_world[:2], dtype=np.float32)
        return np.asarray(state.raw.pos_world[:2], dtype=np.float32)

    def calculate(self, state, calculator) -> float:
        end_time = self.params.get("end_time")
        if (
            end_time is not None
            and calculator.step_counter * calculator.dt >= float(end_time)
        ):
            self.pos_history = []
            self.yaw_rate_history = []
            return 0.0

        phase_scale = _phase_fade_scale(self.params, calculator)

        window_size = int(self.params.get("window_size", 100))
        tracking_sigma = max(float(self.params.get("tracking_sigma", 0.5)), 1e-6)
        straight_tracking_sigma = max(
            float(self.params.get("straight_tracking_sigma", tracking_sigma)),
            1e-6,
        )
        consistency_weight = float(self.params.get("consistency_weight", 0.25))
        turn_command_deadband = float(self.params.get("turn_command_deadband", 0.05))
        min_path_length = float(self.params.get("min_path_length", 0.05))
        min_forward_speed = float(self.params.get("min_forward_speed", 0.05))

        curr_pos = self._get_planar_position(state)
        yaw_rate = self._get_yaw_rate(state, calculator)

        self.pos_history.append(curr_pos)
        self.yaw_rate_history.append(yaw_rate)

        if len(self.pos_history) > window_size:
            self.pos_history.pop(0)
            self.yaw_rate_history.pop(0)

        if len(self.pos_history) < 2:
            return 0.0

        time_elapsed = calculator.dt * max(1, len(self.yaw_rate_history) - 1)
        actual_turn_rate = float(np.mean(self.yaw_rate_history))
        net_heading_change = float(np.sum(self.yaw_rate_history)) * calculator.dt
        heading_path = float(np.sum(np.abs(self.yaw_rate_history))) * calculator.dt
        path_len = sum(
            np.linalg.norm(self.pos_history[i + 1] - self.pos_history[i])
            for i in range(len(self.pos_history) - 1)
        )

        target_turn_rate = self._resolve_target(
            self.params.get("target_turn_rate", 0.0),
            state,
        )
        target_forward_speed_cfg = self.params.get("target_forward_speed", None)
        target_forward_speed = (
            self._resolve_target(target_forward_speed_cfg, state)
            if target_forward_speed_cfg is not None
            else None
        )

        target_metric = target_turn_rate
        integrated_turn_rate = net_heading_change / max(time_elapsed, 1e-6)
        if (
            target_forward_speed is not None
            and abs(target_forward_speed) >= min_forward_speed
            and path_len >= min_path_length
        ):
            target_metric = target_turn_rate / target_forward_speed
            actual_metric = net_heading_change / path_len
        else:
            # Net rotation over the window, not mean |ω|. Stops in-place wiggle
            # from earning spin reward without visible turning.
            actual_metric = integrated_turn_rate

        tracking_reward = np.exp(-np.square(target_metric - actual_metric) / tracking_sigma)

        if abs(target_turn_rate) <= turn_command_deadband:
            leakage_rate = heading_path / max(time_elapsed, 1e-6)
            leakage_reward = np.exp(-np.square(leakage_rate) / straight_tracking_sigma)
            return phase_scale * float(tracking_reward * leakage_reward)

        turning_consistency = abs(net_heading_change) / (heading_path + 1e-6)
        consistency_scale = (1.0 - consistency_weight) + consistency_weight * turning_consistency
        return phase_scale * float(tracking_reward * consistency_scale)

    def reset(self) -> None:
        self.pos_history = []
        self.yaw_rate_history = []


class OneHotTurningComponent(RewardComponent):
    """
    Simple turning reward based on one-hot command vector.
    
    Uses a 3D one-hot command vector to determine behavior:
    - [1, 0, 0] = go straight: penalize angular velocity (want ~0)
    - [0, 1, 0] = turn left: reward positive angular velocity
    - [0, 0, 1] = turn right: reward negative angular velocity
    
    Angular velocity is computed as dot product of ang_vel_body and projected_gravity.
    Rewards are normalized to similar scale across all modes.
    
    Parameters (in params dict):
        max_ang_vel: Maximum angular velocity for clipping (default: 3.0)
        straight_sigma: Sigma for gaussian penalty when going straight (default: 0.5)
        command_names: List of 3 command names for [straight, left, right] 
                      (default: ["cmd_straight", "cmd_left", "cmd_right"])
    
    Example YAML configuration:
        - name: turning_reward
          type: onehot_turning
          weight: 0.5
          params:
            max_ang_vel: 3.0
            straight_sigma: 0.5
            command_names: ["cmd_straight", "cmd_left", "cmd_right"]
    """

    def calculate(self, state, calculator) -> float:
        max_ang_vel = self.params.get("max_ang_vel", 3.0)
        straight_sigma = self.params.get("straight_sigma", 0.5)
        command_names = self.params.get(
            "command_names", ["cmd_straight", "cmd_left", "cmd_right"]
        )
        
        # Get commands (one-hot vector)
        try:
            cmd_straight = state.get_command_by_name(command_names[0])
            cmd_left = state.get_command_by_name(command_names[1])
            cmd_right = state.get_command_by_name(command_names[2])
        except (AttributeError, ValueError):
            # Fallback: try to get from state.commands array
            commands = getattr(state, 'commands', np.array([1.0, 0.0, 0.0]))
            if len(commands) >= 3:
                cmd_straight, cmd_left, cmd_right = commands[0], commands[1], commands[2]
            else:
                cmd_straight, cmd_left, cmd_right = 1.0, 0.0, 0.0
        
        # Compute angular velocity around gravity axis
        # Positive = turning left (counter-clockwise when viewed from above)
        # Negative = turning right (clockwise when viewed from above)
        accurate_projected_gravity = quat_rotate_inverse(
            state.accurate_quat, calculator.gravity_vec
        )
        ang_vel = np.dot(state.accurate_ang_vel_body, -accurate_projected_gravity)
        
        # Compute reward based on mode
        # All rewards are normalized to [0, 1] range for similar scale
        
        if cmd_left > 0.5:
            # Turn left mode: reward positive angular velocity
            # Clip and normalize to [0, 1]
            reward = np.clip(ang_vel, 0, max_ang_vel) / max_ang_vel
            
        elif cmd_right > 0.5:
            # Turn right mode: reward negative angular velocity
            # Clip and normalize to [0, 1]
            reward = np.clip(-ang_vel, 0, max_ang_vel) / max_ang_vel
            
        else:
            # Straight mode: penalize angular velocity (want it close to 0)
            # Use gaussian-like reward: exp(-ang_vel^2 / sigma^2)
            reward = np.exp(-ang_vel**2 / (straight_sigma**2))
        
        return reward


class OneHotForwardComponent(RewardComponent):
    """
    Forward velocity reward that works with one-hot turning commands.
    
    Always rewards forward velocity, regardless of turning mode.
    This ensures the robot keeps moving forward while turning.
    
    Parameters (in params dict):
        target_velocity: Target forward velocity (default: 0.5)
        tracking_sigma: Sigma for tracking reward (default: 0.25)
    
    Example YAML configuration:
        - name: forward_reward
          type: onehot_forward
          weight: 0.6
          params:
            target_velocity: 0.5
            tracking_sigma: 0.25
    """

    def calculate(self, state, calculator) -> float:
        target_vel = self.params.get("target_velocity", 0.5)
        tracking_sigma = self.params.get("tracking_sigma", 0.25)
        
        # Get forward velocity in body frame projected onto forward direction
        projected_forward_vel = np.dot(
            state.accurate_vel_body, calculator.projected_forward_vec
        )
        
        # Exponential tracking reward
        lin_vel_error = np.square(target_vel - projected_forward_vel)
        return np.exp(-lin_vel_error / tracking_sigma)


class OneHotHeadingAlignmentComponent(RewardComponent):
    """
    Rewards aligning the robot's heading with a target world direction.
    The target direction changes depending on the active one-hot steering command.
    """

    def calculate(self, state, calculator) -> float:
        command_names = self.params.get(
            "command_names", ["cmd_straight", "cmd_left", "cmd_right"]
        )
        directions = self.params.get(
            "directions",
            [
                [0.0, 1.0, 0.0],   # North (+Y)
                [-1.0, 0.0, 0.0],  # West (-X)
                [1.0, 0.0, 0.0]    # East (+X)
            ]
        )

        # Get active command index
        active_idx = 0
        try:
            for i, name in enumerate(command_names):
                if state.get_command_by_name(name) > 0.5:
                    active_idx = i
                    break
        except (AttributeError, ValueError):
            commands = getattr(state, "commands", np.array([1.0, 0.0, 0.0]))
            for i, val in enumerate(commands[:len(command_names)]):
                if val > 0.5:
                    active_idx = i
                    break

        target_direction = np.asarray(directions[active_idx], dtype=np.float64)
        target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-8)

        body_forward = np.asarray(calculator.projected_forward_vec, dtype=np.float64)
        world_forward = quat_apply(state.accurate_quat, body_forward)

        world_forward_xy = world_forward[:2]
        target_xy = target_direction[:2]

        world_forward_xy = world_forward_xy / (
            np.linalg.norm(world_forward_xy) + 1e-8
        )
        target_xy = target_xy / (np.linalg.norm(target_xy) + 1e-8)

        alignment = float(np.dot(world_forward_xy, target_xy))
        return 0.5 * (alignment + 1.0)


def _resolve_hybrid_target_xy(state, params) -> np.ndarray:
    """Resolve target travel direction from cardinal one-hot or cos/sin commands."""
    command_names = params.get(
        "command_names", ["cmd_straight", "cmd_left", "cmd_right"]
    )
    directions = params.get(
        "directions",
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )

    try:
        for i, name in enumerate(command_names):
            if state.get_command_by_name(name) > 0.5:
                direction = np.asarray(directions[i], dtype=np.float64)
                target_xy = direction[:2]
                return target_xy / (np.linalg.norm(target_xy) + 1e-8)
    except (AttributeError, ValueError):
        commands = getattr(state, "commands", np.array([1.0, 0.0, 0.0]))
        for i, val in enumerate(commands[: len(command_names)]):
            if val > 0.5:
                direction = np.asarray(directions[i], dtype=np.float64)
                target_xy = direction[:2]
                return target_xy / (np.linalg.norm(target_xy) + 1e-8)

    cos_name = params.get("cos_command_name", "cmd_dir_cos")
    sin_name = params.get("sin_command_name", "cmd_dir_sin")
    cos_h = float(state.get_command_by_name(cos_name))
    sin_h = float(state.get_command_by_name(sin_name))
    target_xy = np.array([cos_h, sin_h], dtype=np.float64)
    return target_xy / (np.linalg.norm(target_xy) + 1e-8)


class HybridDirectionVelocityComponent(RewardComponent):
    """Track velocity along cardinal one-hot or continuous commanded direction."""

    def calculate(self, state, calculator) -> float:
        target_vel = self.params.get("target_velocity", 0.8)
        tracking_sigma = self.params.get("tracking_sigma", 0.2)
        target_xy = _resolve_hybrid_target_xy(state, self.params)

        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))

        projected_vel = float(np.dot(np.asarray(vel_world)[:2], target_xy))
        return np.exp(-np.square(target_vel - projected_vel) / tracking_sigma)


class CommandedPlanarProgressComponent(RewardComponent):
    """Reward signed commanded world-frame progress with a no-motion cost.

    Zero velocity is never a positive reward. Progress is negative below the
    minimum speed, zero at that threshold, and reaches one at target speed.
    Backwards travel is clipped to a bounded negative value.
    """

    def calculate(self, state, calculator) -> float:
        target_xy = _resolve_hybrid_target_xy(state, self.params)
        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))
        along_speed = float(np.dot(np.asarray(vel_world, dtype=np.float64)[:2], target_xy))
        minimum_speed = max(float(self.params.get("minimum_speed", 0.20)), 1e-6)
        target_speed = max(float(self.params.get("target_speed", 0.80)), minimum_speed + 1e-6)
        reverse_clip = max(float(self.params.get("reverse_clip", 1.0)), 0.0)
        progress = (along_speed - minimum_speed) / (target_speed - minimum_speed)
        return float(np.clip(progress, -reverse_clip, 1.0))


class HybridDirectionLateralPenaltyComponent(RewardComponent):
    """Penalizes off-axis motion relative to the commanded travel direction.

    Modes:
    - windowed_displacement (default): net lateral displacement over a sliding
      window. Allows step-to-step lateral oscillation during gait learning.
    - velocity: instantaneous lateral speed (harsh; can block gait discovery).
    """

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self.pos_history: list[np.ndarray] = []

    def _get_planar_position(self, state) -> np.ndarray:
        use_weld_cluster = self.params.get("use_weld_cluster", False)
        if use_weld_cluster and state.mj_model is not None and state.mj_data is not None:
            from ...utils.mujoco_utils import get_largest_weld_cluster_average_pos

            result = get_largest_weld_cluster_average_pos(state.mj_model, state.mj_data)
            if result[1] is not None:
                return np.asarray(result[1][:2], dtype=np.float32)

        if state.accurate.pos_world is not None:
            return np.asarray(state.accurate.pos_world[:2], dtype=np.float32)
        return np.asarray(state.raw.pos_world[:2], dtype=np.float32)

    def _lateral_velocity_penalty(self, state) -> float:
        tracking_sigma = max(float(self.params.get("tracking_sigma", 0.10)), 1e-6)
        target_xy = _resolve_hybrid_target_xy(state, self.params)

        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))

        vel_xy = np.asarray(vel_world, dtype=np.float64)[:2]
        lateral_vel = vel_xy - np.dot(vel_xy, target_xy) * target_xy
        lateral_speed_sq = float(np.dot(lateral_vel, lateral_vel))
        return -lateral_speed_sq / tracking_sigma

    def _windowed_displacement_penalty(self, state, calculator) -> float:
        tracking_sigma = max(float(self.params.get("tracking_sigma", 0.25)), 1e-6)
        window_size = int(self.params.get("window_size", 50))
        target_xy = _resolve_hybrid_target_xy(state, self.params)

        self.pos_history.append(self._get_planar_position(state))
        if len(self.pos_history) > window_size:
            self.pos_history.pop(0)

        if len(self.pos_history) < 2:
            return 0.0

        net_disp = self.pos_history[-1] - self.pos_history[0]
        lateral_disp = net_disp - np.dot(net_disp, target_xy) * target_xy
        lateral_drift_sq = float(np.dot(lateral_disp, lateral_disp))
        return -lateral_drift_sq / tracking_sigma

    def calculate(self, state, calculator) -> float:
        mode = str(self.params.get("mode", "windowed_displacement")).lower()
        if mode == "velocity":
            return self._lateral_velocity_penalty(state)
        if mode == "windowed_displacement":
            return self._windowed_displacement_penalty(state, calculator)
        raise ValueError(
            f"Unknown hybrid_direction_lateral_penalty mode '{mode}'. "
            "Use 'windowed_displacement' or 'velocity'."
        )

    def reset(self) -> None:
        self.pos_history = []


class HybridDirectionHeadingComponent(RewardComponent):
    """Align heading with cardinal one-hot or continuous commanded direction."""

    def calculate(self, state, calculator) -> float:
        target_xy = _resolve_hybrid_target_xy(state, self.params)

        body_forward = np.asarray(calculator.projected_forward_vec, dtype=np.float64)
        world_forward = quat_apply(state.accurate_quat, body_forward)
        world_forward_xy = world_forward[:2]
        world_forward_xy = world_forward_xy / (
            np.linalg.norm(world_forward_xy) + 1e-8
        )

        alignment = float(np.dot(world_forward_xy, target_xy))
        return 0.5 * (alignment + 1.0)


class HybridDirectionYawTrackingComponent(RewardComponent):
    """Reward yawing toward the commanded world heading."""

    def calculate(self, state, calculator) -> float:
        target_xy = _resolve_hybrid_target_xy(state, self.params)

        body_forward = np.asarray(calculator.projected_forward_vec, dtype=np.float64)
        world_forward = quat_apply(state.accurate_quat, body_forward)
        forward_xy = world_forward[:2]
        forward_xy = forward_xy / (np.linalg.norm(forward_xy) + 1e-8)

        cross_z = float(forward_xy[0] * target_xy[1] - forward_xy[1] * target_xy[0])
        dot = float(np.clip(np.dot(forward_xy, target_xy), -1.0, 1.0))
        heading_error = float(np.arctan2(cross_z, dot))

        turn_gain = float(self.params.get("turn_gain", 2.0))
        max_yaw_rate = abs(float(self.params.get("max_yaw_rate", 1.2)))
        desired_yaw_rate = float(
            np.clip(turn_gain * heading_error, -max_yaw_rate, max_yaw_rate)
        )

        tracking_sigma = max(float(self.params.get("tracking_sigma", 0.15)), 1e-6)
        projected_gravity = quat_rotate_inverse(state.accurate_quat, calculator.gravity_vec)
        yaw_rate = float(np.dot(-projected_gravity, state.accurate_ang_vel_body))
        return float(np.exp(-np.square(desired_yaw_rate - yaw_rate) / tracking_sigma))


class ProjectedForwardVelocityComponent(RewardComponent):
    """
    Rewards moving forward along the projected forward vector without a specific target, 
    encouraging the robot to move as fast as possible up to a clipping limit.

    Parameters:
        clip_max: Max rewarded velocity in m/s (default: 2.0)
        normalize: If True, scale reward to [0, 1] by dividing by clip_max (default: True)
    """
    def calculate(self, state, calculator) -> float:
        clip_max = self.params.get("clip_max", 2.0)
        normalize = self.params.get("normalize", True)

        # Get forward velocity in body frame projected onto forward direction
        projected_forward_vel = np.dot(
            state.accurate_vel_body, calculator.projected_forward_vec
        )
        
        safe_clip_max = max(float(clip_max), 1e-6)
        clipped_vel = np.clip(projected_forward_vel, 0.0, safe_clip_max)

        if normalize:
            return clipped_vel / safe_clip_max
        return clipped_vel


class LocalXVelocityComponent(RewardComponent):
    """
    Simple forward reward based on body-frame local x velocity.

    Encourages moving faster in +x direction, with clipping.

    Parameters (in params dict):
        clip_max: Max rewarded local x velocity in m/s (default: 2.0)
        normalize: If True, scale reward to [0, 1] by dividing by clip_max
                   (default: True)
    """

    def calculate(self, state, calculator) -> float:
        clip_max = self.params.get("clip_max", 2.0)
        normalize = self.params.get("normalize", True)

        safe_clip_max = max(float(clip_max), 1e-6)
        forward_x_vel = state.accurate_vel_body[0]
        clipped_vel = np.clip(forward_x_vel, 0.0, safe_clip_max)

        if normalize:
            return clipped_vel / safe_clip_max
        return clipped_vel


class GlobalSpeedComponent(RewardComponent):
    """
    Direction-agnostic reward based on global/world-frame speed magnitude.

    Encourages high speed regardless of movement direction, with clipping.

    Parameters (in params dict):
        clip_max: Max rewarded speed in m/s (default: 2.0)
        normalize: If True, scale reward to [0, 1] by dividing by clip_max
                   (default: True)
    """

    def calculate(self, state, calculator) -> float:
        clip_max = self.params.get("clip_max", 2.0)
        normalize = self.params.get("normalize", True)

        safe_clip_max = max(float(clip_max), 1e-6)
        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))

        speed = np.linalg.norm(vel_world)
        clipped_speed = np.clip(speed, 0.0, safe_clip_max)
        # print(f"Global speed: {speed:.3f}, clipped: {clipped_speed:.3f}")

        if normalize:
            return clipped_speed / safe_clip_max
        return clipped_speed


class WorldYVelocityTrackingComponent(RewardComponent):
    """Tracks world-frame velocity along the global +y axis."""

    def calculate(self, state, calculator) -> float:
        target_vel = self.params.get("target_velocity", 0.6)
        tracking_sigma = self.params.get("tracking_sigma", 0.15)

        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))

        y_vel = float(vel_world[1])
        vel_error = np.square(target_vel - y_vel)
        return np.exp(-vel_error / tracking_sigma)


class OneHotVelocityTrackingComponent(RewardComponent):
    """
    Rewards tracking a target linear velocity along a commanded world direction.
    The target world direction changes depending on the active one-hot steering command.
    """

    def calculate(self, state, calculator) -> float:
        target_vel = self.params.get("target_velocity", 0.8)
        tracking_sigma = self.params.get("tracking_sigma", 0.2)
        command_names = self.params.get(
            "command_names", ["cmd_straight", "cmd_left", "cmd_right"]
        )
        directions = self.params.get(
            "directions",
            [
                [0.0, 1.0, 0.0],   # North (+Y)
                [-1.0, 0.0, 0.0],  # West (-X)
                [1.0, 0.0, 0.0]    # East (+X)
            ]
        )

        # Get active command index
        active_idx = 0
        try:
            for i, name in enumerate(command_names):
                if state.get_command_by_name(name) > 0.5:
                    active_idx = i
                    break
        except (AttributeError, ValueError):
            commands = getattr(state, "commands", np.array([1.0, 0.0, 0.0]))
            for i, val in enumerate(commands[:len(command_names)]):
                if val > 0.5:
                    active_idx = i
                    break

        target_direction = np.asarray(directions[active_idx], dtype=np.float64)
        target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-8)

        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))

        # Project velocity onto target direction (XY plane)
        projected_vel = float(np.dot(vel_world[:2], target_direction[:2]))

        # Exponential tracking reward
        vel_error = np.square(target_vel - projected_vel)
        return np.exp(-vel_error / tracking_sigma)


class WorldXVelocityPenaltyComponent(RewardComponent):
    """Penalizes world-frame lateral drift along the global x axis."""

    def calculate(self, state, calculator) -> float:
        tracking_sigma = self.params.get("tracking_sigma", 0.10)

        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))

        x_vel = float(vel_world[0])
        return -np.square(x_vel) / tracking_sigma


class WorldYVelocityPenaltyComponent(RewardComponent):
    """Penalizes world-frame drift along the global y axis."""

    def calculate(self, state, calculator) -> float:
        tracking_sigma = self.params.get("tracking_sigma", 0.10)

        vel_world = getattr(state, "accurate_vel_world", None)
        if vel_world is None:
            vel_world = getattr(state, "vel_world", np.zeros(3))

        y_vel = float(vel_world[1])
        return -np.square(y_vel) / tracking_sigma


class HeadingAlignmentComponent(RewardComponent):
    """Rewards aligning the robot's forward heading with a target world direction."""

    def calculate(self, state, calculator) -> float:
        target_direction = np.asarray(
            self.params.get("target_direction", [0.0, 1.0, 0.0]), dtype=np.float64
        )
        target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-8)

        body_forward = np.asarray(calculator.projected_forward_vec, dtype=np.float64)
        world_forward = quat_apply(state.accurate_quat, body_forward)

        world_forward_xy = world_forward[:2]
        target_xy = target_direction[:2]

        world_forward_xy = world_forward_xy / (
            np.linalg.norm(world_forward_xy) + 1e-8
        )
        target_xy = target_xy / (np.linalg.norm(target_xy) + 1e-8)

        alignment = float(np.dot(world_forward_xy, target_xy))
        return 0.5 * (alignment + 1.0)


class StateCoveringIntrinsicRewardComponent(RewardComponent):
    """
    Intrinsic reward for state-covering skill discovery via RND.
    
    This component loads a pre-trained RNDCollection (trained on rollouts from
    existing policies) and computes an intrinsic reward that encourages the
    new policy to visit states different from all existing policies.
    
    Based on the ReST (Recurrent Skill Training) approach:
        reward = -log( (1/K) * sum_k exp(-alpha * rnd_error_k) )
    
    Where rnd_error_k is the RND prediction error for policy k.
    - When the current state is similar to states visited by existing policies,
      the RND error is LOW → reward is LOW (discouraged)
    - When the current state is novel (not visited by any existing policy),
      the RND error is HIGH → reward is HIGH (encouraged)
    
    Parameters (in params dict):
        rnd_collection_dir: Path to the saved RNDCollection directory (required)
        device: Device for RND inference (default: "cpu")
        reward_scale: Multiplier for the intrinsic reward (default: 1.0)
        reward_clip: Maximum reward value for clipping (default: 10.0)
    
    Example YAML configuration:
        - name: state_covering
          type: state_covering_intrinsic
          weight: 1.0
          params:
            rnd_collection_dir: "rnd_models"
            device: "cpu"
            reward_scale: 1.0
            reward_clip: 10.0
    """

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        super().__init__(name, weight, **kwargs)
        self._rnd_collection = None
        self._obs_dim = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the RND collection on first use."""
        if self._rnd_collection is not None:
            return

        from ...utils.rnd import RNDCollection

        rnd_dir = self.params.get("rnd_collection_dir")
        if rnd_dir is None:
            raise ValueError(
                "StateCoveringIntrinsicRewardComponent requires "
                "'rnd_collection_dir' parameter pointing to a saved RNDCollection."
            )

        device = self.params.get("device", "cpu")
        self._rnd_collection = RNDCollection.load(rnd_dir, device=device)
        self._obs_dim = self._rnd_collection.obs_dim
        print(f"[StateCovering] Loaded RNDCollection with "
              f"{self._rnd_collection.num_policies} policies, "
              f"obs_dim={self._obs_dim}")

    def calculate(self, state, calculator) -> float:
        """Calculate intrinsic reward based on state novelty.
        
        Uses the full (stacked) observation from the environment to compute
        the RND-based intrinsic reward.
        """
        self._ensure_loaded()

        import torch

        # Get the current observation from the state
        # Use the raw observation (before history stacking) for RND
        obs = state._construct_observation()
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32
        ).unsqueeze(0)

        # Handle dimension mismatch (e.g., if observation stacking is used)
        if obs_tensor.shape[-1] > self._obs_dim:
            # Take only the last obs_dim elements (most recent frame)
            obs_tensor = obs_tensor[:, -self._obs_dim:]
        elif obs_tensor.shape[-1] < self._obs_dim:
            # Pad with zeros if needed
            padding = torch.zeros(1, self._obs_dim - obs_tensor.shape[-1])
            obs_tensor = torch.cat([obs_tensor, padding], dim=-1)

        reward = self._rnd_collection.get_intrinsic_reward(obs_tensor)
        reward_val = reward.item()

        # Apply scaling and clipping
        reward_scale = self.params.get("reward_scale", 1.0)
        reward_clip = self.params.get("reward_clip", 10.0)
        reward_val = np.clip(reward_val * reward_scale, -reward_clip, reward_clip)

        return reward_val

    def reset(self) -> None:
        """Reset is a no-op for this component (RND collection persists)."""
        pass


# Component registry for easy lookup
COMPONENT_REGISTRY = {
    "linear_velocity_tracking": LinearVelocityTrackingComponent,
    "angular_velocity_tracking": AngularVelocityTrackingComponent,
    "commanded_axis_angular_velocity": CommandedAxisAngularVelocityComponent,
    "commanded_roll_flip_completion": CommandedRollFlipCompletionComponent,
    # 'linear_velocity_cmd_tracking': LinearVelocityTrackingCMDComponent,
    # 'angular_velocity_cmd_tracking': AngularVelocityTrackingCMDComponent,
    "contact_flight_time": ContactFlightTimeComponent,
    "persistent_foot_air_time_penalty": PersistentFootAirTimePenaltyComponent,
    "foot_slip_penalty": FootSlipPenaltyComponent,
    "excessive_foot_height_penalty": ExcessiveFootHeightPenaltyComponent,
    "dof_velocity_penalty": DOFVelocityPenaltyComponent,
    "timed_dof_velocity_penalty": TimedDOFVelocityPenaltyComponent,
    "dof_acceleration_penalty": DOFAccelerationPenaltyComponent,
    "contact_penalty": ContactPenaltyComponent,
    "timed_contact_penalty": TimedContactPenaltyComponent,
    "exponential_timed_contact_penalty": ExponentialTimedContactPenaltyComponent,
    "ramped_contact_penalty": RampedContactPenaltyComponent,
    "timed_airborne_spin": TimedAirborneSpinComponent,
    "timed_contact_free_spin": TimedContactFreeSpinComponent,
    "timed_spin_retention": TimedSpinRetentionComponent,
    "gated_passive_spin": GatedPassiveSpinComponent,
    "latched_momentum_retention": LatchedMomentumRetentionComponent,
    "whole_body_release_spin": WholeBodyReleaseSpinComponent,
    "clearance_latch_preparation": ClearanceLatchPreparationComponent,
    "phase_latch_success": PhaseLatchSuccessComponent,
    "timed_height_tracking": TimedHeightTrackingComponent,
    "single_leg_support_penalty": SingleLegSupportPenaltyComponent,
    "jump_reward": JumpRewardComponent,
    "orientation_reward": OrientationRewardComponent,
    "height_tracking": HeightTrackingComponent,
    "torso_contact_penalty": TorsoContactPenaltyComponent,
    "low_height_penalty": LowHeightPenaltyComponent,
    "dof_position_tracking": DOFPositionTrackingComponent,
    "timed_dof_position_tracking": TimedDOFPositionTrackingComponent,
    "plateau_angular_velocity": PlateauAngularVelocityComponent,
    "plateau_spin": PlateauSpinComponent,
    "plateau_height": PlateauHeightComponent,
    "recovery_reward": RecoveryRewardComponent,
    "jump_timer": JumpTimerComponent,
    "tripod_jump": TripodJumpComponent,
    "action_rate": ActionRateComponent,
    "timed_action_rate_penalty": TimedActionRatePenaltyComponent,
    "action_rate_rate": ActionRateRateComponent,
    "action_acceleration": ActionRateRateComponent,
    "action_magnitude_penalty": ActionMagnitudePenaltyComponent,
    "contact_force_penalty": ContactForcePenaltyComponent,
    "world_z_velocity_penalty": WorldZVelocityPenaltyComponent,
    "roll_pitch_angular_velocity_penalty": RollPitchAngularVelocityPenaltyComponent,
    "yaw_angular_velocity_penalty": YawAngularVelocityPenaltyComponent,
    "initial_heading_stability": InitialHeadingStabilityComponent,
    "unwrapped_axis_rotation": UnwrappedAxisRotationComponent,
    "jump_peak_recovery": JumpPeakRecoveryComponent,
    "goal_distance_penalty": GoalDistancePenaltyComponent,
    "goal_progress": GoalProgressComponent,
    "goal_success_bonus": GoalSuccessBonusComponent,
    "windowed_displacement_efficiency": WindowedDisplacementEfficiencyComponent,
    "windowed_turning_curve_tracking": WindowedTurningCurveTrackingComponent,
    "onehot_turning": OneHotTurningComponent,
    "onehot_forward": OneHotForwardComponent,
    "onehot_heading": OneHotHeadingAlignmentComponent,
    "onehot_velocity_tracking": OneHotVelocityTrackingComponent,
    "hybrid_direction_velocity": HybridDirectionVelocityComponent,
    "commanded_planar_progress": CommandedPlanarProgressComponent,
    "hybrid_direction_lateral_penalty": HybridDirectionLateralPenaltyComponent,
    "hybrid_direction_heading": HybridDirectionHeadingComponent,
    "hybrid_direction_yaw_tracking": HybridDirectionYawTrackingComponent,
    "projected_forward_velocity": ProjectedForwardVelocityComponent,
    "local_x_velocity": LocalXVelocityComponent,
    "global_speed": GlobalSpeedComponent,
    "world_y_velocity_tracking": WorldYVelocityTrackingComponent,
    "world_x_velocity_penalty": WorldXVelocityPenaltyComponent,
    "world_y_velocity_penalty": WorldYVelocityPenaltyComponent,
    "heading_alignment": HeadingAlignmentComponent,
    "state_covering_intrinsic": StateCoveringIntrinsicRewardComponent,
}


class RewardCalculator:
    """Elegant component-based reward calculator."""

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize reward calculator with component-based configuration.

        Args:
            cfg: Configuration with task.reward_components specification
        """
        self.cfg = cfg

        # Environment parameters
        self.dt = cfg.control.dt
        self.theta = getattr(
            cfg.environment, "theta", 0.610865
        )  # Default theta for robot

        # Reference vectors - get from observation section or use defaults
        observation = getattr(cfg, "observation", {})
        self.gravity_vec = observation.get("gravity_vec", [0, 0, -1])
        self.projected_forward_vec = observation.get("projected_forward_vec", [1, 0, 0])
        self.projected_upward_vec = observation.get("projected_upward_vec", [0, 0, 1])

        # Create components from configuration
        task = getattr(cfg, "task", {})
        reward_components = task.get("reward_components", [])
        self.components = self._create_components(reward_components)

        # Initialize state
        self.reset()

    def reset(self) -> None:
        """Reset reward calculator state."""
        self.step_counter = 0
        for component in self.components:
            component.reset()

    def calculate(self, state) -> tuple[float, dict[str, Any]]:
        """Calculate reward based on current state.

        Args:
            state: Current environment state

        Returns:
            tuple: (total_reward, info_dict)
        """
        component_values = {}
        total_reward = 0.0

        for component in self.components:
            value = component.calculate(state, self)
            weighted_value = component.weight * value
            total_reward += weighted_value
            component_values[component.name] = value

        info = {
            "component_values": component_values,
            "component_weights": {comp.name: comp.weight for comp in self.components},
            "total_reward": total_reward,
            "num_components": len(self.components),
        }

        self.step_counter += 1
        return total_reward, info

    def _create_components(self, component_configs: list) -> list[RewardComponent]:
        """Create reward components from configuration.

        Args:
            component_configs: List of component configuration dictionaries

        Returns:
            List of initialized reward components
        """
        if not component_configs:
            raise ValueError("No reward_components specified in config")

        components = []
        for config in component_configs:
            component_type = config["type"]
            component_name = config.get("name", component_type)
            component_weight = config.get("weight", 1.0)
            component_params = config.get("params", {})

            if component_type not in COMPONENT_REGISTRY:
                available_types = ", ".join(COMPONENT_REGISTRY.keys())
                raise ValueError(
                    f"Unknown component type: {component_type}. "
                    f"Available types: {available_types}"
                )

            component_class = COMPONENT_REGISTRY[component_type]
            component = component_class(
                component_name, component_weight, **component_params
            )
            components.append(component)

        return components

    @property
    def component_names(self) -> list[str]:
        """Get list of component names."""
        return [comp.name for comp in self.components]

    def get_component(self, name: str) -> Optional[RewardComponent]:
        """Get component by name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None

    def __str__(self) -> str:
        """String representation of the reward calculator."""
        lines = [f"RewardCalculator with {len(self.components)} components:"]
        for comp in self.components:
            lines.append(
                f"  - {comp.name}: {comp.__class__.__name__} (weight: {comp.weight})"
            )
        return "\n".join(lines)


def create_reward_calculator(cfg: OmegaConf) -> RewardCalculator:
    """Factory function to create a reward calculator.

    Args:
        cfg: Configuration object with task.reward_components

    Returns:
        Initialized RewardCalculator instance
    """
    return RewardCalculator(cfg)


def register_component(name: str, component_class: type):
    """Register a new reward component type.

    Args:
        name: Component type name for configuration
        component_class: RewardComponent subclass
    """
    if not issubclass(component_class, RewardComponent):
        raise ValueError("Component class must inherit from RewardComponent")

    COMPONENT_REGISTRY[name] = component_class


def list_available_components() -> list[str]:
    """Get list of all available component types."""
    return list(COMPONENT_REGISTRY.keys())

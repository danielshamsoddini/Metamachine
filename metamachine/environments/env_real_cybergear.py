"""
Real robot environment for CyberGear CAN motors.

This environment is intended for direct sim-to-real deployment of policies
trained in MetaMachine when the real robot is controlled through the
`CyberGearDriver` library instead of the capybarish/ESP32 transport.

Key design goals:
- Reuse MetaMachine's observation/action/reward pipeline through `Base`
- Preserve policy-facing action/observation semantics from simulation
- Keep hardware-specific sign flips and calibration in config
- Support both direct position commands and host-side torque PD

Expected config additions under `real`:

```yaml
environment:
  mode: real

real:
  backend: cybergear
  can_interface: socketcan
  can_channel: can0
  bitrate: 1000000
  module_ids: [42, 44, 46, 48]
  calibration_motor_ids: [41, 42, 43, 44, 45, 46, 47, 48]
  calibrate_on_start: false
  control_mode: operation_control
  poll_sleep: 0.002
  startup_settle_time: 0.5
  command_signs: [1, 1, 1, 1]
  observation_position_signs: [1, 1, 1, 1]
  observation_velocity_signs: [1, 1, 1, 1]
  action_smoothing_alpha: 1.0
  torque_limit: 8.0
  torque_kp: 12.0
  torque_kd: 1.0
  limit_speed: 8.0
  limit_current: 6.0
  limit_torque: 12.0
```
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from .base import Base

try:
    import can
    from CyberGearDriver import CyberGearMotor, CyberMotorMessage, RunMode
    from CyberGearDriver import constants as cybergear_constants

    CYBERGEAR_AVAILABLE = True
except ImportError:
    can = None
    CyberGearMotor = None
    CyberMotorMessage = None
    RunMode = None
    cybergear_constants = None
    CYBERGEAR_AVAILABLE = False

try:
    from bno08x_linux import (
        LinuxI2C,
        BNO08X,
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_GAME_ROTATION_VECTOR,
    )
    BNO08X_AVAILABLE = True
except ImportError:
    BNO08X_AVAILABLE = False

class CyberGearRealMetaMachine(Base):
    """Real MetaMachine environment backed by CyberGear CAN motors."""

    def __init__(self, cfg: OmegaConf) -> None:
        if not CYBERGEAR_AVAILABLE:
            raise ImportError(
                "CyberGear real mode requires python-can and CyberGearDriver."
            )

        self.cfg = cfg
        self.real_cfg = cfg.get("real", {})

        self.motor_ids: List[int] = list(
            self.real_cfg.get("module_ids", self.real_cfg.get("motor_ids", []))
        )
        if not self.motor_ids:
            raise ValueError(
                "real.module_ids must be provided for cybergear real environment"
            )
        if len(self.motor_ids) != int(cfg.control.num_actions):
            raise ValueError(
                f"real.module_ids length ({len(self.motor_ids)}) must match "
                f"control.num_actions ({cfg.control.num_actions})"
            )

        self.calibration_motor_ids: List[int] = list(
            self.real_cfg.get("calibration_motor_ids", self.motor_ids)
        )
        self.poll_sleep = float(self.real_cfg.get("poll_sleep", 0.002))
        self.startup_settle_time = float(self.real_cfg.get("startup_settle_time", 0.5))
        self.control_backend = str(
            self.real_cfg.get("control_mode", "operation_control")
        ).lower()
        self.calibrate_on_start = bool(self.real_cfg.get("calibrate_on_start", False))
        self.torque_limit = float(self.real_cfg.get("torque_limit", 8.0))
        self.torque_kp = float(self.real_cfg.get("torque_kp", cfg.control.get("kp", 12.0)))
        self.torque_kd = float(self.real_cfg.get("torque_kd", cfg.control.get("kd", 1.0)))
        self.limit_speed = float(self.real_cfg.get("limit_speed", 8.0))
        self.limit_current = float(self.real_cfg.get("limit_current", 6.0))
        self.limit_torque = float(self.real_cfg.get("limit_torque", 12.0))
        self.action_smoothing_alpha = float(
            self.real_cfg.get("action_smoothing_alpha", 1.0)
        )
        self.enable_motor_on_reset = bool(
            self.real_cfg.get("enable_motor_on_reset", True)
        )
        self.disable_motor_on_close = bool(
            self.real_cfg.get("disable_motor_on_close", True)
        )
        self.send_pause = float(self.real_cfg.get("send_pause_after_send", 1e-9))

        num_actions = len(self.motor_ids)
        self.command_signs = self._array_cfg("command_signs", num_actions, 1.0)
        self.obs_position_signs = self._array_cfg(
            "observation_position_signs", num_actions, 1.0
        )
        self.obs_velocity_signs = self._array_cfg(
            "observation_velocity_signs", num_actions, 1.0
        )

        if cybergear_constants is not None:
            cybergear_constants.PAUSE_AFTER_SEND = self.send_pause

        self.bus = can.interface.Bus(
            interface=str(self.real_cfg.get("can_interface", "socketcan")),
            channel=str(self.real_cfg.get("can_channel", "can0")),
            bitrate=int(self.real_cfg.get("bitrate", 1_000_000)),
        )

        self._all_motors: Dict[int, CyberGearMotor] = {
            motor_id: CyberGearMotor(motor_id=motor_id, send_message=self._send_message)
            for motor_id in sorted(set(self.calibration_motor_ids) | set(self.motor_ids))
        }
        self.motors: List[CyberGearMotor] = [self._all_motors[motor_id] for motor_id in self.motor_ids]
        self.notifier = can.Notifier(
            self.bus, [motor.message_received for motor in self._all_motors.values()]
        )

        self.motor_enabled = False
        self.last_motor_com_time = time.time()
        self.last_commanded_action = np.zeros(num_actions, dtype=np.float32)
        self.last_state_timestamp = {motor_id: 0.0 for motor_id in self.motor_ids}
        self.last_loop_start = None

        super().__init__(cfg)

        if BNO08X_AVAILABLE:
            try:
                i2c = LinuxI2C(7)
                self.bno = BNO08X(i2c, address=0x4B, debug=False)
                self.bno.enable_feature(BNO_REPORT_ACCELEROMETER, 40)
                self.bno.enable_feature(BNO_REPORT_GYROSCOPE, 40)
                self.bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR, 40)
                self.bno.set_quaternion_euler_vector(BNO_REPORT_GAME_ROTATION_VECTOR)
                print("BNO08x IMU initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize BNO08x IMU: {e}")
                self.bno = None
        else:
            self.bno = None


        self.kps = np.full(num_actions, float(cfg.control.get("kp", 12.0)), dtype=np.float32)
        self.kds = np.full(num_actions, float(cfg.control.get("kd", 1.0)), dtype=np.float32)
        self.rollout_log_enabled = bool(cfg.get("logging", {}).get("log_real_rollout", True))
        self.rollout_log_file = None
        self.rollout_step_idx = 0
        if self.rollout_log_enabled and getattr(self, "_log_dir", None):
            os.makedirs(self._log_dir, exist_ok=True)
            self.rollout_log_path = os.path.join(self._log_dir, "real_rollout.jsonl")
            self.rollout_log_file = open(self.rollout_log_path, "a", encoding="utf-8")
        else:
            self.rollout_log_path = None

        if self.calibrate_on_start:
            self.calibrate_zero_positions()

        self._request_motor_states()

    def _array_cfg(self, key: str, size: int, default: float) -> np.ndarray:
        value = self.real_cfg.get(key, None)
        if value is None:
            return np.full(size, default, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).flatten()
        if arr.size != size:
            raise ValueError(f"real.{key} must have length {size}, got {arr.size}")
        return arr

    def _send_message(self, message: CyberMotorMessage) -> None:
        self.bus.send(
            can.Message(
                arbitration_id=message.arbitration_id,
                data=message.data,
                is_extended_id=message.is_extended_id,
            )
        )

    def _request_motor_states(self) -> None:
        for motor_id in self.motor_ids:
            motor = self._all_motors[motor_id]
            try:
                motor.request_motor_state()
            except Exception:
                continue
        if self.poll_sleep > 0:
            time.sleep(self.poll_sleep)
        now = time.time()
        for motor_id in self.motor_ids:
            state = self._all_motors[motor_id].state
            if isinstance(state, dict) and "position" in state and "velocity" in state:
                self.last_state_timestamp[motor_id] = now

    def _sanitize_for_json(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        if isinstance(value, dict):
            return {k: self._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_json(v) for v in value]
        return value

    def _append_rollout_log(self, payload: Dict[str, Any]) -> None:
        if self.rollout_log_file is None:
            return
        self.rollout_log_file.write(
            json.dumps(self._sanitize_for_json(payload), separators=(",", ":")) + "\n"
        )
        self.rollout_log_file.flush()

    def ready(self) -> bool:
        self._request_motor_states()
        for motor_id in self.motor_ids:
            state = self._all_motors[motor_id].state
            if not isinstance(state, dict):
                return False
            if "position" not in state or "velocity" not in state:
                return False
        return True

    def calibrate_zero_positions(self) -> None:
        calibration_cfg = self.real_cfg.get("calibration", {})
        pos_speeds = calibration_cfg.get("positive_speeds", None)
        neg_speeds = calibration_cfg.get("negative_speeds", None)
        travel_time = float(calibration_cfg.get("travel_time", 5.0))
        midpoint_settle = float(calibration_cfg.get("midpoint_settle_time", 2.5))

        for idx, motor_id in enumerate(self.calibration_motor_ids):
            motor = self._all_motors[motor_id]
            motor.enable()
            motor.stop()
            time.sleep(0.1)
            motor.enable()
            motor.mode(RunMode.VELOCITY)
            motor.set_parameter("limit_cur", self.limit_current)
            motor.set_parameter("limit_torque", self.limit_torque)

            neg_speed = float(neg_speeds[idx]) if neg_speeds is not None else 3.0
            pos_speed = float(pos_speeds[idx]) if pos_speeds is not None else 3.0

            motor.set_parameter("spd_ref", -neg_speed)
            time.sleep(travel_time)
            motor.request_motor_state()
            time.sleep(self.poll_sleep)
            pos1 = float(motor.state.get("position", 0.0))

            motor.set_parameter("spd_ref", pos_speed)
            time.sleep(travel_time)
            motor.request_motor_state()
            time.sleep(self.poll_sleep)
            pos2 = float(motor.state.get("position", 0.0))

            midpoint = 0.5 * (pos1 + pos2)
            motor.mode(RunMode.POSITION)
            motor.set_parameter("limit_spd", self.limit_speed)
            motor.set_parameter("loc_ref", midpoint)
            time.sleep(midpoint_settle)
            motor.set_zero_position()
            motor.stop()

    def _enable_motors(self) -> None:
        for motor in self.motors:
            motor.enable()
            if self.control_backend == "operation_control":
                motor.mode(RunMode.OPERATION_CONTROL)
            elif self.control_backend == "torque_pd":
                motor.mode(RunMode.OPERATION_CONTROL)
            else:
                raise ValueError(
                    f"Unknown real.control_mode '{self.control_backend}'. "
                    "Use 'operation_control' or 'torque_pd'."
                )
            motor.set_parameter("limit_torque", self.limit_torque)
            if self.control_backend == "operation_control":
                motor.set_parameter("limit_spd", self.limit_speed)
        self.motor_enabled = True

    def _disable_motors(self) -> None:
        for motor in self._all_motors.values():
            try:
                motor.stop()
            except Exception:
                pass
        self.motor_enabled = False

    def _reset_robot(self) -> None:
        if self.enable_motor_on_reset and not self.motor_enabled:
            self._enable_motors()

        self._request_motor_states()
        default_pose = np.asarray(self.cfg.control.get("default_dof_pos", np.zeros(len(self.motor_ids))), dtype=np.float32)
        if default_pose.size != len(self.motor_ids):
            default_pose = np.resize(default_pose, len(self.motor_ids))

        if self.control_backend == "operation_control":
            for i, motor in enumerate(self.motors):
                target = float(self.command_signs[i] * default_pose[i])
                motor.control(position=target, velocity=0.0, torque=0.0, kp=float(self.kps[i]), kd=float(self.kds[i]))
        else:
            for motor in self.motors:
                motor.control(position=0.0, velocity=0.0, torque=0.0, kp=0.0, kd=0.0)

        self.last_commanded_action = default_pose.copy()
        self.rollout_step_idx = 0
        self.last_loop_start = time.time()
        time.sleep(self.startup_settle_time)

    def _get_observable_data(self) -> Dict[str, Any]:
        self._request_motor_states()

        dof_pos = np.zeros(len(self.motor_ids), dtype=np.float32)
        dof_vel = np.zeros(len(self.motor_ids), dtype=np.float32)
        init_pos = np.asarray(
            self.cfg.get("initialization", {}).get("init_pos", [0.0, 0.0, 0.3]),
            dtype=np.float32,
        )
        ident_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        zeros3 = np.zeros(3, dtype=np.float32)

        for i, motor_id in enumerate(self.motor_ids):
            state = self._all_motors[motor_id].state
            dof_pos[i] = self.obs_position_signs[i] * float(state.get("position", 0.0))
            dof_vel[i] = self.obs_velocity_signs[i] * float(state.get("velocity", 0.0))

        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        ang_vel_body = np.zeros(3, dtype=np.float32)
        acc = np.zeros(3, dtype=np.float32)
        
        if getattr(self, "bno", None) is not None:
            try:
                i, j, k, real = self.bno.quaternion
                quat = np.array([i, j, k, real], dtype=np.float32)
                
                gx, gy, gz = self.bno.gyro
                ang_vel_body = np.array([gx, gy, gz], dtype=np.float32)
                
                ax, ay, az = self.bno.acceleration
                acc = np.array([ax, ay, az], dtype=np.float32)
            except Exception:
                pass

        return {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "quat": quat,
            "ang_vel_body": ang_vel_body,
            "vel_body": zeros3,
            "pos_world": init_pos,
            "vel_world": zeros3,
            "accurate_quat": quat,
            "accurate_ang_vel_body": ang_vel_body,
            "accurate_vel_body": zeros3,
            "accurate_vel_world": zeros3,
            "accurate_pos_world": init_pos,
            "gyros": np.tile(ang_vel_body[None, :], (len(self.motor_ids), 1)),
            "quats": np.tile(quat[None, :], (len(self.motor_ids), 1)),
            "accs": np.tile(acc[None, :], (len(self.motor_ids), 1)),
            "goal_distances": np.full(len(self.motor_ids), -1.0, dtype=np.float32),
            "goal_distance": -1.0,
            "special_quat": quat,
            "timestamp": time.time(),
            "motor_ids": self.motor_ids,
        }

    def _perform_action(self, action: np.ndarray) -> Dict[str, Any]:
        if not self.motor_enabled:
            self._enable_motors()

        action = np.asarray(action, dtype=np.float32).flatten()
        alpha = float(np.clip(self.action_smoothing_alpha, 0.0, 1.0))
        smoothed = alpha * action + (1.0 - alpha) * self.last_commanded_action

        loop_start = time.time()
        self._request_motor_states()
        measured_pos = np.zeros(len(self.motors), dtype=np.float32)
        measured_vel = np.zeros(len(self.motors), dtype=np.float32)
        for i, motor in enumerate(self.motors):
            state = motor.state
            measured_pos[i] = self.obs_position_signs[i] * float(state.get("position", 0.0))
            measured_vel[i] = self.obs_velocity_signs[i] * float(state.get("velocity", 0.0))

        if self.control_backend == "operation_control":
            for i, motor in enumerate(self.motors):
                position_target = float(self.command_signs[i] * smoothed[i])
                motor.control(
                    position=position_target,
                    velocity=0.0,
                    torque=0.0,
                    kp=float(self.kps[i]),
                    kd=float(self.kds[i]),
                )
        else:
            self._request_motor_states()
            for i, motor in enumerate(self.motors):
                state = motor.state
                sim_pos = self.obs_position_signs[i] * float(state.get("position", 0.0))
                sim_vel = self.obs_velocity_signs[i] * float(state.get("velocity", 0.0))
                torque_sim = self.torque_kp * (smoothed[i] - sim_pos) - self.torque_kd * sim_vel
                torque_hw = float(np.clip(self.command_signs[i] * torque_sim, -self.torque_limit, self.torque_limit))
                motor.control(
                    torque=torque_hw,
                    velocity=0.0,
                    position=0.0,
                    kp=0.0,
                    kd=0.0,
                )

        self.last_commanded_action = smoothed
        self.last_motor_com_time = loop_start

        dt = float(self.cfg.control.dt)
        elapsed = time.time() - loop_start
        sleep_time = max(0.0, dt - elapsed)
        if elapsed < dt:
            time.sleep(sleep_time)
        send_dt = time.time() - loop_start
        self.last_loop_start = time.time()
        tracking_error = smoothed - measured_pos
        self.rollout_step_idx += 1

        self._append_rollout_log(
            {
                "step": self.rollout_step_idx,
                "timestamp": loop_start,
                "policy_action": action.copy(),
                "commanded_action": smoothed.copy(),
                "measured_pos": measured_pos,
                "measured_vel": measured_vel,
                "tracking_error": tracking_error,
                "command_signs": self.command_signs.copy(),
                "obs_position_signs": self.obs_position_signs.copy(),
                "obs_velocity_signs": self.obs_velocity_signs.copy(),
                "kp": self.kps.copy(),
                "kd": self.kds.copy(),
                "control_backend": self.control_backend,
                "compute_time": elapsed,
                "sleep_time": sleep_time,
                "loop_dt": send_dt,
                "motor_ids": self.motor_ids,
            }
        )

        return {
            "compute_time": elapsed,
            "send_dt": send_dt,
            "commands_sent": len(self.motors),
            "motor_enabled": self.motor_enabled,
            "tracking_error_mean_abs": float(np.mean(np.abs(tracking_error))),
        }

    def _post_done(self) -> None:
        if bool(self.real_cfg.get("disable_motor_on_done", False)):
            self._disable_motors()

    def close(self) -> None:
        if self.rollout_log_file is not None:
            try:
                self.rollout_log_file.close()
            except Exception:
                pass
        if self.disable_motor_on_close:
            self._disable_motors()
        if hasattr(self, "notifier") and self.notifier is not None:
            try:
                self.notifier.stop()
            except Exception:
                pass
        if hasattr(self, "bus") and self.bus is not None:
            try:
                self.bus.shutdown()
            except Exception:
                pass
        super().close()

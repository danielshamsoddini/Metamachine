"""Joint-space PD converted to the XML's geared motor-control units."""
import mujoco
import numpy as np

from .brachiation_scene import joint_indices


class BrachiationController:
    def __init__(self, cfg, model):
        self.mode = str(cfg.control.mode)
        if self.mode not in ("torque", "pd"):
            raise ValueError("control.mode must be torque or pd")
        self.qids, self.vids = joint_indices(model)
        self.action_scale = float(cfg.control.get("action_scale", 1.0))
        if self.mode == "torque":
            return
        self.gear = model.actuator_gear[:, 0].copy()
        if (np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
                or np.any(model.jnt_type[model.actuator_trnid[:, 0]] != mujoco.mjtJoint.mjJNT_HINGE)
                or np.any(self.gear == 0)
                or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE)
                or not np.allclose(model.actuator_gainprm[:, 0], 1.0)):
            raise ValueError("PD requires unit-gain geared hinge motors")
        for name in ("kp", "kd", "default_joint_pos", "target_lower", "target_upper", "torque_limits"):
            value = np.asarray(cfg.control[name], dtype=np.float64)
            if value.shape != (model.nu,) or not np.isfinite(value).all():
                raise ValueError(f"control.{name} must contain {model.nu} finite values")
            setattr(self, name, value)
        if np.any(self.kp <= 0) or np.any(self.kd < 0) or np.any(self.torque_limits <= 0):
            raise ValueError("PD gains/limits require kp > 0, kd >= 0, torque_limits > 0")
        bounds = model.jnt_range[model.actuator_trnid[:, 0]]
        if (np.any(self.target_lower >= self.target_upper)
                or np.any(self.default_joint_pos < self.target_lower)
                or np.any(self.default_joint_pos > self.target_upper)
                or np.any(self.target_lower < bounds[:, 0])
                or np.any(self.target_upper > bounds[:, 1])):
            raise ValueError("PD targets/defaults must be ordered and within joint limits")
        available = np.min(np.abs(model.actuator_ctrlrange * self.gear[:, None]), axis=1)
        if np.any(self.torque_limits > available):
            raise ValueError("PD torque_limits exceed the XML motor control limits")

    def targets(self, action, xp=np):
        action = xp.clip(action, -1., 1.)
        center = xp.asarray(self.default_joint_pos)
        span = xp.where(action >= 0, xp.asarray(self.target_upper) - center,
                        center - xp.asarray(self.target_lower))
        return center + action * span

    def controls(self, qpos, qvel, action, xp=np):
        if self.mode == "torque":
            return xp.clip(action, -1., 1.) * self.action_scale
        # Targets are held during a policy step; feedback is recomputed every substep.
        torque = (xp.asarray(self.kp) * (self.targets(action, xp) - qpos[self.qids])
                  - xp.asarray(self.kd) * qvel[self.vids])
        limits = xp.asarray(self.torque_limits)
        return xp.clip(torque, -limits, limits) / xp.asarray(self.gear)

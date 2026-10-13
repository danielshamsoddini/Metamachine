"""Joint-space PD converted to the XML's geared motor-control units."""
import mujoco
import numpy as np

from .brachiation_scene import joint_indices


class BrachiationController:
    def __init__(self, cfg, model):
        self.mode = str(cfg.control.mode)
        if self.mode not in ("torque", "pd"):
            raise ValueError("control.mode must be torque or pd")
        self.target_reference = str(cfg.control.get("target_reference", "default"))
        self.filter_alpha = float(cfg.control.get("filter_alpha", 1.0))
        self.relative = self.target_reference == "current"
        self.stateful = self.relative or self.filter_alpha != 1.0
        if self.target_reference not in ("default", "current") or not np.isfinite(self.filter_alpha) or not 0 < self.filter_alpha <= 1:
            raise ValueError("Invalid target_reference or filter_alpha")
        if self.stateful and self.mode != "pd":
            raise ValueError("Relative/filtered control requires PD mode")
        self.delta_scale = np.asarray(cfg.control.get("delta_scale", [0.25] * model.nu), dtype=float)
        if self.relative and (self.delta_scale.shape != (model.nu,) or not np.isfinite(self.delta_scale).all() or np.any(self.delta_scale <= 0)):
            raise ValueError("delta_scale must have one positive finite radian value per actuator")
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

    def prepare_step(self, qpos, action, previous_filtered, xp=np):
        filtered = (1 - self.filter_alpha) * previous_filtered + self.filter_alpha * xp.clip(action, -1., 1.)
        return filtered, self.targets(filtered, xp=xp, qpos=qpos)

    def targets(self, action, xp=np, qpos=None):
        action = xp.clip(action, -1., 1.)
        if self.relative:
            if qpos is None:
                raise ValueError("Current-relative targets require the policy-step starting qpos")
            return xp.clip(qpos[self.qids] + action * xp.asarray(self.delta_scale), xp.asarray(self.target_lower), xp.asarray(self.target_upper))
        center = xp.asarray(self.default_joint_pos)
        span = xp.where(action >= 0, xp.asarray(self.target_upper) - center,
                        center - xp.asarray(self.target_lower))
        return center + action * span

    def controls(self, qpos, qvel, action, xp=np, targets=None):
        if self.mode == "torque":
            return xp.clip(action, -1., 1.) * self.action_scale
        if self.stateful and targets is None:
            raise ValueError("Filtered/relative control requires targets frozen at the policy-step boundary")
        target = self.targets(action, xp) if targets is None else targets
        # Targets are held during a policy step; feedback is recomputed every substep.
        torque = (xp.asarray(self.kp) * (target - qpos[self.qids])
                  - xp.asarray(self.kd) * qvel[self.vids])
        limits = xp.asarray(self.torque_limits)
        return xp.clip(torque, -limits, limits) / xp.asarray(self.gear)

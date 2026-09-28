"""Explicit observation/reward schema for the brachiation task.

YAML selects the named primitives below; there is no evaluation of YAML as code.
Unknown fields, component names, and parameters fail before MJX compilation.
"""
from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np
from omegaconf import OmegaConf

OBSERVATIONS = {
    "projected_gravity": ("3", "torso-frame unit gravity direction", {}),
    "lin_vel_body": ("3", "torso-frame linear velocity, m/s", {}),
    "ang_vel_body": ("3", "torso-frame angular velocity, rad/s", {}),
    "dof_pos": ("nu", "joint angles in XML actuator order, rad", {}),
    "dof_vel": ("nu", "joint velocities in XML actuator order, rad/s", {}),
    "hand_positions": ("6", "left/right grip-site offsets from torso, m", {"frame": ("body", "world")}),
    "target_bar_positions": ("3*count", "target and following bar centers minus torso position, m; final bar repeated at course end", {"count": "positive_int", "frame": ("body", "world")}),
    "target_bar_axes": ("3*count", "unit directions of target and following finite bars; canonical body +Y axis", {"count": "positive_int", "frame": ("body", "world")}),
    "last_action": ("nu", "previous clipped normalized action", {}),
    "hook_contacts": ("2", "left/right hook contact with any bar, 0 or 1", {}),
    "cart_relative_position": ("3", "cart origin minus torso position, torso frame, m", {}),
    "cart_relative_velocity": ("3", "cart minus torso world linear velocity, torso frame, m/s", {}),
    "cart_projected_gravity": ("3", "world down expressed in cart frame", {}),
    "torso_height": ("1", "world torso height, m", {}),
    "pd_target": ("nu", "previous PD target angles, rad; PD mode only", {}),
    "tracking_error": ("nu", "previous PD target minus current joint angles, rad; PD mode only", {}),
}
REWARDS = {
    "forward_progress": ("dot(torso_after - torso_before, direction)", {"direction": "unit_vector"}),
    "target_distance": ("distance_before - distance_after to the SAME target bar; nearest hand", {"metric": ("bar_axis", "bar_center")}),
    "bar_reached": ("1 on sequential bar-completion event, otherwise 0", {}),
    "reach_improvement": ("increase in best eligible-hand proximity this target; bounded total <= 1 per target", {}),
    "reach_potential": ("training.discounting * Phi(next task state) - Phi(current task state); Phi=eligible-hand proximity, zero on task termination/completion; retained on truncation", {}),
    "unfinished_step": ("1 on nonterminal task steps, 0 on task termination; time-limit truncation is nonterminal", {}),
    "action_l2": ("sum(normalized_action ** 2); NOT torque", {}),
    "action_rate": ("sum((normalized_action - previous_action) ** 2)", {}),
    "fall": ("1 on task failure (height/lateral; cart_task also forbidden support, escape, hook-loss or stall), otherwise 0", {}),
    "torque_l2": ("mean over physics substeps of sum(applied_joint_torque ** 2), Nm^2", {}),
    "power_l1": ("mean over physics substeps of sum(abs(applied_joint_torque * joint_velocity)), W", {}),
    "joint_velocity_l2": ("sum(end_of_control_step_joint_velocity ** 2), (rad/s)^2", {}),
}


def _keys(value, required, optional, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{path}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _bool(value, path):
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")


def _positive_int(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")


def _finite(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{path} must be a finite number")


def _params(values, rules, path):
    _keys(values, rules, (), path)
    for key, rule in rules.items():
        value = values[key]
        if rule == "positive_int":
            _positive_int(value, f"{path}.{key}")
        elif rule == "unit_vector":
            a = np.asarray(value, dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all() or not np.isclose(np.linalg.norm(a), 1.):
                raise ValueError(f"{path}.{key} must be a unit 3-vector")
        elif value not in rule:
            raise ValueError(f"{path}.{key} must be one of {rule}")


def resolve_config(cfg):
    """Expand ONLY the documented pre-schema configs used by archived runs."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "observation" not in cfg and "reward" in cfg and "reward_components" not in cfg.task:
        warnings.warn("Legacy brachiation config: expanding the frozen v0 task recipe. "
                      "Inspect the resolved config and task manifest.", UserWarning)
        defaults = OmegaConf.load(Path(__file__).with_name("brachiation_legacy_recipe.yaml"))
        old = OmegaConf.to_container(cfg.reward)
        expected = {"forward_progress", "target_distance", "next_bar", "action_cost", "action_rate_cost", "fall"}
        if set(old) != expected:
            raise ValueError("Legacy reward keys do not match the archived brachiation recipe")
        cfg.observation = defaults.observation
        cfg.task = OmegaConf.merge(defaults.task, cfg.task)
        for term in cfg.task.reward_components:
            key = term.name
            if key not in old:
                continue
            term.weight = float(old[key]) * (-1 if key in ("action_cost", "action_rate_cost", "fall") else 1)
        cfg.task.nonfinite_reward = -float(old["fall"])
        del cfg["reward"]
    elif "reward" in cfg:
        raise ValueError("Use task.reward_components; the old reward mapping cannot be mixed with the new schema")
    return cfg


class TaskSpec:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.nu = model.nu
        self.joint_names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
        raw = OmegaConf.to_container(cfg, resolve=True)
        if "observation" not in raw:
            raise ValueError("Missing observation section")
        o = raw["observation"]
        _keys(o, ("components", "include_history_steps", "history_order", "history_reset", "clip_observations"), (), "observation")
        _positive_int(o["include_history_steps"], "observation.include_history_steps")
        if o["history_order"] not in ("oldest_first", "newest_first"):
            raise ValueError("observation.history_order must be oldest_first or newest_first")
        if o["history_reset"] not in ("repeat", "zeros"):
            raise ValueError("observation.history_reset must be repeat or zeros")
        self.history_steps, self.history_order, self.history_reset = o["include_history_steps"], o["history_order"], o["history_reset"]
        self.clip = o["clip_observations"]
        if self.clip is not None:
            _finite(self.clip, "observation.clip_observations")
            if self.clip <= 0:
                raise ValueError("observation.clip_observations must be positive or null")
        self.observations, self.layout = [], []
        offset, seen = 0, set()
        if not isinstance(o["components"], list):
            raise ValueError("observation.components must be a list")
        for component in o["components"]:
            _keys(component, ("name", "enabled", "scale", "noise_std", "params"), (), "observation component")
            name = component["name"]
            if name not in OBSERVATIONS or name in seen:
                raise ValueError(f"Unknown or duplicate observation component: {name}")
            seen.add(name)
            _bool(component["enabled"], name + ".enabled")
            _params(component["params"], OBSERVATIONS[name][2], name + ".params")
            if name in ("pd_target", "tracking_error") and component["enabled"] and cfg.control.mode != "pd":
                raise ValueError(f"{name} requires control.mode: pd")
            dim = self.nu if OBSERVATIONS[name][0] == "nu" else (3 * component["params"]["count"] if OBSERVATIONS[name][0] == "3*count" else int(OBSERVATIONS[name][0]))
            for field in ("scale", "noise_std"):
                a = np.asarray(component[field], dtype=float)
                if a.shape not in ((), (dim,)) or not np.isfinite(a).all() or (field == "noise_std" and np.any(a < 0)):
                    raise ValueError(f"{name}.{field} must be finite scalar or length-{dim} vector; noise_std must be nonnegative")
            if component["enabled"]:
                self.observations.append(component)
                self.layout.append({"name": name, "start": offset, "stop": offset + dim,
                                    "size": dim, "meaning": OBSERVATIONS[name][1],
                                    "scale": component["scale"], "noise_std": component["noise_std"],
                                    "params": component["params"]})
                offset += dim
        if offset == 0:
            raise ValueError("Enable at least one observation component")
        if cfg.bars.get("layout", "parallel") in ("spatial", "grid"):
            if not any(c["name"] == "target_bar_axes" for c in self.observations):
                raise ValueError("Spatial courses require target_bar_axes observations")
        self.frame_size, self.observation_size = offset, offset * self.history_steps
        self.noisy = any(np.any(np.asarray(c["noise_std"]) != 0) for c in self.observations)
        t = raw["task"]
        required = ("schema_version", "start", "remove_cart", "fall_height", "lateral_limit",
                    "target_tolerance", "contact_steps", "initial_bar_to_hand_z", "distance_metric",
                    "require_hook_contact", "contact_distance", "terminate_on_fall", "terminate_on_success",
                    "nonfinite_reward", "reward_clip", "reward_components")
        _keys(t, required, ("cart_task",), "task")
        self.cart_task = "cart_task" in t
        if self.cart_task:
            c = t["cart_task"]
            positive_ints = ("transfer_steps", "max_no_hook_steps", "max_no_transfer_steps")
            positive_floats = ("min_hook_force", "containment_margin", "max_torso_lift", "cart_tilt_limit", "reach_radius")
            _keys(c, positive_ints + positive_floats, (), "task.cart_task")
            for key in positive_ints:
                _positive_int(c[key], "task.cart_task." + key)
            for key in positive_floats:
                _finite(c[key], "task.cart_task." + key)
                if c[key] <= 0:
                    raise ValueError("task.cart_task." + key + " must be positive")
            if c["cart_tilt_limit"] >= np.pi / 2:
                raise ValueError("cart_tilt_limit must be below pi/2")
            if c["max_no_transfer_steps"] <= c["transfer_steps"]:
                raise ValueError("Transfer deadline must exceed hold duration")
            if t["start"] != "hanging_in_cart" or t["remove_cart"] or not t["require_hook_contact"] or not t["terminate_on_fall"]:
                raise ValueError("cart_task requires cart start, retained cart, hook contact, and fall termination")
            if "cradle" not in raw.get("cart", {}):
                raise ValueError("cart_task requires a passive cart.cradle")
        for component in self.observations:
            if component["name"].startswith("cart_") and not self.cart_task:
                raise ValueError("Cart observations require task.cart_task")
        if t["schema_version"] != 1:
            raise ValueError("Only task.schema_version: 1 is supported")
        for key in ("require_hook_contact", "terminate_on_fall", "terminate_on_success", "remove_cart"):
            _bool(t[key], "task." + key)
        for key in ("fall_height", "lateral_limit", "target_tolerance", "initial_bar_to_hand_z", "contact_distance", "nonfinite_reward"):
            _finite(t[key], "task." + key)
        if t["target_tolerance"] <= 0 or t["lateral_limit"] <= 0:
            raise ValueError("Task distance tolerances must be positive")
        _positive_int(t["contact_steps"], "task.contact_steps")
        if t["distance_metric"] not in ("bar_axis", "bar_center"):
            raise ValueError("task.distance_metric must be bar_axis or bar_center")
        self.reward_clip = t["reward_clip"]
        if self.reward_clip is not None:
            a = np.asarray(self.reward_clip, dtype=float)
            if a.shape != (2,) or not np.isfinite(a).all() or a[0] >= a[1]:
                raise ValueError("task.reward_clip must be [min, max] or null")
        if not isinstance(t["reward_components"], list):
            raise ValueError("task.reward_components must be a list")
        self.rewards, seen = [], set()
        for term in t["reward_components"]:
            _keys(term, ("name", "type", "enabled", "weight", "scale_by_dt", "params"), (), "reward component")
            name, kind = term["name"], term["type"]
            if not isinstance(name, str) or not name or "/" in name or name in seen:
                raise ValueError(f"Reward names must be unique nonempty names without '/': {name}")
            if kind not in REWARDS:
                raise ValueError(f"Unknown reward type: {kind}")
            seen.add(name)
            _bool(term["enabled"], name + ".enabled")
            _bool(term["scale_by_dt"], name + ".scale_by_dt")
            _finite(term["weight"], name + ".weight")
            _params(term["params"], REWARDS[kind][1], name + ".params")
            if term["enabled"]:
                if kind in ("reach_improvement", "reach_potential", "unfinished_step") and not self.cart_task:
                    raise ValueError(f"{kind} requires task.cart_task")
                if self.cart_task and kind in ("forward_progress", "target_distance"):
                    raise ValueError("cart_task uses bounded reach_improvement, not unrestricted progress rewards")
                self.rewards.append(term)
        self.effort_metrics = any(c["type"] in ("torque_l2", "power_l1") for c in self.rewards)

        self.potential_shaping = any(c["type"] == "reach_potential" for c in self.rewards)
        if self.potential_shaping:
            gamma = raw["training"]["discounting"]
            _finite(gamma, "training.discounting")
            if not 0 < gamma < 1:
                raise ValueError("reach_potential requires 0 < training.discounting < 1")
            if self.reward_clip is not None:
                raise ValueError("reach_potential requires reward_clip: null to preserve telescoping")
            if any(c["type"] == "reach_improvement" for c in self.rewards):
                raise ValueError("Do not mix reach_potential and reach_improvement")

    def manifest(self):
        frames = []
        for i in range(self.history_steps):
            age = self.history_steps - 1 - i if self.history_order == "oldest_first" else i
            frames.append({"age_in_policy_steps": age, "start": i * self.frame_size,
                           "stop": (i + 1) * self.frame_size})
        return {
            "observation_size": self.observation_size, "frame_size": self.frame_size,
            "components": deepcopy(self.layout), "history_frames": frames,
            "history_reset": self.history_reset, "clip_observations": self.clip,
            "observation_processing": "raw -> independent Gaussian noise in raw units -> component scale -> clip -> history -> PPO running normalization if enabled",
            "normalize_observations": bool(self.cfg.training.normalize_observations),
            "joint_action_order": self.joint_names,
            "robot": OmegaConf.to_container(self.cfg.robot, resolve=True) if "robot" in self.cfg else None,
            "cart": OmegaConf.to_container(self.cfg.cart, resolve=True) if "cart" in self.cfg else None,
            "control_mode": str(self.cfg.control.mode),
            "control": OmegaConf.to_container(self.cfg.control, resolve=True),
            "bars": OmegaConf.to_container(self.cfg.bars, resolve=True),
            "policy_dt": float(self.cfg.simulation.control_dt),
            "physics_dt": float(self.cfg.simulation.timestep),
            "rewards": [dict(deepcopy(c), formula=REWARDS[c["type"]][0]) for c in self.rewards],
            "reward_aggregation": "sum(weight * raw_term * (control_dt if scale_by_dt else 1)); cart_task suppresses positive-weight terms on failure except reach_potential terminal debit; optional total clip; non-finite physics overrides total",
            "shaping_discount": float(self.cfg.training.discounting) if self.potential_shaping else None,
            "task_rules": {k: v for k, v in OmegaConf.to_container(self.cfg.task, resolve=True).items() if k != "reward_components"},
            "available_observations": {k: {"size": v[0], "meaning": v[1], "parameters": v[2]} for k, v in OBSERVATIONS.items()},
            "available_reward_types": {k: {"formula": v[0], "parameters": v[1]} for k, v in REWARDS.items()},
        }

"""Device-resident, batched execution of MetaMachine's rigid-body probe task.

Uses the existing morphology/configuration path at startup and a small Warp
implementation of the supported task terms during rollout. Returns borrowed
Warp arrays; callers explicitly copy only when a host snapshot is wanted.
"""

import numpy as np
from omegaconf import OmegaConf

from .backends.state import BackendCapabilityError
from .env_newton import NewtonMetaMachine
from .env_sim import MetaMachine


class _ModelPreparation(MetaMachine):
    """Reuse morphology, gains and initialization without creating a log run."""

    def _setup_logging(self):
        self._log_dir = None


class NativeNewtonMetaMachine:
    def __init__(self, cfg):
        import warp as wp

        from .backends.newton import NewtonBackend
        from .components.newton_probe import ProbeParameters

        self.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        self._validate_config(self.cfg)
        self.num_envs = int(cfg.environment.get("num_envs", 1))
        self.dt = float(cfg.control.dt)
        prep_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        prep_cfg.environment.num_envs = 1
        reference = _ModelPreparation(prep_cfg)
        try:
            if reference.model.njnt == 0 or reference.model.jnt_type[0] != 0:
                raise BackendCapabilityError(
                    "Native probe requires a free root joint first"
                )
            reference._set_initial_state()
            self.physics = NewtonBackend(
                reference.model,
                mode="native",
                device=cfg.simulation.get("newton_device", "cuda:0"),
                num_worlds=self.num_envs,
            )
            self.physics.set_state(
                reference.data.qpos.copy(), reference.data.qvel.copy()
            )
            wp.copy(self.physics.initial_q, self.physics.state.joint_q)
            wp.copy(self.physics.initial_qd, self.physics.state.joint_qd)
            self.frame_skip = reference.frame_skip
            self.action_size = reference.model.nu
            if cfg.control.num_actions != self.action_size:
                raise BackendCapabilityError(
                    "native control.num_actions must match model actuators"
                )
            self.device = self.physics.device
            self._wp = wp

            def floats(value):
                return wp.array(
                    np.broadcast_to(
                        np.asarray(value, dtype=np.float32), (self.action_size,)
                    ).copy(),
                    dtype=wp.float32,
                    device=self.device,
                )

            self._control_params = [
                floats(cfg.control.action_scale),
                floats(
                    cfg.control.custom_limits
                    if cfg.control.custom_limits is not None
                    else cfg.control.symmetric_limit
                ),
                floats(reference.default_dof_pos),
                floats(reference.kps),
                floats(reference.kds),
            ]
            self._qids = wp.array(
                self.physics.actuator_q, dtype=wp.int32, device=self.device
            )
            self._vids = wp.array(
                self.physics.actuator_qd, dtype=wp.int32, device=self.device
            )
            spec = ProbeParameters()
            spec.num_actions = self.action_size
            spec.q_stride, spec.qd_stride = (
                self.physics.q_stride,
                self.physics.qd_stride,
            )
            spec.body_stride = self.physics.body_stride
            spec.torso = self.physics.body_map[reference.torso_body_id]
            spec.history = int(cfg.observation.include_history_steps)
            spec.gravity = wp.vec3(*cfg.observation.gravity_vec)
            spec.forward = wp.vec3(*cfg.observation.projected_forward_vec)
            spec.obs_clip = float(cfg.observation.clip_observations)
            term = cfg.task.termination_conditions
            spec.height = (
                float(term.height_threshold)
                if term.height_threshold is not None
                else -float("inf")
            )
            spec.max_steps = (
                int(term.max_episode_steps) if term.max_episode_steps is not None else 0
            )
            for reward in cfg.task.reward_components:
                if reward.type == "linear_velocity_tracking":
                    spec.linear_weight = float(reward.weight)
                    spec.linear_target = float(
                        reward.params.get("target_velocity", 1.0)
                    )
                    spec.linear_sigma = float(reward.params.get("tracking_sigma", 0.15))
                else:
                    spec.angular_weight = float(reward.weight)
                    spec.angular_target = float(
                        reward.params.get("target_angular_velocity", 0.0)
                    )
                    spec.angular_sigma = float(
                        reward.params.get("tracking_sigma", 0.15)
                    )
            self.spec = spec
            # Legacy torso cvel is measured at its root tree's mass center.
            weights = np.zeros(spec.body_stride, dtype=np.float32)
            model = reference.model
            root = model.body_rootid[reference.torso_body_id]
            for mid, nid in self.physics.body_map.items():
                if model.body_rootid[mid] == root:
                    weights[nid] = model.body_mass[mid]
            weights /= weights.sum()
            self._tree_weights = wp.array(weights, dtype=wp.float32, device=self.device)
        finally:
            reference.close()
        self.observation_size = (6 + 3 * self.action_size) * self.spec.history
        self._ctrl = wp.zeros(
            self.num_envs * self.action_size, dtype=wp.float32, device=self.device
        )
        self._last_action = wp.zeros(
            (self.num_envs, self.action_size), dtype=wp.float32, device=self.device
        )
        self._valid_actions = wp.ones(self.num_envs, dtype=wp.int32, device=self.device)
        self._no_worlds = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros(
            (self.num_envs, self.observation_size), dtype=wp.float32, device=self.device
        )
        self.reward = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.terminated = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.truncated = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.steps = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.reward_components = wp.zeros(
            (self.num_envs, 2), dtype=wp.float32, device=self.device
        )
        self.info = {
            "episode_step": self.steps,
            "reward_components": self.reward_components,
            "valid_actions": self._valid_actions,
            "valid_controls": self.physics.valid_controls,
        }
        self.reset()

    @staticmethod
    def _validate_config(cfg):
        """Compile only supported terms; never use a CPU fallback for others."""
        check = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        check.environment.num_envs = 1
        try:
            NewtonMetaMachine._validate_newton_config(check)
        except ValueError as exc:
            raise BackendCapabilityError(str(exc)) from exc

        def require(ok, message):
            if not ok:
                raise BackendCapabilityError("Newton native probe: " + message)

        require(
            isinstance(cfg.environment.num_envs, int) and cfg.environment.num_envs > 0,
            "num_envs must be positive",
        )
        require(
            not cfg.simulation.get("newton_requires_grad", False),
            "task history is inference-only; use backend step_into for Tape",
        )
        expected = [
            {"name": "projected_gravity"},
            {"name": "ang_vel_body"},
            {"name": "dof_pos", "transform": "cos"},
            {"name": "dof_vel"},
            {"name": "last_action"},
        ]
        require(
            OmegaConf.to_container(cfg.observation.components) == expected,
            "only the rigid_body_probe observation layout is implemented",
        )
        require(
            cfg.observation.include_history_steps > 0
            and cfg.observation.clip_observations > 0,
            "history and observation clip must be positive",
        )
        require(
            not cfg.observation.get("critic", {}).get("enabled", False),
            "critic-only observations are unsupported",
        )
        c, s, init = cfg.control, cfg.simulation, cfg.initialization
        require(
            c.control_mode == "position", "only position target control is implemented"
        )
        for key in (
            "filter",
            "melter",
            "remapping",
            "per_joint_control",
            "obs_masking",
            "position_reference",
            "phase_actuator_release",
        ):
            require(
                not c.get(key, {}).get("enabled", False),
                f"control.{key} is unsupported",
            )
        for key in (
            "target_rate_limit_rad_s",
            "action_remap_type",
            "knee_hip_coupling",
        ):
            require(not c.get(key), f"control.{key} is unsupported")
        require(
            not cfg.environment.get("frozen_joints"), "frozen joints are unsupported"
        )
        require(int(s.get("latency_scheme", -1)) == -1, "latency must be disabled")
        for key in (
            "tn_constraint",
            "random_latency_scheme",
            "noisy_actions",
            "noisy_observations",
            "broken_motors",
            "pd_update_each_substep",
        ):
            require(not s.get(key, False), f"simulation.{key} is unsupported")
        for key in (
            "noisy_init",
            "randomize_orientation",
            "fully_randomize_orientation",
            "randomize_ini_vel",
        ):
            require(
                not init.get(key, False),
                "native task resets are deterministic; disable " + key,
            )
        require(
            not cfg.randomization.get("init_joint_pos", {}).get("enabled", False),
            "native task resets are deterministic",
        )
        term = cfg.task.termination_conditions
        require(
            term.get("termination_strategy") is None
            and term.get("height_position_source") == "torso_world",
            "only torso height and time-limit termination are implemented",
        )
        for key in (
            "height_grace_seconds",
            "height_minimum_duration_seconds",
            "height_emergency_threshold",
            "max_yaw_change_radians",
            "max_sustained_tilt_degrees",
            "cross_track_limit",
        ):
            require(not term.get(key), f"termination {key} is unsupported")
        rewards = list(cfg.task.reward_components)
        require(
            sorted(r.type for r in rewards)
            == ["angular_velocity_tracking", "linear_velocity_tracking"],
            "only linear/angular velocity tracking rewards are implemented",
        )
        for r in rewards:
            target_key = (
                "target_velocity"
                if r.type == "linear_velocity_tracking"
                else "target_angular_velocity"
            )
            require(
                set(r.params) <= {target_key, "tracking_sigma"},
                "reward modifiers are unsupported",
            )
            require(
                isinstance(r.params.get(target_key, 0.0), (int, float)),
                "command-linked rewards are unsupported",
            )
            require(
                float(r.params.get("tracking_sigma", 0.15)) > 0,
                "reward sigma must be positive",
            )

    @property
    def state_view(self):
        return self.physics.state_view

    def _observe(self, reset_phase, mask):
        from .components.newton_probe import probe_observe

        state = self.state_view
        self._wp.launch(
            probe_observe,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                state.body_transform,
                state.body_velocity,
                state.body_com,
                self._tree_weights,
                self._qids,
                self._vids,
                self.spec,
                reset_phase,
                mask,
                self.physics.valid_controls,
                self._valid_actions,
                self._last_action,
                self.steps,
            ],
            outputs=[
                self.obs,
                self.reward,
                self.terminated,
                self.truncated,
                self.reward_components,
            ],
            device=self.device,
        )

    def step(self, actions):
        from .components.newton_probe import probe_controls

        wp = self._wp
        if (
            not isinstance(actions, wp.array)
            or actions.dtype != wp.float32
            or actions.shape != (self.num_envs, self.action_size)
            or actions.device != self.device
        ):
            raise ValueError(
                f"actions must be a float32 Warp array on {self.device}, shape {(self.num_envs, self.action_size)}"
            )
        self._valid_actions.fill_(1)
        state = self.state_view
        wp.launch(
            probe_controls,
            dim=(self.num_envs, self.action_size),
            inputs=[
                actions,
                state.joint_q,
                state.joint_qd,
                self._qids,
                self._vids,
                *self._control_params,
                self.spec,
            ],
            outputs=[self._last_action, self._ctrl, self._valid_actions],
            device=self.device,
        )
        self.physics.step(self._ctrl, self.frame_skip)
        self._observe(0, self._no_worlds)
        return self.obs, self.reward, self.terminated, self.truncated, self.info

    def reset(self, *, seed=None, mask=None, options=None):
        """Device-only deterministic reset, optionally selecting individual worlds."""
        if options:
            raise BackendCapabilityError("Native reset options are not implemented")
        selected = self.physics._all_worlds if mask is None else mask
        self.physics.reset_worlds(selected)
        self._observe(1, selected)
        return self.obs, self.info

    def snapshot(self):
        """Explicit synchronizing diagnostic operation, never called by step."""
        s = self.state_view
        return {
            "joint_q": s.joint_q.numpy(),
            "joint_qd": s.joint_qd.numpy(),
            "body_transform": s.body_transform.numpy(),
            "body_velocity": s.body_velocity.numpy(),
            "observation": self.obs.numpy(),
            "reward": self.reward.numpy(),
            "terminated": self.terminated.numpy(),
            "truncated": self.truncated.numpy(),
        }

    def render(self):
        raise BackendCapabilityError("Native rendering has not been implemented")

    def close(self):
        pass

"""Batched native Newton brachiation with Torch device observations and task state.

Public step outputs are borrowed buffers. Rollout collectors must copy them
before another step/reset. No automatic reset obscures terminal observations.
"""

import math
from contextlib import nullcontext

import mujoco
import numpy as np
import torch
import warp as wp

from . import brachiation_courses as courses
from . import brachiation_torch as task
from .backends.newton_brachiation import BrachiationNewtonBackend
from .brachiation_control import BrachiationController
from .brachiation_scene import build_scene, hanging_qpos
from .brachiation_spec import TaskSpec, resolve_config


class BrachiationNewton:
    backend = "newton"

    def __init__(self, cfg, source_path, *, num_envs=None, device="cuda:0", seed=None):
        wp.init()
        self.cfg = resolve_config(cfg)
        self.cfg.environment.mode = "newton"
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._warp_streams = {}
        self.num_envs = int(num_envs if num_envs is not None else cfg.training.num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.dt = float(cfg.simulation.control_dt)
        self.substeps = round(self.dt / float(cfg.simulation.timestep))
        if self.substeps < 1 or not np.isclose(
            self.substeps * float(cfg.simulation.timestep), self.dt
        ):
            raise ValueError("control_dt must be an integer multiple of timestep")
        self.xml_string = build_scene(self.cfg, source_path)
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.spec = TaskSpec(self.cfg, self.model)
        if not self.spec.cart_task:
            raise ValueError("Native brachiation currently requires task.cart_task")
        self.controller = BrachiationController(self.cfg, self.model)
        self.action_size, self.observation_size = (
            self.model.nu,
            self.spec.observation_size,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(cfg.training.seed if seed is None else seed))
        with self._stream():
            self.physics = BrachiationNewtonBackend(
                self.model,
                num_worlds=self.num_envs,
                device=str(self.device),
                contact_distance=cfg.task.contact_distance,
            )
            # Featherstone uses explicit compliant contacts. Refine only the
            # integration interval; hold each PD torque for the original 2 ms
            # feedback interval and preserve the 20 ms policy interval.
            self.solver_substeps = int(cfg.simulation.get("newton_substeps", 4))
            if self.solver_substeps < 1:
                raise ValueError("simulation.newton_substeps must be positive")
            self.physics.dt /= self.solver_substeps
            self.physics.scene_manifest.update(
                pd_feedback_dt=float(cfg.simulation.timestep),
                integration_dt=self.physics.dt,
                solver_substeps=self.solver_substeps,
            )
            self._prepare()
            self.reset()

    def _warp_stream(self):
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream not in self._warp_streams:
            self._warp_streams[stream.cuda_stream] = wp.stream_from_torch(stream)
        return self._warp_streams[stream.cuda_stream]

    def _stream(self):
        return (
            wp.ScopedStream(self._warp_stream(), sync_enter=False, sync_exit=False)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(
            np.asarray(value).copy(), dtype=dtype, device=self.device
        )

    def _prepare(self):
        m, p, n = self.model, self.physics, self.num_envs
        self.rows = torch.arange(n, device=self.device)
        self.qids = self._tensor(p.actuator_q, torch.long)
        self.vids = self._tensor(p.actuator_qd, torch.long)
        self.torso = p.body_map[m.body("torso").id]
        self.cart = p.body_map[m.body("cart").id]
        sites = [m.site(s + "_grip_site").id for s in ("left", "right")]
        self.hand_bodies = self._tensor(
            [p.body_map[int(m.site_bodyid[s])] for s in sites], torch.long
        )
        self.hand_offsets = self._tensor(m.site_pos[sites])
        p.prepare_torch_scene(hanging_qpos(m, self.cfg, [0, 0, 0]))
        self.body_com = p.local_com
        self.loads, self.touching, self.flags = [
            wp.to_torch(v) for v in (p.loads, p.touching, p.flags)
        ]
        self.control = torch.zeros((n, self.action_size), device=self.device)
        self._wp_control = wp.from_torch(self.control.flatten())
        self._mask = torch.ones(n, dtype=torch.int32, device=self.device)
        self.start_height = float(m.body("course_bar_0").pos[2])
        self.half_lengths = self._tensor(courses.bar_half_lengths(self.cfg))
        self.spatial = courses.is_spatial(self.cfg)
        self.bars = torch.zeros((n, p.bar_count, 3), device=self.device)
        self.axes = torch.zeros_like(self.bars)
        self.bar_quats = torch.zeros((n, p.bar_count, 4), device=self.device)
        if self.cfg.bars.get("layout") == "grid":
            centers, axes, _ = courses.grid_geometry(self.cfg)
            centers[:, 2] += self.start_height
            self._grid_centers, self._grid_axes = (
                self._tensor(centers),
                self._tensor(axes),
            )
        self.gravity = self._tensor([0, 0, -1])
        self.torso_half = self._tensor(m.geom("torso_geom").size)
        self.cradle_inner = self.torso_half[:2] + float(self.cfg.cart.cradle.clearance)
        self.deck_top = float(m.geom("cart_deck").pos[2] + m.geom("cart_deck").size[2])
        self.gains = {}
        if self.controller.mode == "pd":
            for name in (
                "kp",
                "kd",
                "default_joint_pos",
                "target_lower",
                "target_upper",
                "torque_limits",
                "gear",
            ):
                self.gains[name] = self._tensor(getattr(self.controller, name))
        self.scales = [
            (self._tensor(c["scale"]), self._tensor(c["noise_std"]))
            for c in self.spec.observations
        ]
        self.directions = {
            t["name"]: self._tensor(t["params"]["direction"])
            for t in self.spec.rewards
            if t["type"] == "forward_progress"
        }
        self.distance_metrics = {
            t["params"]["metric"]
            for t in self.spec.rewards
            if t["type"] == "target_distance"
        }
        self.history = torch.zeros(
            (n, self.spec.history_steps, self.spec.frame_size), device=self.device
        )
        self.obs = torch.zeros((n, self.observation_size), device=self.device)
        self.reward = torch.zeros(n, device=self.device)
        self.terminated = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros_like(self.terminated)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.info = {
            k: torch.zeros(n, dtype=torch.long, device=self.device)
            for k in (
                "target",
                "contact_steps",
                "last_transfer_hand",
                "no_hook_steps",
                "no_transfer_steps",
            )
        }
        self.info.update(
            course_complete=torch.zeros_like(self.terminated),
            last_action=torch.zeros_like(self.control),
            transfer_counts=torch.zeros((n, 2), dtype=torch.long, device=self.device),
            reach_best=torch.zeros_like(self.reward),
        )
        self.metrics = {}
        self._step_graph = self._reset_graph = None

    def _rand(self, shape):
        return torch.rand(shape, generator=self.generator, device=self.device)

    def _kinematics(self):
        q, qd, body, velocity = self.physics.tensor_state
        pos, quat = body[..., :3], body[..., 3:]
        # Newton velocities are at COM; MuJoCo free qvel is at body origin.
        linear = velocity[..., :3] - torch.linalg.cross(
            velocity[..., 3:], task.rotate(quat, self.body_com), dim=-1
        )
        hands = pos[:, self.hand_bodies] + task.rotate(
            quat[:, self.hand_bodies], self.hand_offsets
        )
        return {
            "q": q,
            "qd": qd,
            "pos": pos,
            "quat": quat,
            "linear": linear,
            "angular": velocity[..., 3:],
            "hands": hands,
        }

    def targets(self, action):
        g = self.gains
        return g["default_joint_pos"] + action * torch.where(
            action >= 0,
            g["target_upper"] - g["default_joint_pos"],
            g["default_joint_pos"] - g["target_lower"],
        )

    def _controls(self, action):
        if self.controller.mode == "torque":
            self.control.copy_(action * float(self.controller.action_scale))
        else:
            q, qd = self.physics.tensor_state[:2]
            g = self.gains
            torque = (
                g["kp"] * (self.targets(action) - q[:, self.qids])
                - g["kd"] * qd[:, self.vids]
            )
            self.control.copy_(
                torque.clamp(-g["torque_limits"], g["torque_limits"]) / g["gear"]
            )

    def _outside(self, k):
        c = self.cfg.task.cart_task
        local = task.inverse_rotate(
            k["quat"][:, self.cart], k["pos"][:, self.torso] - k["pos"][:, self.cart]
        )
        sideways = (
            local[:, :2].abs() > self.cradle_inner + float(c.containment_margin)
        ).any(-1)
        low = local[:, 2] < self.deck_top + 0.5 * self.torso_half[2]
        high = local[:, 2] > self.deck_top + self.torso_half[2] + float(
            c.max_torso_lift
        )
        down = task.inverse_rotate(k["quat"][:, self.cart], self.gravity)
        return (
            sideways | low | high | (-down[:, 2] < math.cos(float(c.cart_tilt_limit)))
        )

    def _potential(self, k, target, hand):
        return task.potential(
            k["hands"],
            self.bars,
            self.axes,
            self.half_lengths,
            target,
            hand,
            float(self.cfg.task.cart_task.reach_radius),
        )

    def _observation_frame(self, k):
        torso_q, torso_p = k["quat"][:, self.torso], k["pos"][:, self.torso]
        values = []
        for component, (scale, noise) in zip(
            self.spec.observations, self.scales, strict=False
        ):
            name, params = component["name"], component["params"]
            if name == "projected_gravity":
                value = task.inverse_rotate(torso_q, self.gravity)
            elif name == "lin_vel_body":
                value = task.inverse_rotate(torso_q, k["linear"][:, self.torso])
            elif name == "ang_vel_body":
                value = task.inverse_rotate(torso_q, k["angular"][:, self.torso])
            elif name == "dof_pos":
                value = k["q"][:, self.qids]
            elif name == "dof_vel":
                value = k["qd"][:, self.vids]
            elif name == "hand_positions":
                value = k["hands"] - torso_p[:, None]
                if params["frame"] == "body":
                    value = task.inverse_rotate(torso_q[:, None], value)
            elif name in ("target_bar_positions", "target_bar_axes"):
                ids = (
                    self.info["target"][:, None]
                    + torch.arange(params["count"], device=self.device)
                ).clamp_max(int(self.cfg.bars.count) - 1)
                value = (
                    (self.bars[self.rows[:, None], ids] - torso_p[:, None])
                    if name == "target_bar_positions"
                    else self.axes[self.rows[:, None], ids]
                )
                if params["frame"] == "body":
                    value = task.inverse_rotate(torso_q[:, None], value)
            elif name == "last_action":
                value = self.info["last_action"]
            elif name == "hook_contacts":
                value = self.touching.float()
            elif name == "cart_relative_position":
                value = task.inverse_rotate(torso_q, k["pos"][:, self.cart] - torso_p)
            elif name == "cart_relative_velocity":
                value = task.inverse_rotate(
                    torso_q, k["linear"][:, self.cart] - k["linear"][:, self.torso]
                )
            elif name == "cart_projected_gravity":
                value = task.inverse_rotate(k["quat"][:, self.cart], self.gravity)
            elif name == "torso_height":
                value = torso_p[:, 2:3]
            elif name == "pd_target":
                value = self.targets(self.info["last_action"])
            elif name == "tracking_error":
                value = self.targets(self.info["last_action"]) - k["q"][:, self.qids]
            else:
                raise ValueError("Unsupported observation: " + name)
            value = value.reshape(self.num_envs, -1)
            if np.any(np.asarray(component["noise_std"]) != 0):
                value = (
                    value
                    + torch.randn(
                        value.shape, generator=self.generator, device=self.device
                    )
                    * noise
                )
            values.append(value * scale)
        frame = torch.cat(values, -1)
        return (
            frame
            if self.spec.clip is None
            else frame.clamp(-self.spec.clip, self.spec.clip)
        )

    def _publish_obs(self):
        ordered = (
            self.history
            if self.spec.history_order == "oldest_first"
            else self.history.flip(1)
        )
        self.obs.copy_(ordered.reshape(self.num_envs, -1))

    @torch.no_grad()
    def enable_cuda_graph(self):
        """Capture fixed-shape step/reset; resets the task to its seeded start.

        An even number of substeps returns to the same ping-pong state address.
        Odd substep configurations remain supported in eager mode.
        """
        if self.device.type != "cuda" or (self.substeps * self.solver_substeps) % 2:
            raise ValueError(
                "CUDA graph execution requires CUDA and an even substep count"
            )
        if self._step_graph is not None:
            return
        self._graph_action = torch.zeros_like(self.control)
        self._graph_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._step_eager(self._graph_action)
                self._reset_eager(self._graph_mask)
        current.wait_stream(stream)
        step_graph = torch.cuda.CUDAGraph()
        step_graph.register_generator_state(self.generator)
        with torch.cuda.graph(step_graph):
            stream = self._warp_stream()
            wp.capture_begin(stream=stream, external=True)
            self._graph_result = self._step_eager(self._graph_action)
            self._warp_step_capture = wp.capture_end(stream=stream)
        reset_graph = torch.cuda.CUDAGraph()
        reset_graph.register_generator_state(self.generator)
        with torch.cuda.graph(reset_graph):
            stream = self._warp_stream()
            wp.capture_begin(stream=stream, external=True)
            self._reset_eager(self._graph_mask)
            self._warp_reset_capture = wp.capture_end(stream=stream)
        self._step_graph, self._reset_graph = step_graph, reset_graph
        self.generator.manual_seed(self.generator.initial_seed())
        self.reset()

    @torch.no_grad()
    def reset(self, mask=None):
        if self._reset_graph is None:
            return self._reset_eager(mask)
        if mask is None:
            self._graph_mask.fill_(True)
        else:
            if mask.shape != (self.num_envs,) or mask.device != self.device:
                raise ValueError(
                    "Reset mask must be a device tensor of shape (num_envs,)"
                )
            self._graph_mask.copy_(mask)
        self._reset_graph.replay()
        return self.obs

    @torch.no_grad()
    def _reset_eager(self, mask=None):
        with self._stream():
            if mask is None:
                self._mask.fill_(1)
            else:
                if mask.shape != (self.num_envs,) or mask.device != self.device:
                    raise ValueError(
                        "Reset mask must be a device tensor of shape (num_envs,)"
                    )
                self._mask.copy_(mask)
            selected = self._mask.bool()
            layout = self.cfg.bars.get("layout", "parallel")
            if layout == "grid":
                centers = self._grid_centers[None].expand(self.num_envs, -1, -1)
                axes = self._grid_axes[None].expand(self.num_envs, -1, -1)
            elif layout == "spatial":
                centers, axes = task.sample_spatial(
                    self.cfg, self.num_envs, self.start_height, self._rand
                )
            else:
                centers = task.sample_parallel(
                    self.cfg, self.num_envs, self.start_height, self._rand
                )
                axes = torch.zeros_like(centers)
                axes[..., 1] = 1
            self.bars.copy_(torch.where(selected[:, None, None], centers, self.bars))
            self.axes.copy_(torch.where(selected[:, None, None], axes, self.axes))
            # Shortest rotation from +Y, including grid rails along +X/+Z.
            w = ((1 + self.axes[..., 1]) / 2).sqrt().clamp_min(1e-8)
            self.bar_quats.copy_(
                torch.stack(
                    (
                        self.axes[..., 2] / (2 * w),
                        torch.zeros_like(w),
                        -self.axes[..., 0] / (2 * w),
                        w,
                    ),
                    -1,
                )
            )
            self.physics.reset_scene(
                self._mask, self.bars[:, 0], self.bars, self.bar_quats
            )
            for key, value in self.info.items():
                fill = (
                    1 if key == "target" else (-1 if key == "last_transfer_hand" else 0)
                )
                shape = (self.num_envs,) + (1,) * (value.ndim - 1)
                value.copy_(torch.where(selected.reshape(shape), fill, value))
            k = self._kinematics()
            self.info["reach_best"].copy_(
                torch.where(
                    selected,
                    self._potential(
                        k, self.info["target"], self.info["last_transfer_hand"]
                    ),
                    self.info["reach_best"],
                )
            )
            frame = self._observation_frame(k)
            history = frame[:, None].expand(-1, self.spec.history_steps, -1).clone()
            if self.spec.history_reset == "zeros":
                history[:, :-1] = 0
            self.history.copy_(
                torch.where(selected[:, None, None], history, self.history)
            )
            for value in (self.reward, self.terminated, self.truncated, self.steps):
                value.copy_(torch.where(selected, 0, value))
            self._publish_obs()
            return self.obs

    @torch.no_grad()
    def step(self, action):
        if self._step_graph is None:
            return self._step_eager(action)
        if (
            action.shape != self.control.shape
            or action.device != self.device
            or action.dtype != torch.float32
        ):
            raise ValueError(
                f"Actions must be float32 {tuple(self.control.shape)} on {self.device}"
            )
        self._graph_action.copy_(action)
        self._step_graph.replay()
        self.metrics = self._graph_result[4]
        return self._graph_result

    @torch.no_grad()
    def _step_eager(self, action):
        if (
            action.shape != self.control.shape
            or action.device != self.device
            or action.dtype != torch.float32
        ):
            raise ValueError(
                f"Actions must be float32 {tuple(self.control.shape)} on {self.device}"
            )
        with self._stream():
            valid_action = torch.isfinite(action).all(-1)
            action = torch.nan_to_num(action).clamp(-1, 1)
            before = self._kinematics()
            # Physics ping-pongs its input storage: retain pre-step quantities.
            before_pos = before["pos"][:, self.torso].clone()
            old_potential = torch.where(
                self.info["course_complete"],
                0.0,
                self._potential(
                    before, self.info["target"], self.info["last_transfer_hand"]
                ),
            )
            old_distance = {
                metric: task.distances(
                    before["hands"],
                    self.bars,
                    self.axes,
                    self.half_lengths,
                    self.info["target"],
                    metric,
                ).amin(-1)
                for metric in self.distance_metrics
            }
            torque_sum, power_sum = (
                torch.zeros_like(self.reward),
                torch.zeros_like(self.reward),
            )
            forbidden = torch.zeros(
                (self.num_envs, 4), dtype=torch.bool, device=self.device
            )
            overflow = torch.zeros_like(self.terminated)
            for _ in range(self.substeps):
                self._controls(action)
                velocity = self.physics.tensor_state[1][:, self.vids].clone()
                self.physics.step(self._wp_control, self.solver_substeps)
                torque = self.physics.motor_torque
                torque_sum += torque.square().sum(-1)
                power_sum += (torque * velocity).abs().sum(-1)
                forbidden[:, :3] |= self.flags[:, :3].bool()
                forbidden[:, 3] |= self._outside(self._kinematics())
                overflow |= self.physics.contact_capacity_reached()
            k = self._kinematics()
            pos = k["pos"][:, self.torso]
            fall = (pos[:, 2] < float(self.cfg.task.fall_height)) | (
                pos[:, 1].abs() > float(self.cfg.task.lateral_limit)
            )
            finite = (
                torch.isfinite(k["q"]).all(-1)
                & torch.isfinite(k["qd"]).all(-1)
                & valid_action
                & ~overflow
            )
            target = self.info["target"]
            distance = task.distances(
                k["hands"], self.bars, self.axes, self.half_lengths, target
            )
            reached, failure, guard, stalled = task.advance(
                self.cfg,
                self.info,
                distance,
                self.loads,
                self.touching.bool(),
                self.flags[:, 3].bool(),
                forbidden.any(-1) | fall | ~finite,
                self.spatial,
            )
            fall |= failure
            current = self._potential(k, target, self.info["last_transfer_hand"])
            best = torch.maximum(self.info["reach_best"], current)
            improvement = torch.where(
                fall | self.info["course_complete"], 0.0, best - self.info["reach_best"]
            )
            success = reached & (target == int(self.cfg.bars.count) - 1)
            next_target = (target + reached.long()).clamp_max(
                int(self.cfg.bars.count) - 1
            )
            complete = self.info["course_complete"] | success
            next_potential = self._potential(
                k, next_target, guard["last_transfer_hand"]
            )
            guard["reach_best"] = torch.where(reached, next_potential, best)
            terminal = (
                (fall & bool(self.cfg.task.terminate_on_fall))
                | (success & bool(self.cfg.task.terminate_on_success))
                | ~finite
            )
            context = {
                "terminal": terminal,
                "reached": reached,
                "fall": fall,
                "finite": finite,
                "action": action,
                "last_action": self.info["last_action"],
                "mean_torque_l2": torque_sum / self.substeps,
                "mean_power_l1": power_sum / self.substeps,
                "joint_velocity": k["qd"][:, self.vids],
                "reach_improvement": improvement,
                "potential_before": old_potential,
                "potential_after": torch.where(complete, 0.0, next_potential),
                "displacement": pos - before_pos,
                "directions": self.directions,
                "distance_improvement": {
                    metric: old_distance[metric]
                    - task.distances(
                        k["hands"],
                        self.bars,
                        self.axes,
                        self.half_lengths,
                        target,
                        metric,
                    ).amin(-1)
                    for metric in self.distance_metrics
                },
            }
            reward, self.metrics = task.reward_terms(self.cfg, self.spec, context)
            self.reward.copy_(reward)
            self.terminated.copy_(terminal)
            self.steps.add_(1)
            self.truncated.copy_(
                (self.steps >= int(self.cfg.environment.episode_length)) & ~terminal
            )
            for key, value in dict(
                guard, target=next_target, course_complete=complete, last_action=action
            ).items():
                self.info[key].copy_(value)
            frame = self._observation_frame(k)
            self.history.copy_(torch.cat((self.history[:, 1:], frame[:, None]), 1))
            self._publish_obs()
            self.metrics.update(
                progress=torch.where(finite, pos[:, 0] - before_pos[:, 0], 0.0),
                bars_reached=reached.float(),
                fall=fall.float(),
                success=success.float(),
                robot_floor=forbidden[:, 0].float(),
                limb_cart=forbidden[:, 1].float(),
                nonhook_bar=forbidden[:, 2].float(),
                cart_escape=forbidden[:, 3].float(),
                stalled=stalled.float(),
                cart_seated=self.flags[:, 3].float(),
                no_hook_failure=(
                    guard["no_hook_steps"]
                    >= int(self.cfg.task.cart_task.max_no_hook_steps)
                ).float(),
                contact_overflow=overflow.float(),
                nonfinite_physics=(
                    ~(torch.isfinite(k["q"]).all(-1) & torch.isfinite(k["qd"]).all(-1))
                ).float(),
            )
            return self.obs, self.reward, self.terminated, self.truncated, self.metrics

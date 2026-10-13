"""Shared Featherstone dynamics with optional legacy synchronization.

Native step/step_into accept device controls and perform no host reads. Model
compilation and explicit host reset/snapshot operations are outside the rollout.
"""

from dataclasses import dataclass

import numpy as np

from .state import BackendCapabilityError, DeviceStateView


@dataclass
class NewtonStepBuffers:
    """Allocate once per retained timestep before a differentiable rollout."""

    state: object
    control: object
    contacts: object
    actuator_force: object
    actuator_torque: object
    valid_controls: object
    solver: object
    collision_pipeline: object


class NewtonBackend:
    name = "newton"

    def __init__(
        self,
        model,
        data=None,
        *,
        device="cpu",
        mode="compat",
        num_worlds=1,
        requires_grad=False,
        translation=None,
    ):
        try:
            import newton
            import warp as wp
        except ImportError as exc:
            raise ImportError(
                "Newton mode requires pip install 'metamachine[newton]'"
            ) from exc
        from .newton_model import NewtonModelTranslation

        if mode not in ("compat", "native"):
            raise ValueError("Newton mode must be 'compat' or 'native'")
        if not isinstance(num_worlds, int) or num_worlds < 1:
            raise ValueError("num_worlds must be a positive integer")
        if mode == "compat" and (num_worlds != 1 or requires_grad):
            raise BackendCapabilityError("compat supports one world without gradients")
        self.model, self.mode = model, mode
        self.num_worlds, self.requires_grad = num_worlds, requires_grad
        self._newton, self._wp = newton, wp
        self.translation = translation if translation is not None else NewtonModelTranslation(model)
        template = self.translation.builder
        self.body_map = self.translation.body_map
        self.joint_map = self.translation.joint_map
        self._qstarts, self._vstarts = (
            self.translation._qstarts,
            self.translation._vstarts,
        )
        self.q_stride = template.joint_coord_count
        self.qd_stride = template.joint_dof_count
        self.body_stride = template.body_count
        builder = template
        if num_worlds > 1:
            builder = newton.ModelBuilder()
            # Worlds share coordinates but cannot collide with each other.
            builder.replicate(template, world_count=num_worlds)
        self.newton_model = builder.finalize(device=device, requires_grad=requires_grad)
        self.device = self.newton_model.device
        self.dt = float(model.opt.timestep)
        self.num_motors = model.nu
        self.newton_model.joint_target_ke.zero_()
        self.newton_model.joint_target_kd.zero_()
        self.solver = newton.solvers.SolverFeatherstone(
            self.newton_model, angular_damping=0.0
        )
        self.collision_pipeline = newton.CollisionPipeline(self.newton_model)
        self._build_actuator_map()
        self._buffers = [self.allocate_step(), self.allocate_step()]
        self._state_index = 0
        self.control = self._buffers[0].control
        self.contacts = self._buffers[0].contacts
        self.actuator_force = self._buffers[0].actuator_force
        self.actuator_torque = self._buffers[0].actuator_torque
        self.valid_controls = self._buffers[0].valid_controls
        self._views = [self.view(b.state) for b in self._buffers]
        self._all_worlds = wp.ones(num_worlds, dtype=wp.int32, device=self.device)
        q, v = self.translation._to_newton(model.qpos0, np.zeros(model.nv))
        self.initial_q = wp.array(
            np.tile(q, num_worlds), dtype=wp.float32, device=self.device
        )
        self.initial_qd = wp.array(
            np.tile(v, num_worlds), dtype=wp.float32, device=self.device
        )
        self._mirror = None
        self._data = None
        if mode == "compat":
            import mujoco

            from .newton_compat import MuJoCoStateMirror

            self._data = data if data is not None else mujoco.MjData(model)
            self._mirror = MuJoCoStateMirror(self.translation, self._data)
            self._host_ctrl = wp.zeros(
                self.num_motors, dtype=wp.float32, device=self.device
            )
        self.reset()

    @property
    def data(self):
        if self.mode != "compat":
            raise BackendCapabilityError(
                "Native mode has no MjData mirror; use state_view or an explicit snapshot"
            )
        return self._data

    @property
    def state(self):
        return self._buffers[self._state_index].state

    @property
    def state_view(self):
        return self._views[self._state_index]

    def view(self, state):
        return DeviceStateView(
            state.joint_q,
            state.joint_qd,
            state.body_q,
            state.body_qd,
            self.newton_model.body_com,
            self.num_worlds,
            self.q_stride,
            self.qd_stride,
            self.body_stride,
        )

    def allocate_step(self):
        """Call before capture/Tape. Distinct buffers retain the gradient graph."""
        wp = self._wp
        count = self.num_worlds * self.num_motors
        kwargs = {
            "dtype": wp.float32,
            "device": self.device,
            "requires_grad": self.requires_grad,
        }
        # Featherstone keeps J/M/H/L on the solver, not just on output State.
        # A retained Tape timestep therefore owns solver and collision scratch
        # as well as state/control buffers. All are allocated before recording.
        solver = self.solver
        pipeline = self.collision_pipeline
        if self.requires_grad:
            solver = self._newton.solvers.SolverFeatherstone(
                self.newton_model, angular_damping=0.0
            )
            pipeline = self._newton.CollisionPipeline(self.newton_model)
        return NewtonStepBuffers(
            self.newton_model.state(),
            self.newton_model.control(),
            pipeline.contacts(),
            wp.zeros(count, **kwargs),
            wp.zeros(count, **kwargs),
            wp.ones(self.num_worlds, dtype=wp.int32, device=self.device),
            solver,
            pipeline,
        )

    def _build_actuator_map(self):
        wp, m = self._wp, self.model
        ids, offsets = [], [0]
        self.actuator_q = np.array(
            [self._qstarts[self.joint_map[j]] for j in m.actuator_trnid[:, 0]]
        )
        dofs = np.array(
            [self._vstarts[self.joint_map[j]] for j in m.actuator_trnid[:, 0]]
        )
        self.actuator_qd = dofs
        for dof in range(self.qd_stride):
            ids.extend(np.flatnonzero(dofs == dof).tolist())
            offsets.append(len(ids))
        self._offsets = wp.array(offsets, dtype=wp.int32, device=self.device)
        self._motor_ids = wp.array(ids, dtype=wp.int32, device=self.device)
        arrays = [
            np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 0], -np.inf),
            np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 1], np.inf),
            m.actuator_gainprm[:, 0],
            np.where(m.actuator_forcelimited, m.actuator_forcerange[:, 0], -np.inf),
            np.where(m.actuator_forcelimited, m.actuator_forcerange[:, 1], np.inf),
            m.actuator_gear[:, 0],
        ]
        self._motor_params = [
            wp.array(a, dtype=wp.float32, device=self.device) for a in arrays
        ]

    def _check_controls(self, ctrl):
        wp = self._wp
        if (
            not isinstance(ctrl, wp.array)
            or ctrl.dtype != wp.float32
            or ctrl.shape != (self.num_worlds * self.num_motors,)
            or ctrl.device != self.device
        ):
            raise ValueError(
                f"Native controls must be a flat float32 Warp array on {self.device}, "
                f"shape {(self.num_worlds * self.num_motors,)}"
            )

    def _map_controls(self, ctrl, buffers):
        from .newton_kernels import map_motor_controls

        self._check_controls(ctrl)  # Metadata only, never scans device contents.
        buffers.valid_controls.fill_(1)
        self._wp.launch(
            map_motor_controls,
            dim=(self.num_worlds, self.qd_stride),
            inputs=[
                ctrl,
                self._offsets,
                self._motor_ids,
                *self._motor_params,
                self.num_motors,
                self.qd_stride,
            ],
            outputs=[
                buffers.control.joint_f,
                buffers.actuator_force,
                buffers.actuator_torque,
                buffers.valid_controls,
            ],
            device=self.device,
        )

    def _integrate(
        self, state_in, state_out, control, contacts, solver=None, pipeline=None
    ):
        state_in.clear_forces()
        (pipeline or self.collision_pipeline).collide(state_in, contacts)
        (solver or self.solver).step(state_in, state_out, control, contacts, self.dt)

    def step_into(self, state_in, ctrl, buffers):
        """One functional frame with caller-owned output/control/contact storage.

        For Tape, preallocate a distinct buffers object for every timestep and
        leave input states/controls alive until backward. No ping-pong reuse.
        """
        if state_in is buffers.state:
            raise ValueError("step_into requires distinct input/output states")
        self._map_controls(ctrl, buffers)
        self._integrate(
            state_in,
            buffers.state,
            buffers.control,
            buffers.contacts,
            buffers.solver,
            buffers.collision_pipeline,
        )
        return buffers.state

    def step(self, ctrl, n_frames=1):
        """Reusable inference buffers. Native mode performs only device work."""
        if self.requires_grad:
            raise BackendCapabilityError(
                "Use step_into with preallocated per-timestep buffers for Tape"
            )
        if not isinstance(n_frames, (int, np.integer)) or n_frames < 0:
            raise ValueError("n_frames must be a nonnegative integer")
        if self.mode == "compat":
            ctrl = self._prepare_compat_controls(ctrl)
        self._map_controls(ctrl, self._buffers[0])
        for _ in range(n_frames):
            next_index = 1 - self._state_index
            self._integrate(
                self.state, self._buffers[next_index].state, self.control, self.contacts
            )
            self._state_index = next_index
        if self.mode == "compat":
            self._data.time += n_frames * self.dt
            self._sync_view()
        return self.state_view

    def _prepare_compat_controls(self, ctrl):
        ctrl = np.asarray(ctrl, dtype=float)
        if ctrl.shape != (self.num_motors,) or not np.isfinite(ctrl).all():
            raise ValueError(f"Expected finite control of shape {(self.num_motors,)}")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError(
                "Newton initial support does not include external applied forces"
            )
        self.data.ctrl[:] = ctrl
        self._host_ctrl.assign(ctrl.astype(np.float32))
        return self._host_ctrl

    def _sync_view(self):
        if self._mirror is None:
            raise BackendCapabilityError(
                "Native mode cannot synchronize a compatibility mirror"
            )
        self._mirror.sync(self.state, self.actuator_force, self.control.joint_f)

    def set_state(self, qpos, qvel):
        """Explicit host-side state upload; outside the native rollout loop."""
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,):
            raise ValueError("State shape does not match morphology")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("State must be finite")
        q, v = self.translation._to_newton(qpos, qvel)
        for buffers in self._buffers:
            buffers.state.joint_q.assign(np.tile(q, self.num_worlds))
            buffers.state.joint_qd.assign(np.tile(v, self.num_worlds))
            buffers.state.clear_forces()
            self._newton.eval_fk(
                self.newton_model,
                buffers.state.joint_q,
                buffers.state.joint_qd,
                buffers.state,
            )
        if self.mode == "compat":
            self._sync_view()

    def reset(self, qpos=None, qvel=None):
        if (qpos is None) != (qvel is None):
            raise ValueError("Provide both qpos and qvel, or neither")
        if self.mode == "compat":
            import mujoco

            mujoco.mj_resetData(self.model, self.data)
        self.reset_worlds(self._all_worlds)
        if qpos is not None:
            self.set_state(qpos, qvel)
        elif self.mode == "compat":
            self._sync_view()
        return self.state_view

    def reset_worlds(self, mask):
        """Selective device reset to construction pose; no host mask inspection."""
        from .newton_kernels import (
            reset_coordinates,
            reset_valid_controls,
            reset_world_values,
        )

        wp = self._wp
        if (
            not isinstance(mask, wp.array)
            or mask.dtype != wp.int32
            or mask.shape != (self.num_worlds,)
            or mask.device != self.device
        ):
            raise ValueError(
                "reset mask must be a device int32 array of shape (num_worlds,)"
            )
        for buffers in self._buffers:
            for initial, out, stride in [
                (self.initial_q, buffers.state.joint_q, self.q_stride),
                (self.initial_qd, buffers.state.joint_qd, self.qd_stride),
            ]:
                wp.launch(
                    reset_coordinates,
                    dim=out.size,
                    inputs=[initial, mask, stride],
                    outputs=[out],
                    device=self.device,
                )
            buffers.state.clear_forces()
            self._newton.eval_fk(
                self.newton_model,
                buffers.state.joint_q,
                buffers.state.joint_qd,
                buffers.state,
            )
            for out, stride in [
                (buffers.control.joint_f, self.qd_stride),
                (buffers.actuator_force, self.num_motors),
                (buffers.actuator_torque, self.num_motors),
            ]:
                if out.size:
                    wp.launch(
                        reset_world_values,
                        dim=out.size,
                        inputs=[mask, stride],
                        outputs=[out],
                        device=self.device,
                    )
            wp.launch(
                reset_valid_controls,
                dim=self.num_worlds,
                inputs=[mask],
                outputs=[buffers.valid_controls],
                device=self.device,
            )

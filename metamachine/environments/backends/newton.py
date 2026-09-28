"""Newton Featherstone dynamics with a MuJoCo-compatible task-state view.

MuJoCo compiles the existing morphology and evaluates *kinematics* only.
Neither mj_step nor mj_forward is used by this adapter. Contact forces and
accelerometer readings are deliberately not fabricated in the legacy view.
"""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _rotation(wxyz):
    return Rotation.from_quat(np.roll(wxyz, -1)).as_matrix()


class NewtonBackend:
    name = "newton"

    def __init__(self, model, data, *, device="cpu"):
        try:
            import newton
            import warp as wp
        except ImportError as exc:
            raise ImportError(
                "Newton mode requires pip install 'metamachine[newton]'"
            ) from exc

        self.model, self.data = model, data
        self._newton = newton
        self._validate_model()

        # Export the already compiled morphology: defaults, density-derived
        # masses and principal inertias must not be independently reinterpreted.
        with tempfile.TemporaryDirectory(prefix="metamachine-newton-") as tmp:
            path = Path(tmp) / "compiled.xml"
            mujoco.mj_saveLastXML(str(path), model)
            builder = newton.ModelBuilder()
            builder.add_mjcf(
                str(path),
                collapse_fixed_joints=False,
                ignore_inertial_definitions=False,
                parse_sites=False,
            )

        self.body_map = self._name_map(builder.body_label, model.nbody, "body", start=1)
        self.joint_map = self._name_map(builder.joint_label, model.njnt, "joint")
        # Copy compiled inertial properties at full precision (XML export is
        # rounded), and rotate principal inertia into each body's local frame.
        for mid, nid in self.body_map.items():
            mass = float(model.body_mass[mid])
            rot = _rotation(model.body_iquat[mid])
            inertia = rot @ np.diag(model.body_inertia[mid]) @ rot.T
            builder.body_mass[nid] = mass
            builder.body_inv_mass[nid] = 1.0 / mass if mass else 0.0
            builder.body_com[nid] = wp.vec3(*model.body_ipos[mid])
            builder.body_inertia[nid] = wp.mat33(inertia)
            builder.body_inv_inertia[nid] = wp.mat33(
                np.linalg.inv(inertia) if mass else np.zeros((3, 3))
            )

        self.newton_model = builder.finalize(device=device)
        self._qstarts = self.newton_model.joint_q_start.numpy()
        self._vstarts = self.newton_model.joint_qd_start.numpy()
        # Motor efforts are supplied explicitly below; no implicit servo or
        # extra global angular damping is allowed to change task controls.
        self.newton_model.joint_target_ke.zero_()
        self.newton_model.joint_target_kd.zero_()
        self.solver = newton.solvers.SolverFeatherstone(
            self.newton_model, angular_damping=0.0
        )
        self.control = self.newton_model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.newton_model)
        self.contacts = self.collision_pipeline.contacts()
        self.reset(model.qpos0, np.zeros(model.nv))

    def _name_map(self, labels, count, kind, start=0):
        short = [label.rsplit("/", 1)[-1] for label in labels]
        result = {}
        for i in range(start, count):
            name = getattr(self.model, kind)(i).name
            if not name or short.count(name) != 1:
                raise ValueError(
                    f"Newton requires unique named {kind}s; cannot map {name!r}"
                )
            result[i] = short.index(name)
        return result

    def _validate_model(self):
        m = self.model

        def require(ok, message):
            if not ok:
                raise ValueError("Newton initial support: " + message)

        require(
            m.neq == m.ntendon == m.nmocap == m.nflex == m.nplugin == 0,
            "equality constraints, tendons, mocap, flex bodies and plugins are unsupported",
        )
        require(np.all(m.body_gravcomp == 0), "gravity compensation is unsupported")
        require(m.nsensor == 0, "sensors are unsupported; use state observations")
        require(
            m.nmesh == m.nhfield == 0, "only primitive collision geometry is supported"
        )
        require(
            np.all(np.isin(m.geom_type, [0, 2, 3, 4, 5, 6])),
            "unsupported collision geometry",
        )
        require(np.all(m.body_jntnum <= 1), "use at most one joint per body")
        require(
            np.all(np.isin(m.jnt_type, [0, 2, 3])),
            "only free, slide and hinge joints are supported",
        )
        require(
            np.all(m.jnt_stiffness == 0) and np.all(m.dof_frictionloss == 0),
            "joint springs and frictionloss are unsupported",
        )
        require(
            np.all(m.jnt_actfrclimited == 0),
            "joint actuator force limits are unsupported",
        )
        for j, kind in enumerate(m.jnt_type):
            if kind != mujoco.mjtJoint.mjJNT_FREE:
                require(
                    m.qpos0[m.jnt_qposadr[j]] == 0,
                    "nonzero joint reference positions are unsupported",
                )
            else:
                require(
                    m.body_parentid[m.jnt_bodyid[j]] == 0,
                    "free joints must connect to world",
                )
                v = m.jnt_dofadr[j]
                require(
                    np.all(m.dof_damping[v : v + 6] == 0)
                    and np.all(m.dof_armature[v : v + 6] == 0),
                    "free-joint damping and armature are unsupported",
                )
        require(
            np.all(m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            and np.all(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
            and np.all(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
            and np.all(m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE),
            "only stateless joint motors with fixed gain and no bias are supported",
        )
        require(
            np.all(m.actuator_gear[:, 1:] == 0),
            "only scalar motor gear ratios are supported",
        )
        require(
            all(m.jnt_type[j] in (2, 3) for j in m.actuator_trnid[:, 0]),
            "motors must actuate slide or hinge joints",
        )
        require(
            m.opt.density == 0 and m.opt.viscosity == 0 and np.all(m.opt.wind == 0),
            "fluid forces are unsupported",
        )
        require(
            m.opt.disableflags == 0 and m.opt.enableflags == 0,
            "custom MuJoCo option flags are unsupported",
        )

    def _to_newton(self, qpos, qvel):
        q = self.newton_model.joint_q.numpy().copy()
        v = np.zeros(self.newton_model.joint_dof_count, dtype=np.float32)
        for mid, nid in self.joint_map.items():
            mq, mv = self.model.jnt_qposadr[mid], self.model.jnt_dofadr[mid]
            nq, nv = self._qstarts[nid], self._vstarts[nid]
            if self.model.jnt_type[mid] == mujoco.mjtJoint.mjJNT_FREE:
                q[nq : nq + 3] = qpos[mq : mq + 3]
                quat = np.asarray(qpos[mq + 3 : mq + 7], dtype=float)
                norm = np.linalg.norm(quat)
                if norm < 1e-12:
                    raise ValueError("Free-joint quaternion must be nonzero")
                q[nq + 3 : nq + 7] = np.roll(quat / norm, -1)
                rot = _rotation(qpos[mq + 3 : mq + 7])
                omega = rot @ qvel[mv + 3 : mv + 6]
                offset = rot @ self.model.body_ipos[self.model.jnt_bodyid[mid]]
                # Newton 1.6: free qd is world COM linear + world angular.
                # MuJoCo: world body-origin linear + body-local angular.
                v[nv : nv + 3] = qvel[mv : mv + 3] + np.cross(omega, offset)
                v[nv + 3 : nv + 6] = omega
            else:
                q[nq], v[nv] = qpos[mq], qvel[mv]
        return q, v

    def set_state(self, qpos, qvel):
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,):
            raise ValueError("State shape does not match morphology")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("State must be finite")
        q, v = self._to_newton(qpos, qvel)
        for state in (self.state, self._next_state):
            state.joint_q.assign(q)
            state.joint_qd.assign(v)
            state.clear_forces()
            self._newton.eval_fk(
                self.newton_model, state.joint_q, state.joint_qd, state
            )
        self._sync_view()

    def reset(self, qpos, qvel):
        # Both ping-pong buffers and solver caches belong to the episode.
        self.state = self.newton_model.state()
        self._next_state = self.newton_model.state()
        self.control.joint_f.zero_()
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(qpos, qvel)

    def _sync_view(self):
        q, v = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
        for mid, nid in self.joint_map.items():
            mq, mv = self.model.jnt_qposadr[mid], self.model.jnt_dofadr[mid]
            nq, nv = self._qstarts[nid], self._vstarts[nid]
            if self.model.jnt_type[mid] == mujoco.mjtJoint.mjJNT_FREE:
                self.data.qpos[mq : mq + 3] = q[nq : nq + 3]
                quat = np.roll(q[nq + 3 : nq + 7], 1).astype(float)
                quat /= np.linalg.norm(quat)
                self.data.qpos[mq + 3 : mq + 7] = quat
                rot = _rotation(quat)
                omega = v[nv + 3 : nv + 6]
                offset = rot @ self.model.body_ipos[self.model.jnt_bodyid[mid]]
                self.data.qvel[mv : mv + 3] = v[nv : nv + 3] - np.cross(omega, offset)
                self.data.qvel[mv + 3 : mv + 6] = rot.T @ omega
            else:
                self.data.qpos[mq], self.data.qvel[mv] = q[nq], v[nv]
        # Kinematics only: never let MuJoCo solve Newton contacts or dynamics.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)

    def step(self, ctrl, n_frames):
        ctrl = np.asarray(ctrl, dtype=float)
        if ctrl.shape != (self.model.nu,) or not np.isfinite(ctrl).all():
            raise ValueError(f"Expected finite control of shape {(self.model.nu,)}")
        if not isinstance(n_frames, (int, np.integer)) or n_frames < 0:
            raise ValueError("n_frames must be a nonnegative integer")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError(
                "Newton initial support does not include external applied forces"
            )
        self.data.ctrl[:] = ctrl
        limited = np.where(
            self.model.actuator_ctrllimited,
            np.clip(ctrl, *self.model.actuator_ctrlrange.T),
            ctrl,
        )
        force = limited * self.model.actuator_gainprm[:, 0]
        force = np.where(
            self.model.actuator_forcelimited,
            np.clip(force, *self.model.actuator_forcerange.T),
            force,
        )
        torque = force * self.model.actuator_gear[:, 0]
        joint_f = np.zeros(self.newton_model.joint_dof_count, dtype=np.float32)
        self.data.qfrc_actuator[:] = 0
        self.data.actuator_force[:] = force
        for i, mid in enumerate(self.model.actuator_trnid[:, 0]):
            joint_f[self._vstarts[self.joint_map[mid]]] += torque[i]
            self.data.qfrc_actuator[self.model.jnt_dofadr[mid]] += torque[i]
        self.control.joint_f.assign(joint_f)
        for _ in range(n_frames):
            self.state.clear_forces()
            self.collision_pipeline.collide(self.state, self.contacts)
            self.solver.step(
                self.state,
                self._next_state,
                self.control,
                self.contacts,
                self.model.opt.timestep,
            )
            self.state, self._next_state = self._next_state, self.state
            self.data.time += self.model.opt.timestep
        self._sync_view()

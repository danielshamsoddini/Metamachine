"""Explicit host synchronization for legacy tasks, never used by native steps."""

import mujoco
import numpy as np

from .newton_model import _rotation


class MuJoCoStateMirror:
    def __init__(self, translation, data):
        self.model, self.data = translation.model, data
        self.joint_map = translation.joint_map
        self._qstarts, self._vstarts = translation._qstarts, translation._vstarts

    def sync(self, state, actuator_force, joint_force):
        q, v = state.joint_q.numpy(), state.joint_qd.numpy()
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

        self.data.actuator_force[:] = actuator_force.numpy()
        forces = joint_force.numpy()
        for mid, nid in self.joint_map.items():
            mv = self.model.jnt_dofadr[mid]
            nv = self._vstarts[nid]
            if self.model.jnt_type[mid] != mujoco.mjtJoint.mjJNT_FREE:
                self.data.qfrc_actuator[mv] = forces[nv]

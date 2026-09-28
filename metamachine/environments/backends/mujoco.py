"""Original CPU MuJoCo integration, including post-constraint quantities."""

import mujoco
import numpy as np


class MuJoCoBackend:
    name = "mujoco"

    def __init__(self, model, data):
        self.model, self.data = model, data

    def step(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = np.asarray(qpos)
        self.data.qvel[:] = np.asarray(qvel)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, qpos, qvel):
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(qpos, qvel)

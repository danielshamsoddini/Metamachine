"""Opt-in symmetric upright pose placed at first foot contact, without a drop."""
import numpy as np
import mujoco


def symmetric_ground_start(env, qpos, qvel, cfg):
    if env.num_joint != 8:
        raise ValueError('symmetric ground start requires eight hip/ankle joints')
    bounds=np.asarray(cfg.get('ankle_range',[.8,1.2]),dtype=float)
    if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:
        raise ValueError('invalid ground-start ankle range')
    qpos=qpos.copy();qvel=np.zeros_like(qvel)
    joints=np.asarray(env.joint_idx,dtype=int)
    pose=np.array([0.,float(env.np_random.uniform(*bounds))]*4)
    hip_range=float(cfg.get('hip_randomization_rad',0.))
    if not np.isfinite(hip_range) or hip_range<0:
        raise ValueError('hip randomization must be finite and nonnegative')
    if hip_range>0:
        pose[::2]=env.np_random.uniform(-hip_range,hip_range,4)
    if np.any(pose<env.model.jnt_range[joints,0]) or np.any(pose>env.model.jnt_range[joints,1]):
        raise ValueError('ground-start pose exceeds joint limits')
    qpos[env.model.jnt_qposadr[joints]]=pose
    data=mujoco.MjData(env.model);data.qpos[:]=qpos;mujoco.mj_forward(env.model,data)
    ids=np.array([env.model.geom(f'{leg}_ankle_geom').id for leg in ['front_right','front_left','rear_left','rear_right']])
    if np.any(env.model.geom_type[ids]!=int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        raise ValueError('ground start requires capsule feet')
    axial=np.abs(data.geom_xmat[ids].reshape(-1,3,3)[:,2,2])
    lowest=data.geom_xpos[ids,2]-env.model.geom_size[ids,0]-env.model.geom_size[ids,1]*axial
    floor=data.geom_xpos[env.model.geom('floor').id,2]
    qpos[2]+=float(floor-lowest.min())
    return qpos,qvel

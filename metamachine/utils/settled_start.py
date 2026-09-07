"""Opt-in supported standing reset, using the environment's deployed PD pipeline."""
import numpy as np
import mujoco


def settle_start(env, cfg):
    m,d=env.model,env.data
    qa=m.jnt_qposadr[env.joint_idx];va=m.jnt_dofadr[env.joint_idx]
    rng=env.np_random
    target=np.asarray(env.default_dof_pos,dtype=float).copy()
    bounds=np.array([cfg.get('hip_offset_rad',.04),cfg.get('ankle_offset_rad',.06)]*4)
    target+=rng.uniform(-bounds,bounds)
    d.qpos[qa]=target
    # Perturb upright roll/pitch before settling; yaw remains the sampled yaw.
    tilt=rng.uniform(-cfg.get('tilt_rad',.05236),cfg.get('tilt_rad',.05236),2)
    dq=np.zeros(4);mujoco.mju_euler2Quat(dq,np.r_[tilt,0.],'xyz')
    q=np.zeros(4);mujoco.mju_mulQuat(q,d.qpos[3:7].copy(),dq);d.qpos[3:7]=q
    mujoco.mj_forward(m,d)
    feet=[i for i in range(m.ngeom) if 'ankle_geom' in (m.geom(i).name or '')]
    if len(feet)!=4:raise ValueError('settled reset requires four ankle capsules')
    floor=m.geom('floor').id
    def lowest():
        axis=np.abs(d.geom_xmat[feet].reshape(-1,3,3)[:,2,2])
        return d.geom_xpos[feet,2]-m.geom_size[feet,0]-m.geom_size[feet,1]*axis-d.geom_xpos[floor,2]
    d.qpos[2]-=float(lowest().min())-.002
    d.qvel[:]=0.;mujoco.mj_forward(m,d)
    env._reset_actuator_response_model()
    for step in range(round(3./env.cfg.control.dt)):
        env._perform_action(target)
        if step>=round(1.2/env.cfg.control.dt):
            ground=sum(int(d.contact[i].geom1==floor or d.contact[i].geom2==floor) for i in range(d.ncon))
            if ground>=2 and np.max(abs(d.qvel[va]))<.25 and np.linalg.norm(d.qvel[:6])<.15:break
    up=d.xmat[m.body('ant0').id].reshape(3,3)[:,2]
    lean=float(np.degrees(np.arctan2(np.linalg.norm(up[:2]),up[2])))
    if lean>8 or d.qpos[2]<.25 or lowest().min()<-.01 or np.linalg.norm(d.qvel[:6])>.3:
        raise ValueError('settled reset failed standing validity guard')
    d.qvel[:3]+=rng.uniform(-.015,.015,3)
    d.qvel[3:6]+=rng.uniform(-.04,.04,3)
    d.qvel[va]+=rng.uniform(-.08,.08,8)
    # Optional activation-state curriculum: perturb AFTER settling so it survives.
    curriculum=cfg.get('activation_curriculum', {})
    recovery=False;strength=0.;projection=0.
    if curriculum.get('enabled', False):
        episode=getattr(env,'_settled_dr_episode',0)
        env._settled_dr_episode=episode+1
        strength=max(float(curriculum.get('minimum_strength',0.)),min(1., episode/max(1,int(curriculum.get('ramp_resets',100)))))
        recovery=bool(rng.random()<.2)
        amplitude=(.5+.5*strength)*(1.5 if recovery else 1.)
        saved=d.qpos.copy()
        offsets=rng.uniform(-1.,1.,8)*np.array([.12,.18]*4)*amplitude
        d.qpos[qa]+=offsets
        # Keep all actuated joints inside their physical limits.
        d.qpos[qa]=np.clip(d.qpos[qa],m.jnt_range[env.joint_idx,0]+.03,m.jnt_range[env.joint_idx,1]-.03)
        tilt=rng.uniform(-1.,1.,2)*np.radians(6)*amplitude
        mujoco.mju_euler2Quat(dq,np.r_[tilt,0.],'xyz')
        mujoco.mju_mulQuat(q,d.qpos[3:7].copy(),dq);d.qpos[3:7]=q
        mujoco.mj_forward(m,d)
        projection=-float(lowest().min())+.001
        if abs(projection)>.06:
            # Bound the height correction; reduce this sample, never spawn deep penetration.
            d.qpos[:]=saved
            d.qpos[qa]+=offsets*.3
            mujoco.mj_forward(m,d)
            projection=-float(lowest().min())+.001
        d.qpos[2]+=projection
        d.qvel[:3]+=rng.uniform(-.08,.08,3)*amplitude
        d.qvel[3:6]+=rng.uniform(-.35,.35,3)*amplitude
        d.qvel[va]+=rng.uniform(-.4,.4,8)*amplitude
    d.time=0.;mujoco.mj_forward(m,d)
    up=d.xmat[m.body('ant0').id].reshape(3,3)[:,2]
    lean=float(np.degrees(np.arctan2(np.linalg.norm(up[:2]),up[2])))
    env.reset_pos=d.qpos[:2].copy()
    env.last_pos_sim=d.qpos[qa].copy();env.last_last_pos_sim=env.last_pos_sim.copy()
    env.last_vel_sim=d.qvel[va].copy();env.last_last_vel_sim=env.last_vel_sim.copy()
    env.last_com_pos=d.qpos[:3].copy()
    env.settled_start_diagnostics=dict(height=float(d.qpos[2]),tilt_deg=lean,
        joint_positions=d.qpos[qa].tolist(),hold_target=target.tolist(),minimum_clearance=float(lowest().min()),
        curriculum_strength=strength,recovery=recovery,height_projection_m=projection)

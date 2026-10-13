"""Opt-in finite reset distribution; no reference actions or motion-tracking reward."""
import hashlib
import json
from pathlib import Path
import numpy as np

def control_contract(cfg):
    result = {k: list(map(float, cfg.control[k])) for k in ('kp', 'kd', 'target_lower', 'target_upper', 'torque_limits')}
    if cfg.control.get('target_reference', 'default') != 'default' or cfg.control.get('filter_alpha', 1.) != 1.:
        result.update(target_reference=str(cfg.control.get('target_reference', 'default')),
                      filter_alpha=float(cfg.control.get('filter_alpha', 1.)),
                      delta_scale=list(map(float, cfg.control.get('delta_scale', [.25]*len(result['kp'])))),
                      default_joint_pos=list(map(float, cfg.control.default_joint_pos)))
    return json.dumps(result, sort_keys=True)


def load_bank(env, source_path):
    path=Path(env.cfg.task.reset_bank.path)
    if not path.is_absolute():path=Path(source_path).resolve().parent/path
    path=path.resolve()
    with np.load(path,allow_pickle=False) as f:
        if str(f['scene_sha256']) != hashlib.sha256(env._xml_string.encode()).hexdigest():
            raise ValueError('Reset bank scene differs from the configured physics/geometry')
        if str(f['control_json']) != control_contract(env.cfg):
            raise ValueError('Reset bank controller differs from validated controller')
        bank={k:f[k].copy() for k in ('qpos','qvel','last_action')}
    n=len(bank['qpos'])
    for key,width in [('qpos',env._mj_model.nq),('qvel',env._mj_model.nv),('last_action',env._mj_model.nu)]:
        if not n or bank[key].shape!=(n,width) or not np.isfinite(bank[key]).all():
            raise ValueError('Invalid reset bank array: '+key)
    if (np.abs(bank['last_action'])>1).any():raise ValueError('Reset actions exceed [-1,1]')
    env.spec.reset_bank_metadata={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'state_count':n,'nominal_probability':float(env.cfg.task.reset_bank.nominal_probability)}
    env.cfg.task.reset_bank.path=str(path)
    return bank


def archive_bank(env, output):
    """Keep the exact reset distribution with each run, independent of workspace edits."""
    if "reset_bank" not in env.cfg.task:
        return
    import shutil
    destination=Path(output).resolve()/"reset_bank.npz"
    shutil.copy2(env.cfg.task.reset_bank.path,destination)
    env.cfg.task.reset_bank.path=str(destination)

"""Episode-fixed ankle geometry sampling from immutable nominal ranges."""
import numpy as np


def sample_ankle_lengths(ranges, cfg, rng=np.random):
    """Legacy percentage is per-leg; explicit independent_percentage opts into
    PD-style shared percentage * independent per-leg percentage scaling.
    Offset is in metres and is applied last, once.
    """
    percentage = float(cfg.get('percentage', .1))
    independent = float(cfg.get('independent_percentage', 0.))
    offset = float(cfg.get('offset', 0.))
    if not all(np.isfinite(x) for x in (percentage, independent, offset)):
        raise ValueError('ankle length parameters must be finite')
    if not (0 <= percentage < 1 and 0 <= independent < 1):
        raise ValueError('ankle length percentages must be in [0, 1)')
    if 'independent_percentage' not in cfg:
        lengths = {k: float(rng.uniform(*v) + offset) for k, v in ranges.items()}
        shared = None
    else:
        shared = float(rng.uniform(1-percentage, 1+percentage))
        lengths = {k: float(np.mean(v) * shared * rng.uniform(1-independent, 1+independent) + offset)
                   for k, v in ranges.items()}
    if any(not np.isfinite(v) or v <= 0 for v in lengths.values()):
        raise ValueError('randomization.ankle_length produced a non-positive or invalid geom length')
    return lengths, {'shared_scale': shared, 'lengths_m': lengths.copy(),
                     'total_scales': {k: (v-offset)/np.mean(ranges[k]) for k,v in lengths.items()}}

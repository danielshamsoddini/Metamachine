"""
Config Builder for Modular-Leg Robots

Builds a headless MetaMachine OmegaConf config from a flat morphology
sequence.  Used by both evolution scripts and tests.

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Optional

from metamachine.environments.configs.config_registry import ConfigRegistry


def create_config_from_morphology(
    morphology: list[int],
    config_name: str = "quadruped_pose_opt",
    render: bool = False,
    pose_optimization: bool = True,
    keep_default_dof_pos: bool = False,
    log_dir: Optional[str] = None,
) -> object:
    """Build a headless MetaMachine config for a given morphology sequence.

    The returned config has pose optimisation *enabled* by default (when the
    config name includes it), so every fitness evaluation automatically
    optimises the robot's initial orientation and default joint positions via
    MuJoCo MJX.

    Args:
        morphology:  Flat list of ints ``[pid, pdock, cdock, rot, ...]``.
        config_name: Name of the registered config (default includes pose opt).
        render:      Whether to enable rendering (disable for headless eval).
        pose_optimization: Whether to enable MJX pose optimisation.  Set to
                     ``False`` to skip the (slow) optimisation step and use the
                     default joint positions as-is.
        keep_default_dof_pos: When ``True`` and ``pose_optimization=False``,
                     preserve the config's ``default_dof_pos`` value instead
                     of zeroing it out.  Useful when the base config's joint
                     offsets match the morphology being evaluated (e.g. the
                     ``basic_quadruped`` config seeding a preset-based
                     fast-evolution run).  Default ``False`` (zero out) to
                     maintain backward-compatible behaviour.
        log_dir:     Optional log directory.

    Returns:
        OmegaConf config object ready to pass to ``MetaMachine(cfg)``.
    """
    cfg = ConfigRegistry.create_from_name(config_name)

    # Morphology
    cfg.morphology.configuration = list(morphology)

    # Headless defaults for fast fitness evaluation
    cfg.simulation.render = render
    cfg.simulation.render_mode = "none"
    cfg.environment.num_envs = 1

    # Disable randomisation for deterministic fitness
    cfg.initialization.randomize_orientation = False
    cfg.initialization.noisy_init = False
    cfg.initialization.randomize_ini_vel = False
    cfg.randomization.init_joint_pos.enabled = False

    # Pose optimisation override
    if not pose_optimization:
        cfg.pose_optimization.enabled = False
        if not keep_default_dof_pos:
            # Zero out hardcoded joint offsets (e.g. [0, -1, 1, 1, -1]) so
            # they don't bias controllers that weren't designed for them.
            cfg.control.default_dof_pos = [0.0] * cfg.control.num_actions

    if log_dir is not None:
        cfg.logging.data_dir = log_dir
        cfg.logging.create_log_dir = True

    return cfg

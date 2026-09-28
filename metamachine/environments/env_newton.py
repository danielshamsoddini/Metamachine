"""Experimental Newton support for simple rigid-body MetaMachine tasks.

The task lifecycle is inherited from MetaMachine. Initial support covers
state-based observations/rewards/termination and fixed primitive morphologies.
"""

from .env_sim import MetaMachine


class NewtonMetaMachine(MetaMachine):
    """Gymnasium task API with Newton Featherstone dynamics (CPU by default)."""

    def __init__(self, cfg):
        self._validate_newton_config(cfg)
        super().__init__(cfg)

    @staticmethod
    def _validate_newton_config(cfg):
        def reject(message):
            raise ValueError("Newton initial support: " + message)

        sim = cfg.simulation
        if sim.get("render_mode", "none") not in (None, "none"):
            reject("render_mode must be 'none'")
        if cfg.environment.get("num_envs", 1) != 1:
            reject("only one Gymnasium environment is supported")
        for key in (
            "terrain",
            "reset_terrain",
            "add_scaffold_walls",
            "randomize_mass",
            "randomize_damping",
        ):
            if sim.get(key, False):
                reject(f"simulation.{key} is unsupported")
        for key, value in cfg.get("randomization", {}).items():
            if (
                hasattr(value, "get")
                and value.get("enabled", False)
                and key != "init_joint_pos"
            ):
                reject(f"randomization.{key} is unsupported")
        if cfg.morphology.get("randomization", {}).get("enabled", False):
            reject("morphology randomization is unsupported")
        from omegaconf import ListConfig

        if isinstance(cfg.morphology.get("asset_file"), (list, ListConfig)):
            reject("asset lists are unsupported")
        if cfg.get("pose_optimization", {}).get("enabled", False):
            reject("pose optimization is unsupported")
        for key in ("settled_start", "symmetric_ground_start"):
            if cfg.initialization.get(key, {}).get("enabled", False):
                reject(f"initialization.{key} is unsupported")
        allowed_obs = {
            "projected_gravity",
            "ang_vel_body",
            "dof_pos",
            "dof_vel",
            "last_action",
        }
        for term in cfg.observation.components:
            if term.name not in allowed_obs:
                reject(f"observation {term.name!r} is not validated")
        if cfg.observation.get("modular", {}).get("enabled", False):
            reject("modular sensor observations are unsupported")
        allowed_rewards = {
            "linear_velocity_tracking",
            "angular_velocity_tracking",
            "action_rate",
        }
        for term in cfg.task.reward_components:
            if term.type not in allowed_rewards:
                reject(f"reward {term.type!r} is not validated")
        term = cfg.task.termination_conditions
        if term.get("termination_strategy") not in (
            None,
            "ballance",
            "ballance_up",
            "ballance_upsidedown",
            "ballance_auto",
        ):
            reject("contact-dependent termination is unsupported")
        if term.get("terminate_on_body_contact") or term.get(
            "terminate_on_geom_contact"
        ):
            reject("contact-dependent termination is unsupported")
        if cfg.task.get("goal", {}).get("enabled", False):
            reject("goal tasks are not validated")

    def _initialize_simulation(self):
        from .backends.newton import NewtonBackend

        model, data = super()._initialize_simulation()
        if model.njnt == 0 or model.jnt_type[0] != 0:
            raise ValueError("Newton MetaMachine requires a free root joint first")
        self.physics = NewtonBackend(
            model, data, device=self.cfg.simulation.get("newton_device", "cpu")
        )
        return model, data

    def set_state(self, qpos, qvel):
        self.physics.set_state(qpos, qvel)

    def _pre_reset(self):
        self.physics.reset(self.model.qpos0, self.init_qvel * 0)

    def do_simulation(self, ctrl, n_frames):
        self.physics.step(ctrl, n_frames)

    def reload_model(self, xml_string):
        raise ValueError(
            "Newton initial support does not include runtime model reloads"
        )

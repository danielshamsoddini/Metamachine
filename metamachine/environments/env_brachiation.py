"""Metamachine MJX brachiation: YAML-defined inputs, rewards, and task thresholds."""
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import State
from mujoco import mjx

from .env_mjx import MJXMetaMachine
from .brachiation_control import BrachiationController
from .brachiation_scene import build_scene, hanging_qpos, joint_indices
from .brachiation_spec import TaskSpec, resolve_config
from .brachiation_terms import OBSERVATION_FUNCTIONS, REWARD_FUNCTIONS
from . import brachiation_cart_task as cart_task
from . import brachiation_courses as courses


class BrachiationMJX(MJXMetaMachine):
    """Batched physics and task state; configurable terms are in brachiation_terms.py."""

    reset_info_keys = ("target", "last_action", "contact_steps", "course_complete", "obs_history")

    def __init__(self, cfg, source_path):
        cfg = resolve_config(cfg)
        self.cfg = self._config = cfg
        self._log_dir = None
        self._sim_dt = float(cfg.simulation.timestep)
        self._ctrl_dt = float(cfg.simulation.control_dt)
        self._n_substeps = round(self._ctrl_dt / self._sim_dt)
        if self._n_substeps < 1 or not np.isclose(self._n_substeps * self._sim_dt, self._ctrl_dt):
            raise ValueError("control_dt must be a positive integer multiple of timestep")
        self._xml_string = build_scene(cfg, source_path)
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml_string)
        self.spec = TaskSpec(cfg, self._mj_model)
        self.controller = BrachiationController(cfg, self._mj_model)
        self._setup_mjx_model()
        self._num_actions = self._mj_model.nu
        q, v = joint_indices(self._mj_model)
        self.qids, self.vids = jp.array(q), jp.array(v)
        self.root_q = int(self._mj_model.joint("root").qposadr[0])
        self.root_v = int(self._mj_model.joint("root").dofadr[0])
        self.torso = self._mj_model.body("torso").id
        self.hands = jp.array([self._mj_model.site(f"{s}_grip_site").id for s in ("left", "right")])
        self.bar_geoms = jp.array([self._mj_model.geom(f"course_bar_geom_{i}").id for i in range(courses.bar_count(cfg))])
        self.hook_geoms = jp.array([[self._mj_model.geom(f"{side}_hook_{i}").id for i in range(1, 8)]
                                   for side in ("left", "right")])
        if self.spec.cart_task:
            cart_task.setup(self)
            self.reset_info_keys = self.reset_info_keys + cart_task.GUARD_INFO_KEYS
        self._hanging_offset = jp.array(hanging_qpos(self._mj_model, cfg, [0, 0, 0]))
        self._cart_bar_height = float(self._mj_model.body("course_bar_0").pos[2])
        self._template = mjx.make_data(self._mj_model)
        self._setup_rendering(cfg)

    @property
    def unwrapped(self):
        return self

    @property
    def backend(self):
        return "mjx"

    @property
    def observation_size(self):
        return self.spec.observation_size

    def _course(self, key):
        if courses.is_spatial(self.cfg):
            return courses.sample_course(key, self.cfg, self._cart_bar_height)[0]
        b = self.cfg.bars
        keys = jax.random.split(key, 4)
        n = int(b.count)
        spacing = jax.random.uniform(keys[0], (n - 1,), minval=b.spacing[0], maxval=b.spacing[1])
        x = jp.concatenate([jp.zeros(1), jp.cumsum(spacing)])
        y = jax.random.uniform(keys[1], (n,), minval=b.lateral_offset[0], maxval=b.lateral_offset[1])
        z = jax.random.uniform(keys[2], (), minval=b.height[0], maxval=b.height[1])
        dz = jax.random.uniform(keys[3], (n,), minval=b.height_offset[0], maxval=b.height_offset[1])
        heights = z + dz
        if self.cfg.task.start == "hanging_in_cart":
            heights = (heights + self._cart_bar_height).at[0].set(self._cart_bar_height)
        return jp.stack([x, y, heights], axis=-1)

    def _metrics(self):
        result = {k: jp.array(0.) for k in ("progress", "bars_reached", "fall", "success", "reward_adjustment")}
        if self.spec.cart_task:
            result.update({k: jp.array(0.) for k in ("robot_floor", "limb_cart", "nonhook_bar", "cart_escape", "stalled", "no_hook_failure", "cart_seated")})
        for term in self.spec.rewards:
            result["reward_raw/" + term["name"]] = jp.array(0.)
            result["reward_weighted/" + term["name"]] = jp.array(0.)
        return result

    def reset(self, rng):
        rng, course_key = jax.random.split(rng)
        if courses.is_spatial(self.cfg):
            bars, quats, _ = courses.sample_course(course_key, self.cfg, self._cart_bar_height)
        else:
            bars, quats = self._course(course_key), self._template.mocap_quat
        qpos = self._hanging_offset.at[self.root_q:self.root_q + 3].add(bars[0])
        data = self._template.replace(qpos=qpos, mocap_pos=bars, mocap_quat=quats)
        data = mjx.forward(self._mjx_model, data)
        info = {"rng": rng, "target": jp.array(1, dtype=jp.int32),
                "contact_steps": jp.array(0, dtype=jp.int32), "course_complete": jp.array(False),
                "last_action": jp.zeros(self.action_size)}
        if self.spec.cart_task:
            info.update(cart_task.initial_info(self, data))
        obs, info = self._observation(data, info, reset=True)
        return State(data, obs, jp.array(0.), jp.array(0.), self._metrics(), info)

    def _observation(self, data, info, reset=False):
        info = dict(info)
        components = self.spec.observations
        keys = [None] * len(components)
        if self.spec.noisy:
            split = jax.random.split(info["rng"], len(components) + 1)
            info["rng"], keys = split[0], split[1:]
        values = []
        for i, component in enumerate(components):
            value = OBSERVATION_FUNCTIONS[component["name"]](self, data, info, component["params"])
            if np.any(np.asarray(component["noise_std"]) != 0):
                value = value + jax.random.normal(keys[i], value.shape) * jp.asarray(component["noise_std"])
            values.append(value * jp.asarray(component["scale"]))
        frame = jp.concatenate(values)
        if self.spec.clip is not None:
            frame = jp.clip(frame, -self.spec.clip, self.spec.clip)
        if reset:
            if self.spec.history_reset == "repeat":
                history = jp.tile(frame, (self.spec.history_steps, 1))
            else:
                history = jp.zeros((self.spec.history_steps, self.spec.frame_size)).at[-1].set(frame)
        else:
            history = jp.concatenate([info["obs_history"][1:], frame[None]], axis=0)
        info["obs_history"] = history  # internal storage is always oldest-to-newest
        ordered = history if self.spec.history_order == "oldest_first" else history[::-1]
        return ordered.ravel(), info

    def _hook_contacts(self, data, target=None):
        contact = data._impl.contact
        g1, g2 = contact.geom[:, 0], contact.geom[:, 1]
        bars = self.bar_geoms if target is None else jp.reshape(self.bar_geoms[target], (1,))
        bar1 = jp.any(g1[:, None] == bars, axis=1)
        bar2 = jp.any(g2[:, None] == bars, axis=1)
        def touching(hand_geoms):
            hook1 = jp.any(g1[:, None] == hand_geoms, axis=1)
            hook2 = jp.any(g2[:, None] == hand_geoms, axis=1)
            return jp.any(((hook1 & bar2) | (hook2 & bar1)) & (contact.dist < float(self.cfg.task.contact_distance)))
        return jax.vmap(touching)(self.hook_geoms)

    def _distance(self, data, target, metric=None):
        metric = self.cfg.task.distance_metric if metric is None else metric
        if metric == "bar_axis" and courses.is_spatial(self.cfg):
            return jp.min(courses.hand_distances(self, data, target))
        delta = data.site_xpos[self.hands] - data.mocap_pos[target]
        if metric == "bar_axis":
            dy = jp.maximum(jp.abs(delta[:, 1]) - float(self.cfg.bars.half_length), 0.)
            distances = jp.sqrt(delta[:, 0] ** 2 + delta[:, 2] ** 2 + dy ** 2)
        else:  # validated bar_center
            distances = jp.linalg.norm(delta, axis=-1)
        return jp.min(distances)

    def _reward(self, context):
        metrics, total = {}, jp.array(0.)
        for term in self.spec.rewards:
            raw = REWARD_FUNCTIONS[term["type"]](self, context, term["params"])
            raw = jp.where(context["finite"], raw, 0.)
            # Keep the potential debit on failure; dropping it rewards failed reaches.
            if self.spec.cart_task and float(term["weight"]) > 0 and term["type"] != "reach_potential":
                raw = jp.where(context["fall"], 0., raw)
            contribution = float(term["weight"]) * raw * (self.dt if term["scale_by_dt"] else 1.)
            metrics["reward_raw/" + term["name"]] = raw
            metrics["reward_weighted/" + term["name"]] = contribution
            total = total + contribution
        reward = total
        if self.spec.reward_clip is not None:
            reward = jp.clip(reward, *self.spec.reward_clip)
        reward = jp.where(context["finite"], reward, float(self.cfg.task.nonfinite_reward))
        metrics["reward_adjustment"] = reward - total
        return reward, metrics

    def step(self, state, action):
        action = jp.clip(action, -1., 1.)
        def physics_step(_, carry):
            data, torque_sum, power_sum, forbidden = carry
            ctrl = self.controller.controls(data.qpos, data.qvel, action, xp=jp)
            updated = mjx.step(self._mjx_model, data.replace(ctrl=ctrl))
            if self.spec.effort_metrics:
                torque = updated.qfrc_actuator[self.vids]
                torque_sum += jp.sum(torque ** 2)
                power_sum += jp.sum(jp.abs(torque * data.qvel[self.vids]))
            if self.spec.cart_task:
                contact = updated._impl.contact
                flags = cart_task.contact_flags(self, contact.geom, contact.dist)
                forbidden |= jp.concatenate([flags[:3], cart_task.outside_cart(self, updated)[None]])
            return updated, torque_sum, power_sum, forbidden

        data, torque_sum, power_sum, forbidden = jax.lax.fori_loop(
            0, self._n_substeps, physics_step,
            (state.pipeline_state, jp.array(0.), jp.array(0.), jp.zeros(4, dtype=bool)))
        target = state.info["target"]
        pos = data.xpos[self.torso]
        fall = (pos[2] < float(self.cfg.task.fall_height)) | (jp.abs(pos[1]) > float(self.cfg.task.lateral_limit))
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        guard_info, guard_metrics = {}, {}
        reach_improvement = jp.array(0.)
        if self.spec.cart_task:
            contacts = data._impl.contact
            flags = cart_task.contact_flags(self, contacts.geom, contacts.dist)
            reached, failure, guard_info, stalled = cart_task.advance(
                self, state.info, cart_task.hand_distances(self, data, target),
                cart_task.hook_loads(self, data), self._hook_contacts(data), flags[3],
                jp.any(forbidden) | fall | ~finite)
            fall |= failure
            count = jp.array(0, dtype=jp.int32)
            potential = cart_task.reach_potential(self, data, target, state.info["last_transfer_hand"])
            best = jp.maximum(state.info["reach_best"], potential)
            reach_improvement = jp.where(fall | state.info["course_complete"], 0., best - state.info["reach_best"])
            guard_info["reach_best"] = best
            guard_metrics = dict(robot_floor=forbidden[0].astype(jp.float32), limb_cart=forbidden[1].astype(jp.float32),
                                 nonhook_bar=forbidden[2].astype(jp.float32), cart_escape=forbidden[3].astype(jp.float32),
                                 stalled=stalled.astype(jp.float32), cart_seated=flags[3].astype(jp.float32),
                                 no_hook_failure=(guard_info["no_hook_steps"] >= int(self.cfg.task.cart_task.max_no_hook_steps)).astype(jp.float32))
        else:
            touching = jp.any(self._hook_contacts(data, target)) if self.cfg.task.require_hook_contact else jp.array(True)
            near = self._distance(data, target) < float(self.cfg.task.target_tolerance)
            count = jp.where(touching & near & ~state.info["course_complete"], state.info["contact_steps"] + 1, 0)
            reached = count >= int(self.cfg.task.contact_steps)
        success = reached & (target == int(self.cfg.bars.count) - 1)
        next_target = jp.minimum(target + reached.astype(jp.int32), int(self.cfg.bars.count) - 1)
        if self.spec.cart_task:
            guard_info["reach_best"] = jp.where(reached,
                cart_task.reach_potential(self, data, next_target, guard_info["last_transfer_hand"]), guard_info["reach_best"])
        done = ((fall & bool(self.cfg.task.terminate_on_fall))
                | (success & bool(self.cfg.task.terminate_on_success)) | ~finite)
        context = {"terminal": done, "before": state.pipeline_state, "after": data, "target": target,
                   "action": action, "last_action": state.info["last_action"], "reached": reached,
                   "fall": fall, "finite": finite, "reach_improvement": reach_improvement, "mean_torque_l2": torque_sum / self._n_substeps,
                   "mean_power_l1": power_sum / self._n_substeps}
        if self.spec.potential_shaping:
            context.update(cart_task.potential_context(
                self, state.pipeline_state, data, state.info, next_target,
                guard_info["last_transfer_hand"], state.info["course_complete"] | success))
        reward, reward_metrics = self._reward(context)
        info = dict(state.info, target=next_target, last_action=action,
                    contact_steps=jp.where(reached, 0, count), course_complete=state.info["course_complete"] | success)
        info.update(guard_info)
        obs, info = self._observation(data, info)
        metrics = dict(state.metrics)  # preserve metrics added by Brax evaluation wrappers
        metrics.update(reward_metrics)
        metrics.update(guard_metrics)
        metrics.update(progress=jp.where(finite, pos[0] - state.pipeline_state.xpos[self.torso, 0], 0.),
                       bars_reached=reached.astype(jp.float32), fall=fall.astype(jp.float32),
                       success=success.astype(jp.float32))
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done.astype(jp.float32),
                             metrics=metrics, info=info)

    def render(self, *args, **kwargs):
        raise NotImplementedError("Use validate_brachiation.py to render the generated training scene")

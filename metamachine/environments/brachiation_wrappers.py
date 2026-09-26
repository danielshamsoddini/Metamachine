"""Brax wrapper that samples a fresh course on every episode reset."""
from brax.envs.base import Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
import jax
import jax.numpy as jp


class FreshCourseReset(Wrapper):
    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, action):
        info = dict(state.info)
        info["steps"] = jp.where(state.done, 0., info["steps"])
        state = self.env.step(state.replace(done=jp.zeros_like(state.done), info=info), action)
        def reset_done(state):
            keys = jax.vmap(lambda k: jax.random.split(k, 2))(state.info["rng"])
            fresh = self.env.reset(keys[:, 0])

            def select(new, old):
                done = state.done.reshape(state.done.shape + (1,) * (old.ndim - state.done.ndim))
                return jp.where(done, new, old)

            info = dict(state.info)
            for name in self.unwrapped.reset_info_keys:
                info[name] = select(fresh.info[name], info[name])
            info["rng"] = keys[:, 1]
            return state.replace(pipeline_state=jax.tree.map(select, fresh.pipeline_state, state.pipeline_state),
                                 obs=select(fresh.obs, state.obs), info=info)

        return jax.lax.cond(jp.any(state.done), reset_done, lambda s: s, state)


def wrap_brachiation(env, episode_length, action_repeat=1, randomization_fn=None):
    if randomization_fn is not None:
        raise ValueError("Course randomization is implemented by the task reset")
    return FreshCourseReset(EpisodeWrapper(VmapWrapper(env), episode_length, action_repeat))

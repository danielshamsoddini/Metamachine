"""GPU PPO utilities for native batched environments (no simulator dependency)."""

import math

import torch
from torch import nn
from torch.distributions import Normal


class RunningMoments(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("variance", torch.ones(size))
        self.register_buffer("count", torch.tensor(1.0e-4))

    @torch.no_grad()
    def update(self, data):
        values = data.reshape(-1, self.mean.numel())
        mean, variance = values.mean(0), values.var(0, unbiased=False)
        count = values.shape[0]
        delta, total = mean - self.mean, self.count + count
        second = (
            self.variance * self.count
            + variance * count
            + delta.square() * self.count * count / total
        )
        self.mean.add_(delta * count / total)
        self.variance.copy_(second / total)
        self.count.copy_(total)

    def forward(self, data):
        return (data - self.mean) / torch.sqrt(self.variance + 1.0e-8)


class ActorCritic(nn.Module):
    def __init__(self, observations, actions, hidden=(256, 256)):
        super().__init__()

        def network(output, gain):
            layers, size = [], observations
            for width in hidden:
                layer = nn.Linear(size, width)
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers.extend((layer, nn.Tanh()))
                size = width
            layer = nn.Linear(size, output)
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
            return nn.Sequential(*layers, layer)

        self.actor = network(actions, 0.01)
        self.critic = network(1, 1.0)
        self.log_std = nn.Parameter(torch.zeros(actions))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def log_probability(distribution, raw_action):
        # Stable tanh Jacobian, also when tanh(raw_action) rounds to +/-1.
        correction = 2 * (
            math.log(2) - raw_action - torch.nn.functional.softplus(-2 * raw_action)
        )
        return (distribution.log_prob(raw_action) - correction).sum(-1)

    def sample(self, obs):
        distribution = Normal(
            self.actor(obs), self.log_std.clamp(-5, 2).exp(), validate_args=False
        )
        raw = distribution.sample()
        return raw.tanh(), raw, self.log_probability(distribution, raw), self.value(obs)

    def evaluate(self, obs, raw_action):
        distribution = Normal(
            self.actor(obs), self.log_std.clamp(-5, 2).exp(), validate_args=False
        )
        sample = distribution.rsample()
        entropy = -self.log_probability(distribution, sample)
        return self.log_probability(distribution, raw_action), entropy, self.value(obs)


def generalized_advantage(
    rewards, values, next_values, terminated, truncated, gamma, lam
):
    """Timeouts bootstrap final observations but stop recursion across resets."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        advantages[t] = carry
    return advantages, advantages + values


class Rollout:
    """Owned GPU storage; never retain borrowed environment output views."""

    def __init__(self, steps, worlds, observations, actions, device):
        self.obs = torch.zeros((steps, worlds, observations), device=device)
        self.raw_obs = torch.zeros_like(self.obs)
        self.raw_action = torch.zeros((steps, worlds, actions), device=device)
        for name in ("log_probability", "reward", "value", "next_value"):
            setattr(self, name, torch.zeros((steps, worlds), device=device))
        self.terminated = torch.zeros((steps, worlds), dtype=torch.bool, device=device)
        self.truncated = torch.zeros_like(self.terminated)

    @torch.no_grad()
    def collect(self, env, policy, normalize, reward_scale=1.0):
        metrics = {}
        for t in range(self.reward.shape[0]):
            # Copy before step() overwrites the environment observation buffer.
            self.raw_obs[t].copy_(env.obs)
            self.obs[t].copy_(normalize(env.obs))
            action, raw, logp, value = policy.sample(self.obs[t])
            self.raw_action[t].copy_(raw)
            self.log_probability[t].copy_(logp)
            self.value[t].copy_(value)
            obs, reward, terminated, truncated, info = env.step(action)
            self.reward[t].copy_(reward * reward_scale)
            self.terminated[t].copy_(terminated)
            self.truncated[t].copy_(truncated)
            # Sanitization is only for invalid terminal states; such transitions
            # never bootstrap. Nonfinite physics still receives its failure cost.
            self.next_value[t].copy_(policy.value(normalize(torch.nan_to_num(obs))))
            self.next_value[t].masked_fill_(terminated, 0.0)
            for key, val in info.items():
                if key not in metrics:
                    metrics[key] = torch.zeros((), device=env.device)
                metrics[key] += val.float().mean()
            env.reset(terminated | truncated)
        return {k: v / self.reward.shape[0] for k, v in metrics.items()}


def update(policy, optimizer, rollout, training):
    # Preserve the configured unroll horizon even when several sequences are
    # collected from each world to fill a training batch.
    chunks = []
    horizon = int(training["unroll_length"])
    for begin in range(0, rollout.reward.shape[0], horizon):
        span = slice(begin, begin + horizon)
        chunks.append(
            generalized_advantage(
                rollout.reward[span],
                rollout.value[span],
                rollout.next_value[span],
                rollout.terminated[span],
                rollout.truncated[span],
                float(training["discounting"]),
                float(training.get("gae_lambda", 0.95)),
            )
        )
    advantages = torch.cat([item[0] for item in chunks])
    returns = torch.cat([item[1] for item in chunks])
    flat_adv = advantages.flatten()
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1.0e-8)
    obs = rollout.obs.flatten(0, 1)
    action = rollout.raw_action.flatten(0, 1)
    old_logp = rollout.log_probability.flatten()
    returns = returns.flatten()
    batch = obs.shape[0]
    minibatches = int(training["num_minibatches"])
    if batch % minibatches:
        raise ValueError("Rollout size must be divisible by num_minibatches")
    clip = float(training.get("clipping_epsilon", 0.3))
    totals = torch.zeros(5, device=obs.device)
    updates = 0
    for _ in range(int(training["num_updates_per_batch"])):
        permutation = torch.randperm(batch, device=obs.device)
        for indices in permutation.split(batch // minibatches):
            logp, entropy, value = policy.evaluate(obs[indices], action[indices])
            difference = logp - old_logp[indices]
            ratio = difference.exp()
            surrogate = torch.minimum(
                ratio * flat_adv[indices],
                ratio.clamp(1 - clip, 1 + clip) * flat_adv[indices],
            )
            actor_loss = -surrogate.mean()
            value_loss = 0.5 * (value - returns[indices]).square().mean()
            entropy_mean = entropy.mean()
            loss = (
                actor_loss
                + float(training.get("value_cost", 0.5)) * value_loss
                - float(training["entropy_cost"]) * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy.parameters(), float(training["max_grad_norm"])
            )
            optimizer.step()
            with torch.no_grad():
                totals += torch.stack(
                    (
                        actor_loss,
                        value_loss,
                        entropy_mean,
                        ((ratio - 1) - difference).mean(),
                        grad_norm,
                    )
                )
            updates += 1
    totals /= updates
    return dict(
        zip(
            ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"),
            totals,
            strict=False,
        )
    )

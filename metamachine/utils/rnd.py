"""
Random Network Distillation (RND) for State-Covering Skill Discovery

A standalone RND implementation for computing intrinsic rewards that encourage
a new policy to visit different states from a set of existing policies.

Inspired by:
- Burda et al., "Exploration by Random Network Distillation" (2018)
- ReST: Recurrent Skill Training for unsupervised skill discovery

The key idea: Each existing policy gets its own RND module trained on rollout data.
The RND prediction error is LOW for states the policy has visited (familiar),
and HIGH for novel states. We combine across policies to reward the new policy
for visiting states that are different from ALL existing policies.

Usage:
    # 1. Create an RND module and train it on rollout data from a policy
    rnd = SimpleRND(obs_dim=12, hidden_dims=[256, 256], device="cuda:0")
    rnd.train_on_data(rollout_observations, epochs=50, batch_size=256)

    # 2. Create a collection for multiple policies
    collection = RNDCollection(obs_dim=12, device="cuda:0")
    collection.add_from_rollout_data(rollout_data_policy_A)
    collection.add_from_rollout_data(rollout_data_policy_B)
    collection.add_from_rollout_data(rollout_data_policy_C)

    # 3. Get intrinsic reward (high when state is novel to all policies)
    reward = collection.get_intrinsic_reward(current_obs)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _mlp(sizes: list[int], activation: type = nn.ReLU) -> nn.Sequential:
    """Build a simple MLP."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class SimpleRND(nn.Module):
    """A simple Random Network Distillation module.

    Contains a fixed random target network and a trainable predictor network.
    The predictor is trained to match the target's output on observed states.
    The prediction error (MSE) serves as a measure of state novelty:
    - Low error → state has been seen (familiar)
    - High error → state is novel

    Args:
        obs_dim: Dimension of the observation/state input.
        hidden_dims: Hidden layer dimensions for both networks.
        output_dim: Output embedding dimension.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list[int] = (256, 256),
        output_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.device = device

        # Fixed random target network (never trained)
        self.target = _mlp([obs_dim] + list(hidden_dims) + [output_dim])
        self.target.eval()
        for param in self.target.parameters():
            param.requires_grad = False

        # Trainable predictor network
        self.predictor = _mlp([obs_dim] + list(hidden_dims) + [output_dim])

        # Running normalization for input observations
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_var = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        self.obs_count = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

        self.to(device)

    def update_obs_normalization(self, obs: torch.Tensor) -> None:
        """Update running mean/variance for observation normalization."""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.obs_count * batch_count / total_count
        new_var = m2 / total_count

        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(new_var)
        self.obs_count.copy_(total_count)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics."""
        return (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)

    def get_rnd_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute RND prediction error (novelty score) for given observations.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim) or (obs_dim,).

        Returns:
            Prediction error per sample, shape (batch_size,).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        obs_norm = self.normalize_obs(obs)

        with torch.no_grad():
            target_out = self.target(obs_norm)
            predictor_out = self.predictor(obs_norm)
            error = torch.mean((target_out - predictor_out) ** 2, dim=-1)
        return error

    def get_training_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute training loss (MSE between predictor and target outputs).

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            Scalar loss.
        """
        obs_norm = self.normalize_obs(obs)
        with torch.no_grad():
            target_out = self.target(obs_norm)
        predictor_out = self.predictor(obs_norm)
        loss = torch.mean((target_out - predictor_out) ** 2)
        return loss

    def train_on_data(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> list[float]:
        """Train the predictor network on collected rollout observations.

        This fits the predictor to match the fixed target network on the given
        observation data. After training, the prediction error will be low for
        states similar to those in the training data, and high for novel states.

        Args:
            observations: Array of observations, shape (N, obs_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Optimizer learning rate.
            verbose: Print training progress.

        Returns:
            List of per-epoch average losses.
        """
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        elif observations.device != torch.device(self.device):
            observations = observations.to(self.device)

        # Update normalization statistics
        self.update_obs_normalization(observations)

        optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)
        dataset = torch.utils.data.TensorDataset(observations)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.predictor.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for (batch_obs,) in dataloader:
                loss = self.get_training_loss(batch_obs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  RND Training - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        self.predictor.eval()
        return losses

    def save(self, path: str) -> None:
        """Save the RND module to disk."""
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "output_dim": self.output_dim,
                "hidden_dims": [layer.out_features for layer in self.predictor if isinstance(layer, nn.Linear)][:-1],
                "target_state_dict": self.target.state_dict(),
                "predictor_state_dict": self.predictor.state_dict(),
                "obs_mean": self.obs_mean,
                "obs_var": self.obs_var,
                "obs_count": self.obs_count,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SimpleRND":
        """Load a saved RND module from disk."""
        data = torch.load(path, map_location=device, weights_only=False)
        hidden_dims = data.get("hidden_dims", [256, 256])
        rnd = cls(
            obs_dim=data["obs_dim"],
            hidden_dims=hidden_dims,
            output_dim=data["output_dim"],
            device=device,
        )
        rnd.target.load_state_dict(data["target_state_dict"])
        rnd.predictor.load_state_dict(data["predictor_state_dict"])
        rnd.obs_mean.copy_(data["obs_mean"])
        rnd.obs_var.copy_(data["obs_var"])
        rnd.obs_count.copy_(data["obs_count"])
        return rnd


class RNDCollection:
    """Collection of trained RND modules for multiple existing policies.

    Computes an intrinsic reward that encourages visiting states different
    from ALL existing policies, following the ReST approach:

        reward = -log( (1/K) * sum_k exp(-alpha * rnd_error_k) )

    Where rnd_error_k is the RND prediction error for policy k.
    High reward = state is novel to all policies.

    Args:
        obs_dim: Observation dimension.
        hidden_dims: Hidden dimensions for RND networks.
        output_dim: Output dimension for RND networks.
        alpha: Sharpness parameter for the reward (higher = sharper).
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list[int] = (256, 256),
        output_dim: int = 64,
        alpha: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.alpha = alpha
        self.device = device
        self.rnd_modules: List[SimpleRND] = []
        self.policy_names: List[str] = []

    @property
    def num_policies(self) -> int:
        return len(self.rnd_modules)

    def add_rnd_module(self, rnd: SimpleRND, name: str = "") -> None:
        """Add a pre-trained RND module."""
        self.rnd_modules.append(rnd)
        self.policy_names.append(name or f"policy_{len(self.rnd_modules) - 1}")

    def add_from_rollout_data(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        name: str = "",
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> SimpleRND:
        """Create and train a new RND module from rollout observations.

        Args:
            observations: Observations from a single policy, shape (N, obs_dim).
            name: Name for this policy's RND module.
            epochs: Training epochs for the RND predictor.
            batch_size: Training batch size.
            learning_rate: Training learning rate.
            verbose: Print progress.

        Returns:
            The trained SimpleRND module.
        """
        if isinstance(observations, np.ndarray):
            obs_dim = observations.shape[-1]
        else:
            obs_dim = observations.shape[-1]

        if obs_dim != self.obs_dim:
            raise ValueError(
                f"Observation dimension mismatch: expected {self.obs_dim}, got {obs_dim}"
            )

        rnd = SimpleRND(
            obs_dim=self.obs_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            device=self.device,
        )

        policy_name = name or f"policy_{len(self.rnd_modules)}"
        if verbose:
            print(f"\n[RNDCollection] Training RND for '{policy_name}' "
                  f"({len(observations)} samples)...")

        rnd.train_on_data(
            observations,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        self.rnd_modules.append(rnd)
        self.policy_names.append(policy_name)
        return rnd

    def get_intrinsic_reward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute the state-covering intrinsic reward.

        The reward is high when the observation is novel to ALL existing policies
        (i.e., far from states visited by any existing policy).

        Following ReST:
            For each policy k, compute rnd_error_k (prediction error).
            reward = -log( (1/K) * sum_k exp(-alpha * rnd_error_k) )

        Args:
            obs: Current observation(s), shape (obs_dim,) or (batch_size, obs_dim).

        Returns:
            Intrinsic reward, shape (batch_size,) or scalar.
        """
        if len(self.rnd_modules) == 0:
            raise ValueError("No RND modules in collection. Add policies first.")

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        # Collect RND errors from all policies
        # Shape: (batch_size, num_policies)
        errors = torch.stack(
            [rnd.get_rnd_reward(obs) for rnd in self.rnd_modules], dim=-1
        )

        # ReST-style reward: -log(mean(exp(-alpha * error)))
        # High reward = novel to all policies
        exp_neg = torch.exp(-self.alpha * errors)
        mean_exp = exp_neg.mean(dim=-1)  # Average over policies
        reward = -torch.log(mean_exp + 1e-8)

        return reward

    def get_intrinsic_reward_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Numpy-friendly wrapper for get_intrinsic_reward.
        
        Args:
            obs: Observation array, shape (obs_dim,) or (batch_size, obs_dim).
            
        Returns:
            Intrinsic reward as numpy array.
        """
        reward = self.get_intrinsic_reward(obs)
        return reward.cpu().numpy()

    def save(self, save_dir: str) -> None:
        """Save all RND modules to a directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        import json
        metadata = {
            "obs_dim": self.obs_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "alpha": self.alpha,
            "num_policies": len(self.rnd_modules),
            "policy_names": self.policy_names,
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save each RND module
        for i, (rnd, name) in enumerate(zip(self.rnd_modules, self.policy_names)):
            rnd.save(str(save_path / f"rnd_{i}_{name}.pt"))

        print(f"[RNDCollection] Saved {len(self.rnd_modules)} RND modules to {save_dir}")

    @classmethod
    def load(cls, load_dir: str, device: str = "cpu") -> "RNDCollection":
        """Load a saved RNDCollection from a directory."""
        import json

        load_path = Path(load_dir)
        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        collection = cls(
            obs_dim=metadata["obs_dim"],
            hidden_dims=metadata["hidden_dims"],
            output_dim=metadata["output_dim"],
            alpha=metadata["alpha"],
            device=device,
        )

        for i, name in enumerate(metadata["policy_names"]):
            rnd_path = load_path / f"rnd_{i}_{name}.pt"
            rnd = SimpleRND.load(str(rnd_path), device=device)
            collection.add_rnd_module(rnd, name=name)

        print(f"[RNDCollection] Loaded {collection.num_policies} RND modules from {load_dir}")
        return collection

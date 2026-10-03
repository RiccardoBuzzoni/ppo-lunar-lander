"""
Rollout buffer for PPO.

Role in the pipeline
--------------------
PPO is an on-policy algorithm, it collects a fixed-size batch of experience with the current policy,
computes advantages for that batch, performs a handful of gradient updates on it, and then throws the
data away. This module implements that short-lived storage.

Concretely, it is responsible for:
1)  Storing one rollout's woth of transitions collected across 'num_envs' parallel environments over
    'num_steps' timestamps each, as flat pre-allocated tensor of shape (num_steps, num_envs, *feature_shape).
2)  Computing advantages and returns via Generalized Advantage Estimation (GAE) once the rollout is complete,
    using the value function's bootstrap estimate for the final state of each environment.
3)  Serving the collected data back out as shuffled minibatches for the PPO update epochs ('update_epochs' 
    passes over the same rollout, as configured in 'PPOConfig').
    
The buffer is intentionally stateless across rollouts: 'reset()' discards the previous rollout entirely 
once its data has been consumed by the update step, keeping the implementation faithful to PPO's on-policy
nature.
"""

from typing import Iterator, Tuple
import numpy as np
import torch

class RolloutBuffer:
    """
    Fixed-size, pre-allocated storage for one PPO rollout.
    
    Parameters:
    num_steps: int
        Number of timesteps collected per environment before an update.
    num_envs: int
        Number of parallel environments.
    obs_shape: tuple
        Shape of a single environment's observation.
    action_shape: tuple
        Shape of a single environment's action.
    device: torch.device or str
        Device the tensors live on (should match the policy's device).
    gamma: float
        Discount factor used in GAE.
    gae_lamda: float
        Lambda smoothing parameter used int GAE (bias/variance trade.off).
    """
    def __init__(
            self,
            num_steps: int,
            num_envs: int,
            obs_shape: tuple,
            action_shape: tuple,
            device: torch.device,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        shape = (num_steps, num_envs)
        self.obs = torch.zeros(shape + obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(shape + action_shape, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(shape, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=device)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=device)

        # Filled in by compute_returns_and_advantages()
        self.advantages = torch.zeros(shape, dtype=torch.float32, device=device)
        self.returns = torch.zeros(shape, dtype=torch.float32, device=device)

        self.step = 0 # current write position

    def add(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            log_prob: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
            value: torch.Tensor,
    ) -> None:
        """
        Store one timestep of transitions from all parallel envs.
        """
        if self.step >= self.num_steps:
            raise RuntimeError(
                "RolloutBuffer is full; call reset() before adding more data."
            )
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value

        self.step += 1

    def compute_returns_and_advantages(
            self, last_value: torch.Tensor, last_done: torch.Tensor
    ) -> None:
        """
        Compute GAE advantages and bootstrapped returns for the full rollout.
        
        Must be called after the buffer has been completely filled ('step == num_steps'), using
        the critic's  value estimate for the state immediately following the last stored
        transition (the "bootstrap" value), since the true return beyond the rollout horizon
        is unknown.
        
        Parameters:
        last_value: torch.Tensor
            Value estimate for the state after the final stored step.
        last_done: torch.Tensor
            Done flags for the state after the final stored step.
        """
        last_gae_lam = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # TD residual between the bootstrapped one-step return and the current value estimate.
            delta = (
                self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            )
            # Recursive GAE accumulation:
            # exponentially-weigthed sum of future deltas, discounted by (gamma*gae_lambda) each step.
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

            # Returns are used as the value function's regression target;
            # advantages are used for the policy gradient term.
            self.returns = self.advantages + self.values

    def get(self, batch_size: int, num_minibatches: int) -> Iterator[dict]:
        """
        Yield shuffled minibatches for the PPO update epochs.
        
        Flattens the (num_steps, num_envs, ...) storage into a single batch dimension,
        shuffles the indices, and yields one dict of tensors per minibatch. Advantages
        are normalised per-minibatch to stabilise the policy gradient scale.
        
        Parameters:
        batch_size: int
            Total number of samples in the rollout (num_steps * num_envs).
        num_minibatches: int
            Number of minibatches to split the batch into per epoch.
        """
        minibatch_size = batch_size // num_minibatches
        indices = np.random.permutation(batch_size)

        # Flatten the (num_steps, num_envs, ...) leading dims into one.
        flat_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        flat_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]

            mb_advantages = flat_advantages[mb_idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            yield{
                "obs": flat_obs[mb_idx],
                "actions": flat_actions[mb_idx],
                "log_probs": flat_log_probs[mb_idx],
                "advantages": flat_advantages[mb_idx],
                "returns": flat_returns[mb_idx],
                "values": flat_values[mb_idx],
            }

    def reset(self) -> None:
        """
        Reset the write position, allowing the buffer to be reused in-place.
        """
        self.step = 0
        
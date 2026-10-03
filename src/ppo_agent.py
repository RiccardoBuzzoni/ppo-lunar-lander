"""
Actor-Critic network and PPO update logic.

Role in the pipeline
--------------------
This module defines what to learn (the policy and value networks) and how to learn it (the PPO
clipped-objective update), while 'buffer.py' is responsible for what data to learn from (the
rollout storage and advantage computation) and 'train.py' orchestrates the loop between the two.

Two things live here:
1) 'ActorCritic' - A small MLP with two heads sharing no parameters:
    - The actor outputs the mean of a diagonal Gauss policy over continuous actions.
    - The critic outputs a scalar state-value estimate.
    Continuous actions are why the policy is a Gaussian distribution rather than a categorical
    one.
2) 'PPOAgent' - Wraps the network and optimiser, and implements the PPO update: for 'update_epochs'
    passes over the rollout buffer's minibatches, it computes the clipped surrogate policy loss,
    the value loss, subtracts an entropy bonus to encourage exploration, and takes a gradient step
    with gradient norm clipping for stability.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from config import PPOConfig

def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Orthogonal weight initialisation, standard practice for PPO.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous control.
    
    The actor is a diagonal Gaussian policy: the network outputs the mean of the distribution
    as a function of the state, while the log standard deviation is a single learnable parameter
    vector shared across all states. The critic is a separate MLP head estimating the state-value
    function V(s), used to compute advantages via GAE.
    
    Parameters:
    obs_dim: int
        Dimensionality of the flattened observation space.
    action_dim: int
        Dimensionality of the continuous action space.
    hidden_size: int
        Width of each hidden layer.
    num_hidden_layers: int
        Number of hidden layers in both actor and critic MLPs.
    log_std_init: float
        Initial value for the learnable log standard deviation.
    """
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_size: int = 64,
            num_hidden_layers: int = 2,
            log_std_init: float = 0.0,
    ) -> None:
        super().__init__()

        def make_mlp(out_dim: int, out_std: float) -> nn.Sequential:
            layers = []
            in_dim = obs_dim
            for _ in range(num_hidden_layers):
                layers.append(_layer_init(nn.Linear(in_dim, hidden_size)))
                layers.append(nn.Tanh())
                in_dim = hidden_size
            layers.append(_layer_init(nn.Linear(in_dim, out_dim), std=out_std))
            return nn.Sequential(*layers)

        # Critic outputs a single scalar value: small final-layer std (1.0) is
        # standard for value heads.
        self.critic = make_mlp(out_dim=1, out_std=1.0)

        # Actor outputs the mean of the Gaussian: small final-layer std (0.01)
        # keeps the initial policy close to a uniform/undecided distribution.
        self.actor_mean = make_mlp(out_dim=action_dim, out_std=0.01)

        # State-independent log-std, one value per action dimension.
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return the critic's state-value estimate, shape (batch,).
        """
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
            self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action (or evaluate a given one) under the current policy.
        
        Parameters:
        obs: Torch.tensor
            shape (batch, obs_dim)
        action: Optional[torch.Tensor]
            If provided, the distribution is evaluated at this action instead of sampling a
            new one.
        
        Returns:
        action: torch.Tensor
            shape (batch, action_dim)
        log_prob: torch.Tensor
            shape (batch, )
            Sum of per-dimension log-probabilities, needed for the PPO probability ratio.
        entropy: torch.Tensor
            shape (batch,)
            Policy entropy, used for the entropy bonus.
        value: torch.Tensor
            shape (batch,)
            Critic's value estimate for 'obs'.
            
        """
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)

        return action, log_prob, entropy, value

class PPOAgent:
    """
    Wraps an 'ActorCritic' network with an optimiser and the PPO update rule.
    
    Parameters:
    obs_dim: int
    action_dim: int
    cfg: PPOConfig
        Hyperparameters (learning rate, clip coefficient, loss weights, etc.).
    """
    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            log_std_init=cfg.log_std_init,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.learning_rate, eps=1e-5
        )

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action for environment stepping.
        """
        action, log_prob, _, value = self.network.get_action_and_value(obs)
        return action, log_prob, value

    def update(self, buffer, batch_size: int, ent_coef: Optional[float] = None) -> dict:
        """
        Run 'update_epochs' passes of PPO's clipped-objective update over the buffer.
        
        For each minibatch, the probability ration between the current and old policy
        is clipped to [1 - clip_coef, 1 + clip_coef] to prevent destrictively large
        policy updates.
        
        Returns:
        dict of average diagnostic losses over the update, useful for logging
        (policy_loss, value_loss, entropy, approx_kl).
        """
        cfg = self.cfg
        current_ent_coef = cfg.ent_coef if ent_coef is None else ent_coef
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        num_updates = 0

        for _ in range(cfg.update_epochs):
            for mb in buffer.get(batch_size, cfg.num_minibatches):
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(mb["obs"], mb["actions"])

                # Probability ratio r_t(theta) = pi_new(a|s) / pi_old(a|s),
                # computed in log-space for numerical stability.
                log_ratio = new_log_prob - mb["log_probs"]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Approximate KL divergence between old and new policy,
                    # a useful diagnostic to detect overly large updates.
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                # --- Clipped surrogate policy loss ---
                advantages = mb["advantages"]
                surr1 = advantages * ratio
                surr2 = advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value function loss ---
                if cfg.clip_value_loss:
                    value_clipped = mb["values"] + torch.clamp(
                        new_value - mb["values"], -cfg.clip_coef, cfg.clip_coef
                    )
                    value_loss_unclipped = (new_value - mb["returns"]) ** 2
                    value_loss_clipped = (value_clipped - mb["returns"]) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = policy_loss - current_ent_coef * entropy_loss + cfg.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl.item()
                num_updates += 1

        return {k: v / num_updates for k, v in stats.items()}
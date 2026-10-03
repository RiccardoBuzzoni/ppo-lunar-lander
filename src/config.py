"""
Configuration for the PPO continuous control project.

Centralises all hyperparameters in a single dataclass so that
train.py, evaluate.py and demo scripts share one source of truth.
"""

from dataclasses import dataclass, field
import torch

@dataclass
class PPOConfig:
    # --- Environment ---
    env_id: str = "LunarLanderContinuous-v3"
    num_envs: int = 8               # parallel environments
    seed: int = 42

    # --- Rollout / Training loop ---
    total_timesteps: int = 2_000_000
    num_steps: int = 2048           # steps collected per environment before each update
    num_minibatches: int = 32
    update_epochs: int = 10         # PPO epochs per rollout

    # --- PPO hyperparameters ---
    gamma: float = 0.99 # discount factor
    gae_lambda: float = 0.95        # GAE smoothing parameter
    clip_coef: float = 0.2          # PPO clipping epsilon
    clip_value_loss: bool = True
    ent_coef: float = 0.01          # initial entropy bonus coefficient (encourages exploration)
    ent_coef_final: float = 0.0     # entropy bonus coefficient at the end of training
    vf_coef: float = 0.5            # value function loss weigth
    anneal_ent_coef: bool = True    # linearly decay ent_coef from ent_coef to ent_coef_final
    max_grad_norm: float = 0.5      # gradient clipping

    # --- Optimiser ---
    learning_rate: float = 3e-4
    anneal_lr: bool = True          # linearly decay LR to 0 over training

    # --- Policy network ---
    hidden_size: int = 64
    num_hidden_layers: int = 2
    log_std_init: float = 0.0       # initial log std for the Gaussian policy

    # --- Observation/ reward optimisation ---
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # --- Loggin / Checkpoints ---
    log_interval: int = 1           # log every N updates
    checkpoint_interval: int = 50   # save model every N updates
    result_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"

    # --- Device ---
    # "auto" picks CUDA if available, otherwise falls back to CPU.
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Device can also be forced explicitly
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU...")
            self.device = "cpu"

    @property
    def batch_size(self) -> int:
        """
        Total samples collected per rollout across all parallel envs.
        """
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """
        Size of each minibatch used during the PPO update epochs.
        """
        return self.batch_size // self.num_minibatches

if __name__ == "__main__":
    # Quick sanity check when running file directly
    cfg = PPOConfig()
    print(cfg)
    print(f"batch_size = {cfg.batch_size}, minibatch_size = {cfg.minibatch_size}")
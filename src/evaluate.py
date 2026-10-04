"""
Evaluation script for a trained PPO policy.

Role in the pipeline
--------------------
'train.py' optimizes the policy. This script measures how good it actually is, independent
of the training loop. It loads a saved checkpoint (the 'ActorCritic' state dict), runs the
policy for a number of full episodes with exploration turned off, and reports the raw
(unnormalized) episodic return.

Two choices that differ from training on purpose:
- Deterministic actions: instead of sampling from the Gaussian policy, used during training
  to explore, evaluation uses the distribution's mean directly. This reflects the policy's
  "best guess" rather than its exploratory behavior, and it is standard practice for reporting
  final performance.
- No reward normalization: training uses 'NormalizeReward' to stabilize the value function's
  learning signal, but a normalized reward is meaningless for reporting. Evaluation therefore
  skips that wrapper entirely and reports true environment returns.
  
Caveat on observation normalization: the policy was trained on observations normalized with
running statistics accumulated over millions of steps. This script re-wraps the environment
with a fresh 'NormalizeObservation' wrapper, whose statistics start from scratch and adapt
online as evaluation episodes run. For a short evaluation this can slightly mismatch the
distribution the policy was trained on. Running more episodes gives the running statistics
time to settle and produces a more reliable estimate.
"""

import argparse
import os
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
from config import PPOConfig
from ppo_agent import ActorCritic

def _find_obs_rms(env):
    """
    Walk the wrapper chain to find the NormalizeObservation wrapper's running stats.
    """
    e = env
    while e is not None:
        if hasattr(e, "obs_rms"):
            return e.obs_rms
        e = getattr(e, "env", None)
    return None

def make_eval_env(env_id: str, normalize_obs: bool, seed: int, render: bool, video_dir: str, obs_rms_path: Optional[str] = None):
    """
    Build a single (non-vectorized) environment for evaluation.
    """
    render_mode = "rgb_array" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        if obs_rms_path is not None and os.path.exists(obs_rms_path):
            data = np.load(obs_rms_path)
            rms = _find_obs_rms(env)
            if rms is not None:
                rms.mean = data["mean"]
                rms.var = data["var"]
                rms.count = float(data["count"])
                print(f"Loaded observation normalization stats from {obs_rms_path}")
        else:
            print(
                "Warning: no saved observation normalization stats found at "
                f"{obs_rms_path}. Falling back to fresh stats, which may not "
                "match the distribution the policy was trained on."
            )
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space
        )
    if render:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)
    env.reset
    return env

@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    cfg: PPOConfig,
    num_episodes: int = 20,
    deterministic: bool = True,
    render: bool = False,
    video_dir: str = "results/videos",
    obs_rms_path: Optional[str] = None,
) -> dict:
    """
    Run the trained policy for 'num_episodes' episodes and report return statistics.
    
    Returns a dict with mean/std/min/max episodic return, which callers (CLI entry point, 
    notebooks, future hyperparameter sweeps) can use directly instead of re-parsing printed
    output.
    """
    device = torch.device(cfg.device)
    env = make_eval_env(cfg.env_id, cfg.normalize_obs, cfg.seed, render, video_dir, obs_rms_path)

    obs_space = env.observation_space
    action_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Box), f"Expected Box observation space, got {type(obs_space)}"
    assert isinstance(action_space, gym.spaces.Box), f"Expected Box action space, got {type(action_space)}"

    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))

    network = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        log_std_init=cfg.log_std_init,
    ).to(device)
    network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    network.eval()

    returns = []
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=cfg.seed + episode)
        done = False
        episodic_return = 0.0
        info: dict = {}

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            if deterministic:
                # Use the Gaussian's mean directly instead of sampling,
                # reflecting the policy's best guess rather than
                # exploratory behavior.
                action = network.actor_mean(obs_tensor).squeeze(0)
            else:
                action, _, _, _ = network.get_action_and_value(obs_tensor)
                action = action.squeeze(0)

            clipped_action = torch.clamp(
                action,
                torch.tensor(action_space.low, device=device),
                torch.tensor(action_space.high, device=device),
            )
            obs, reward, terminated, truncated, info = env.step(clipped_action.cpu().numpy())

            done = terminated or truncated

            # Prefer the raw, unnormalized return reported by
            # RecordEpisodeStatistics (via info["episode"]) when
            # available, since 'reward' here may be wrapper-modified
            # upstream in other configurations.
            episodic_return += float(reward)

        if "episode" in info:
            episodic_return = float(info["episode"]["r"])

        returns.append(episodic_return)
        print(f"Episode {episode + 1}/{num_episodes}: return = {episodic_return:.2f}")

    env.close()

    stats = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }
    print(
        f"\nEvaluation over {num_episodes} episodes: "
        f"mean={stats['mean_return']:.2f} +/- {stats['std_return']:.2f} "
        f"(min={stats['min_return']:.2f}, max={stats['max_return']:.2f})"
    )
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy.")
    parser.add_argument(
        "--checkpoint", type=str, default="results/checkpoints/ppo_final.pt", help="Path to the model checkpoint."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the deterministic mean.")
    parser.add_argument("--render", action="store_true", help="Record videos of evaluation episodes.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Override device.")
    args = parser.parse_args()

    config = PPOConfig()
    if args.device is not None:
        config.device = args.device
        config.__post_init__()

    evaluate(
        checkpoint_path=args.checkpoint,
        cfg=config,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        obs_rms_path=os.path.join(os.path.dirname(args.checkpoint), "obs_rms.npz"),
    )
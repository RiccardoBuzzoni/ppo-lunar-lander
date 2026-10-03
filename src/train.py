"""
Training entry point for PPO on continuous control environments.

Role in the pipeline
--------------------
This script is the orchestrator: it wires together 'PPOConfig' (hyperparameters),
'RolloutBuffer' (storage + GAE), and 'PPOAgent' (network + update rule) into the
actual training loop.

At a high level, each "update" iteration does:
1) Roll out 'num_steps' timesteps across 'num_envs' parallel environments using
   the current policy, storing transitions in the buffer.
2) Bootstrap the value of the final state and compute GAE advantages/returns for
   the whole rollout.
3) Run PPO's clipped-objective update for 'update_epochs' epochs over shuffled
   minibatches of that rollout.
4) Log diagnostics and periodically checkpoint the policy.

This repeats until 'total_timesteps' have been collected.
"""

import os
import time
import gymnasium as gym
import numpy as np
import torch
from buffer import RolloutBuffer
from config import PPOConfig
from ppo_agent import PPOAgent

def _find_obs_rms(env):
    """
    Walk the wrapper chain to find the NormalizeObservation wrapper's
    running statistics (obs_rms), regardless of how deep it sits in
    the stack. Returns None if no such wrapper is present.
    """
    e = env
    while e is not None:
        if hasattr(e, "obs_rms"):
            return e.obs_rms
        e = getattr(e, "env", None)
    return None

def make_env(env_id: str, seed: int, idx: int, normalize_obs: bool, normalize_reward: bool):
    """
    Factory returning a thunk that builds a single environment instance.
    
    Used by SyncVectorEnv, which expects a list of callables rather than already-constructed environments.
    """
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space
            )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
            env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10.0, 10.0)) # type: ignore
        env.reset(seed=seed + idx)
        return env
    return thunk

def train(cfg: PPOConfig) -> None:
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # --- Environment setup ---
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(cfg.env_id, cfg.seed, i, cfg.normalize_obs, cfg.normalize_reward)
            for i in range(cfg.num_envs)
        ]
    )
    # Both spaces are expected to be Box for this project (continuous
    # observations and actions). Asserting this narrows the type for
    # the static checker and fails fast with a clear message if an
    # incompatible (e.g. discrete) environment is passed by mistake.
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space
    assert isinstance(obs_space, gym.spaces.Box), f"Expected Box observation space, got {type(obs_space)}"
    assert isinstance(action_space, gym.spaces.Box), f"Expected Box action space, got {type(action_space)}"
 
    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_low = torch.tensor(action_space.low, device=device)
    action_high = torch.tensor(action_space.high, device=device)

    # --- Agent and buffer ---
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, cfg=cfg)
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        device=device,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    num_updates = cfg.total_timesteps // cfg.batch_size
    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        # Linear learning rate annealing over the course of training.
        if cfg.anneal_lr:
            agent.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # Linear entropy coefficient annealing: high early (exploration),
        # decaying toward ent_coef_final late in training (refinement),
        # so the policy isn't still being pushed toward exploratory
        # actions once it should be converging.
        if cfg.anneal_ent_coef:
            current_ent_coef = cfg.ent_coef_final + frac * (cfg.ent_coef - cfg.ent_coef_final)
        else:
            current_ent_coef = cfg.ent_coef

        # --- Rollout collection ---
        buffer.reset()
        for _ in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs = next_obs
            done = next_done

            action, log_prob, value = agent.act(obs)
            # Gaussian sampling can produce actions outside the valid
            # range: clip before stepping the environment (the clipped
            # action is what actually gets executed and is what we
            # should train on, since log_prob was computed pre-clip
            # following common PPO continuous-control practice).
            clipped_action = torch.clamp(action, action_low, action_high)

            next_obs_np, reward, terminated, truncated, infos = envs.step(
                clipped_action.cpu().numpy()
            )
            next_done_np = np.logical_or(terminated, truncated)

            buffer.add(obs, action, log_prob, torch.tensor(reward, dtype=torch.float32, device=device), done, value)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            # gymnasium's vector env auto-resets: episode stats are
            # reported via the `infos` dict when an episode ends.
            if "episode" in infos:
                finished = infos["episode"]["_r"] if "_r" in infos["episode"] else None
                episodic_returns = infos["episode"]["r"]
                if finished is not None:
                    episodic_returns = episodic_returns[finished]
                if len(np.atleast_1d(episodic_returns)) > 0:
                    mean_return = float(np.mean(episodic_returns))
                    print(f"global_step={global_step}, episodic_return={mean_return:.2f}")

        # --- Bootstrap value for the state after the last stored step ---
        with torch.no_grad():
            last_value = agent.network.get_value(next_obs)
        buffer.compute_returns_and_advantages(last_value, next_done)
 
        # --- PPO update ---
        stats = agent.update(buffer, batch_size=cfg.batch_size, ent_coef=current_ent_coef)

        if update % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            print(
                f"update={update}/{num_updates} step={global_step} sps={sps} "
                f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} approx_kl={stats['approx_kl']:.5f} "
                f"ent_coef={current_ent_coef:.4f}"  
            )
 
        if update % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"ppo_update{update}.pt")
            torch.save(agent.network.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint at the end of training.
    final_path = os.path.join(cfg.checkpoint_dir, "ppo_final.pt")
    torch.save(agent.network.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")

    # Save observation normalization statistics (if used) so evaluate.py
    # can reproduce the exact input distribution the policy was trained
    # on, instead of starting from fresh (mismatched) running stats.
    # Each parallel sub-env keeps its own obs_rms (SyncVectorEnv wraps
    # each individually); we average across them for a single, more
    # data-informed estimate.
    if cfg.normalize_obs:
        rms_list = [_find_obs_rms(e) for e in envs.envs]
        rms_list = [r for r in rms_list if r is not None]
        if rms_list:
            mean = np.mean([r.mean for r in rms_list], axis=0)
            var = np.mean([r.var for r in rms_list], axis=0)
            count = float(np.sum([r.count for r in rms_list]))
            obs_rms_path = os.path.join(cfg.checkpoint_dir, "obs_rms.npz")
            np.savez(obs_rms_path, mean=mean, var=var, count=count)
            print(f"Saved observation normalization stats to {obs_rms_path}")
 
    envs.close()
 
 
if __name__ == "__main__":
    import argparse
    from dataclasses import fields
 
    parser = argparse.ArgumentParser(description="Train PPO on a continuous control environment.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total_timesteps (useful for quick smoke tests, e.g. 20000).")
    parser.add_argument("--num-envs", type=int, default=None, help="Override num_envs.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_steps per rollout.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Override device.")
    args = parser.parse_args()
 
    config = PPOConfig()
    for f in fields(config):
        cli_value = getattr(args, f.name, None)
        if cli_value is not None:
            setattr(config, f.name, cli_value)
    # Re-run device resolution in case --device was overridden after __post_init__.
    config.__post_init__()
 
    train(config)

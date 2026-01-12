"""
Test script for evaluating trained RL models.

This script loads a trained model checkpoint and evaluates it on test environments,
computing correlation metrics (Kendall Tau, Spearman Rho) and saving results.
"""

import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import torch

from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics
from core.metrics import compute_correlation_metrics
from utils import save_merged_data


def make_env(env_id: str, seed: int, cfg: dict = None):
    """
    Create an environment factory function.
    
    Args:
        env_id: Environment ID
        seed: Random seed
        cfg: Environment configuration dictionary
        
    Returns:
        A thunk function that creates the environment
    """
    if cfg is None:
        cfg = {}
    
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def main():
    """Main function to run model evaluation."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = './runs/container-v0__config__2026-01-12_16_11/ckpt/120.pt'
    #ckpt_path = './runs/container-v0__config__2026-01-12_20_50/ckpt/60.pt'

    env_id = 'container-v0'
    env_entry_point = 'envs.container_vector_env:ContainerVectorEnv'
    
    num_steps = 51
    num_envs = 50
    seed = 10
    
    # Register environment
    gym.envs.register(
        id=env_id,
        entry_point=env_entry_point,
    )
    
    # Initialize agent and load weights
    # Always load to CPU first, then move to target device
    agent = Agent(device=device, name='container').to(device)
    agent.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    agent.eval()
    
    # Create vectorized test environments
    envs = SyncVectorEnv([
        make_env(env_id, seed, dict(mode="test", index=i)) 
        for i in range(num_envs)
    ])
    
    # Initialize
    trajectories = []
    obs = envs.reset()
    valid_node = obs['valid_mask']
    data_keys = obs['data_key']
    
    # Reset again and run inference
    envs.reset()
    for step in range(num_steps):
        with torch.no_grad():
            action, logits = agent(obs)
        
        # --- 新增 POMO 核心逻辑：强制第一步起始点不同 ---
        # if step == 0:
        #     n_traj = action.shape[1]
        #     action = torch.arange(n_traj, device=device).repeat(num_envs, 1)
        # ----------------------------------------------
        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())
    
    # Process trajectories: shape (env, traj, step)
    resulting_traj = np.array(trajectories).transpose(1, 2, 0)
    
    # Save merged data for inspection
    save_merged_data(obs, resulting_traj, data_keys, valid_node)
    
    # Construct target sequence (0, 1, 2, ..., n-2, 0)
    target = np.concatenate([np.arange(resulting_traj.shape[-1] - 1), [0]])
    target = np.tile(target, (resulting_traj.shape[0], resulting_traj.shape[1], 1))
    
    # Compute correlation metrics
    tau_mean, rho_mean, tau_max, rho_max = compute_correlation_metrics(
        resulting_traj.copy(), target
    )
    
    # Print results
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"tau_mean: {tau_mean:.4f}")
    print(f"rho_mean: {rho_mean:.4f}")
    print(f"tau_max:  {tau_max:.4f}")
    print(f"rho_max:  {rho_max:.4f}")
    print("=" * 50)
    
    # Close environments
    envs.close()


if __name__ == "__main__":
    main()
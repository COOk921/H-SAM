"""
Unified Test & Evaluation Script for H-SAM.

This script performs the complete evaluation pipeline:
1. Load trained model and run inference on test environments
2. Save per-group predictions
3. Merge sub-blocks within each voyage
4. Compute final metrics (Kendall, Spearman, Rehandle Rate)
"""

import os
import shutil
import warnings
from typing import Dict, List, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")

import gym
import numpy as np
import pandas as pd
import torch

from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics
from core.metrics import evaluate_correlation_metrics


# =============================================================================
# Configuration
# =============================================================================
class TestConfig:
    """Configuration for test pipeline."""
    # Model
    ckpt_path: str = './runs/container-v0__config__2026-01-15_12_09/ckpt/180.pt'
    
    # Environment
    env_id: str = 'container-v0'
    env_entry_point: str = 'envs.container_vector_env:ContainerVectorEnv'
    
    # Test settings
    num_steps: int = 51
    num_envs: int = 1
    seed: int = 10
    
    # Output
    result_dir: str = './result/'
    merged_output_dir: str = './result/merged/'
    save_intermediate: bool = True  # Whether to save per-group CSV files


# =============================================================================
# Phase 1: Model Inference
# =============================================================================
def make_env(env_id: str, seed: int, cfg: dict = None):
    """Create an environment factory function."""
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


def run_inference(config: TestConfig) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Run model inference on test environments.
    
    Returns:
        observations: Node features (num_envs, max_nodes, dim)
        trajectories: Predicted trajectories (num_envs, traj, steps)
        data_keys: Group keys for each environment
        valid_nodes: Number of valid nodes per environment
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Register environment
    gym.envs.register(
        id=config.env_id,
        entry_point=config.env_entry_point,
    )
    
    # Initialize agent
    print("Loading model...")
    agent = Agent(device=device, name='container').to(device)
    agent.load_state_dict(torch.load(config.ckpt_path, map_location='cpu'))
    agent.eval()
    
    # Create vectorized test environments
    print(f"Creating {config.num_envs} test environments...")
    envs = SyncVectorEnv([
        make_env(config.env_id, config.seed, dict(mode="test", index=i)) 
        for i in range(config.num_envs)
    ])
    
    # Initialize
    trajectories = []
    obs = envs.reset()
    valid_nodes = obs['valid_mask']
    data_keys = obs['data_key']
    
    # Run inference
    print(f"Running inference for {config.num_steps} steps...")
    envs.reset()
    for step in range(config.num_steps):
        with torch.no_grad():
            action, _ = agent(obs)
        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())
    
    # Process trajectories: (steps, env, traj) -> (env, traj, steps)
    resulting_traj = np.array(trajectories).transpose(1, 2, 0)
    observations = obs['observations']
    
    envs.close()
    
    return observations, resulting_traj, data_keys, valid_nodes


# =============================================================================
# Phase 2: Data Grouping & Saving
# =============================================================================
def create_group_dataframes(
    observations: np.ndarray, 
    trajectories: np.ndarray, 
    data_keys: List[str]
) -> Dict[str, List[pd.DataFrame]]:
    """
    Convert inference results to grouped DataFrames.
    
    Returns:
        Dictionary mapping voyage_key -> list of cluster DataFrames
    """
    columns = [
        'target', 'order', 'Unit Weight (kg)', 'Unit POD',
        'from_yard', 'from_bay', 'from_col','position_id', 'from_layer', 'pred'
    ]
    
    # Use first trajectory only
    traj_data = trajectories[:, 0, :-1]
    traj_data = traj_data.reshape(observations.shape[0], observations.shape[1], 1)
    merged_data = np.concatenate([observations, traj_data], axis=-1)
    
    # Group by voyage (first part of key before cluster id)
    groups = defaultdict(list)
    for i in range(merged_data.shape[0]):
        df = pd.DataFrame(merged_data[i], columns=columns)
        df['voyage_cluster'] = data_keys[i]
        full_key = data_keys[i]
        voyage_key = full_key.split("',")[0] + "'" if "'," in str(full_key) else str(full_key)
        
        groups[voyage_key].append(df)
        
    return dict(groups)


def save_intermediate_results(
    observations: np.ndarray,
    trajectories: np.ndarray,
    data_keys: List[str],
    result_dir: str
):
    """Save per-environment CSV files for debugging."""
    columns = [
        'target', 'order', 'Unit Weight (kg)', 'Unit POD',
        'from_yard', 'from_bay', 'from_col','position_id', 'from_layer', 'pred'
    ]
       
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    traj_data = trajectories[:, 0, :-1]
    traj_data = traj_data.reshape(observations.shape[0], observations.shape[1], 1)
    merged_data = np.concatenate([observations, traj_data], axis=-1)
    
    for i in range(merged_data.shape[0]):
        df = pd.DataFrame(merged_data[i], columns=columns)
        file_name = os.path.join(result_dir, f"{data_keys[i]}.csv")
        df.to_csv(file_name, index=False)
    
    print(f"Saved {merged_data.shape[0]} intermediate CSV files to {result_dir}")


# =============================================================================
# Phase 3: Merge Sub-blocks
# =============================================================================
def sort_sub_blocks_by_order(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Sort sub-blocks by first row's 'order' value (ground truth ordering)."""
    if len(df_list) <= 1:
        return df_list
    
    first_order_vals = [df['order'].values[0] for df in df_list]
    sort_indices = np.argsort(first_order_vals)
    
    return [df_list[i] for i in sort_indices]


def merge_and_rerank(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge sorted sub-blocks and rerank predictions globally.
    
    Steps:
    1. Filter out invalid rows (all zeros except pred)
    2. Rerank pred within each block and apply global offset
    3. Concatenate all blocks
    """
    merged_df = pd.DataFrame()
    num_offset = 0
    
    for df in df_list:
        # Filter invalid rows (features all zeros)
        mask = ~(df.iloc[:, :-1] == 0).all(axis=1)
        df_valid = df[mask].copy()
        
        if df_valid.empty:
            continue
        
        # Rerank pred locally and add global offset
        df_valid['pred'] = df_valid['pred'].rank(method='dense').astype(int) - 1 + num_offset
        num_offset += len(df_valid)
        
        merged_df = pd.concat([merged_df, df_valid], ignore_index=True)
    
    return merged_df


# =============================================================================
# Phase 4: Evaluation
# =============================================================================
def evaluate_all_groups(
    groups: Dict[str, List[pd.DataFrame]],
    output_dir: str = None
) -> Dict[str, float]:
    """
    Evaluate all voyage groups.
    
    Returns:
        Dictionary with average metrics
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    total_kendall = 0
    total_rho = 0
    total_rehandle = 0
    total_rehandle_GT = 0
    num_groups = len(groups)
    
    print(f"\n{'='*60}")
    print(f"Evaluating {num_groups} voyage groups...")
    print(f"{'='*60}")
    
    for voyage_key, df_list in groups.items():
        # Sort sub-blocks
        sorted_dfs = sort_sub_blocks_by_order(df_list)
        
        # Merge and rerank
        merged_df = merge_and_rerank(sorted_dfs)
        
        if merged_df.empty:
            print(f"Group {voyage_key}: Empty after filtering, skipping...")
            continue

       
        # Evaluate
        kendall, rho, rehandle,rehandle_GT = evaluate_correlation_metrics(merged_df, voyage_key)
        total_kendall += kendall
        total_rho += rho
        total_rehandle += rehandle
        total_rehandle_GT += rehandle_GT
        
        # Save merged result
        if output_dir:
            safe_name = voyage_key.replace("'", "").replace(",", "_") + ".csv"
            merged_df.to_csv(os.path.join(output_dir, safe_name), index=False)
    
    # Compute averages
    results = {
        'avg_kendall': total_kendall / num_groups if num_groups > 0 else 0,
        'avg_rho': total_rho / num_groups if num_groups > 0 else 0,
        'avg_rehandle_rate': total_rehandle / num_groups if num_groups > 0 else 0,
        'avg_rehandle_rate_GT': total_rehandle_GT / num_groups if num_groups > 0 else 0,
        'num_groups': num_groups
    }
    
    return results


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    """Run the complete test and evaluation pipeline."""
    config = TestConfig()
    
    print("="*60)
    print("H-SAM Test & Evaluation Pipeline")
    print("="*60)
    print(f"Model: {config.ckpt_path}")
    print(f"Num Envs: {config.num_envs}, Num Steps: {config.num_steps}")
    print("="*60)
    
    # Phase 1: Run inference
    print("\n[Phase 1] Running model inference...")
    observations, trajectories, data_keys, valid_nodes = run_inference(config)
    print(f"Inference complete. Shape: {trajectories.shape}")
    
    # Phase 2: Group data
    print("\n[Phase 2] Grouping predictions by voyage...")
    groups = create_group_dataframes(observations, trajectories, data_keys)
    print(f"Created {len(groups)} voyage groups")
    
    # Optional: Save intermediate files
    if config.save_intermediate:
        save_intermediate_results(
            observations, trajectories, data_keys, config.result_dir
        )
    
    # Phase 3 & 4: Merge and evaluate
    print("\n[Phase 3 & 4] Merging sub-blocks and evaluating...")
    results = evaluate_all_groups(groups, config.merged_output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Number of Groups Evaluated: {results['num_groups']}")
    print(f"Average Kendall Tau:        {results['avg_kendall']:.4f}")
    print(f"Average Spearman Rho:       {results['avg_rho']:.4f}")
    print(f"Average Rehandle Rate:      {results['avg_rehandle_rate']:.4f}")
    print(f"GT Rehandle Rate:           {results['avg_rehandle_rate_GT']:.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
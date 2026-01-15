"""
Metrics calculation for RL evaluation.

This module contains functions for computing various evaluation metrics
including rehandle rate and correlation coefficients.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from typing import Tuple, List
import pdb

def count_increasing_pairs(arr: np.ndarray) -> int:
    """
    Helper function: Counts pairs (i, j) such that i < j but arr[i] < arr[j].
    Input arr represents the LAYER of containers in the order of LOADING.
    If we load Layer 1 before Layer 2, that's a rehandle (increasing pair).
    """
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] < arr[j]:
                count += 1
    return count


def compute_rehandle_rate(
    from_layer: np.ndarray, 
    from_col: np.ndarray, 
    from_bay: np.ndarray, 
    from_yard: np.ndarray, 
    Unit_POD: np.ndarray,
    voyage_cluster: np.ndarray,
    denominator: str = 'same'
) -> Tuple[int, int, float]:
    """
    Compute container rehandle statistics.

    Args:
        from_layer: Layer positions of containers (in retrieval order)
        from_col: Column positions of containers
        from_bay: Bay positions of containers
        from_yard: Yard positions of containers
        denominator: 'same' for same-stack pairs, 'all' for all pairs

    Returns:
        Tuple of (rehandle_pairs, total_pairs, rate)
    """
    from_layer = np.asarray(from_layer)
    from_col = np.asarray(from_col)
    from_bay = np.asarray(from_bay)
    from_yard = np.asarray(from_yard)
    Unit_POD = np.asarray(Unit_POD)
    voyage_cluster = np.asarray(voyage_cluster)

  
    if not (from_layer.shape == from_col.shape == from_bay.shape == from_yard.shape):
        raise ValueError("All input arrays must have the same shape/length.")

    N = from_layer.size
    
    keys_list = []
    for i in range(N):
        key_tuple = (
            from_yard[i], 
            from_bay[i], 
            from_col[i], 
            Unit_POD[i], 
            voyage_cluster[i]
        )
        keys_list.append(key_tuple)
    
    # Find unique keys and their inverse mapping
    unique_keys_dict = {}
    inverse = np.zeros(N, dtype=int)
    
    for i, key in enumerate(keys_list):
        if key not in unique_keys_dict:
            unique_keys_dict[key] = len(unique_keys_dict)
        inverse[i] = unique_keys_dict[key]
    
    num_unique_keys = len(unique_keys_dict)

    rehandle_pairs = 0
    total_same_stack_pairs = 0
    
    for gid in range(num_unique_keys):
        idxs = np.nonzero(inverse == gid)[0]
        m = idxs.size
        total_same_stack_pairs += m * (m - 1) // 2
        if m < 2:
            continue
        layers = from_layer[idxs]
        rehandle_pairs += count_increasing_pairs(layers)
   
    
    if denominator == 'same':
        denom = total_same_stack_pairs
    elif denominator == 'all':
        denom = N * (N - 1) // 2
    else:
        raise ValueError("denominator must be 'same' or 'all'")
    rate = (rehandle_pairs / denom) if denom > 0 else 0.0
    return int(rehandle_pairs), int(denom), float(rate)
    


def calculate_rehandle_from_df(df: pd.DataFrame, order_col: str = 'pred') -> Tuple[int, int, float]:
    """
    Calculate rehandle rate from a DataFrame.
    
    Args:
        df: DataFrame with columns: from_yard, from_bay, from_col, from_layer, and order_col
        order_col: Column name to use for sorting (retrieval order), default 'pred'
        
    Returns:
        Tuple of (rehandle_count, total_same_stack_pairs, rehandle_rate)
    """
    # Sort by the specified order column to get retrieval sequence
    sorted_df = df.sort_values(by=order_col).reset_index(drop=True)
    
    # Extract position features in retrieval order
    from_layer = sorted_df['from_layer'].values
    from_col = sorted_df['from_col'].values
    from_bay = sorted_df['from_bay'].values
    from_yard = sorted_df['from_yard'].values
    Unit_POD = sorted_df['Unit POD'].values
    voyage_cluster = sorted_df['voyage_cluster'].values
   
    
    return compute_rehandle_rate(from_layer, from_col, from_bay, from_yard,Unit_POD,voyage_cluster, denominator='same')


# def calculation_metrics(
#     sequence: np.ndarray, 
#     feature: np.ndarray, 
#     metric_type: str = 'URR', 
#     agg_mode: str = 'best'
# ) -> float:
#     """
#     Calculate rehandle metrics for RL evaluation (batch mode).
    
#     Args:
#         sequence: Predicted retrieval order of shape (env, traj, step)
#         feature: Feature array of shape (node, feature_dim), last 4 cols are yard, bay, col, layer
#         metric_type: 'URR' (count/N, standard for papers) or 'Ratio' (normalized [0,1])
#         agg_mode: 'best' (min rate), 'mean' (avg rate), 'worst' (max rate)
        
#     Returns:
#         Average rehandle rate across all environments
#     """
#     avg_rate = 0
#     num_envs = sequence.shape[0]
#     num_trajs = sequence.shape[1]
    
#     for i in range(num_envs):
#         traj_rates = []
#         for j in range(num_trajs):
#             seq_indices = sequence[i, j]
#             from_layer = feature[seq_indices, -1]
#             from_col = feature[seq_indices, -2]
#             from_bay = feature[seq_indices, -3]
#             from_yard = feature[seq_indices, -4]
            
#             rehandles, n_total, ratio = compute_rehandle_rate(
#                 from_layer, from_col, from_bay, from_yard, denominator='same'
#             )
            
#             if metric_type == 'URR':
#                 val = rehandles / n_total if n_total > 0 else 0
#             else:
#                 val = ratio
            
#             traj_rates.append(val)
            
#         if agg_mode == 'best':
#             env_score = min(traj_rates)
#         elif agg_mode == 'mean':
#             env_score = np.mean(traj_rates)
#         else:
#             env_score = max(traj_rates)
            
#         avg_rate += env_score

#     return avg_rate / num_envs


def evaluate_correlation_metrics(merged_df: pd.DataFrame, group_key: str) -> Tuple[float, float, float]:
    """
    Evaluate correlation metrics and rehandle rate for a merged group.
    
    Args:
        merged_df: DataFrame with columns: target, order, Unit Weight (kg), Unit POD, 
                   from_yard, from_bay, from_col, from_layer, pred
        group_key: Group identifier (for logging)
        
    Returns:
        Tuple of (kendall_corr, spearman_rho, rehandle_rate)
    """
    if merged_df.empty:
        return 0.0, 0.0, 0.0
    
    # 1. Compute correlation metrics
    kendall_corr, _ = kendalltau(merged_df['order'].values, merged_df['pred'].values)
    rho, _ = spearmanr(merged_df['order'].values, merged_df['pred'].values)
    
    # 2. Compute rehandle rate using predicted order
    rehandle_count, total_pairs, rehandle_rate = calculate_rehandle_from_df(merged_df, order_col='pred')
    rehandle_count_GT, total_pairs_GT, rehandle_rate_GT = calculate_rehandle_from_df(merged_df, order_col='order')
    
    # 3. Print results
    print(f"Group {group_key} - Kendall: {kendall_corr:.4f}, Rho: {rho:.4f}, Rehandle Rate: {rehandle_rate:.4f} ({rehandle_count}/{total_pairs}),GT Rehandle Rate:  { rehandle_rate_GT:.4f} ({rehandle_count_GT}/{total_pairs_GT})")


    
    return kendall_corr, rho, rehandle_rate,rehandle_rate_GT


def compute_correlation_metrics(
    resulting_traj: np.ndarray, 
    target: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute Kendall Tau and Spearman correlation metrics.
    
    Args:
        resulting_traj: Resulting trajectory of shape (env, traj, step)
        target: Target sequence of shape (env, traj, step)
        
    Returns:
        Tuple of (tau_mean, rho_mean, tau_max, rho_max)
    """
    tau_list = []
    rho_list = []
    
    for i in range(resulting_traj.shape[0]):
        arr = resulting_traj[i, 0, :-1]
        zero_index = np.where(arr == 0)[0][0]
        new_arr = np.concatenate([arr, arr])[zero_index:zero_index + len(arr) + 1]
        resulting_traj[i, 0] = new_arr
        
        tau, _ = kendalltau(target[i, 0], resulting_traj[i, 0])
        rho, _ = spearmanr(target[i, 0], resulting_traj[i, 0])
        
        print(f"tau: {tau:.4f}; rho: {rho:.4f}")
        if not np.isnan(tau):
            tau_list.append(tau)
            rho_list.append(rho)
            
   
    tau_mean = np.mean(tau_list) if tau_list else 0
    rho_mean = np.mean(rho_list) if rho_list else 0
    tau_max = np.max(tau_list) if tau_list else 0
    rho_max = np.max(rho_list) if rho_list else 0
    
    # print("="*30)
    # print(f"tau_mean: {tau_mean:.4f}; rho_mean: {rho_mean:.4f}")
    return tau_mean, rho_mean, tau_max, rho_max

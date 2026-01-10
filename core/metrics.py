"""
Metrics calculation for RL evaluation.

This module contains functions for computing various evaluation metrics
including rehandle rate and correlation coefficients.
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from typing import Tuple, List


def count_increasing_pairs(arr: np.ndarray) -> int:
    """
    Count pairs (i < j) with arr[i] < arr[j].
    Uses coordinate-compression + Fenwick (BIT) -> O(len(arr) log U)
    
    Args:
        arr: Input array
        
    Returns:
        Number of increasing pairs
    """
    if arr.size < 2:
        return 0
    vals, inv = np.unique(arr, return_inverse=True)
    U = vals.size
    bit = [0] * (U + 1)

    def bit_add(i):
        i += 1
        while i <= U:
            bit[i] += 1
            i += i & -i

    def bit_sum(i):
        if i < 0:
            return 0
        s = 0
        i += 1
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    cnt = 0
    for rank in inv:
        if rank > 0:
            cnt += bit_sum(rank - 1)
        bit_add(rank)
    return cnt


def compute_rehandle_rate(
    from_layer: np.ndarray, 
    from_col: np.ndarray, 
    from_bay: np.ndarray, 
    from_yard: np.ndarray, 
    denominator: str = 'same'
) -> Tuple[int, int, float]:
    """
    Compute container rehandle statistics.

    Args:
        from_layer: Layer positions of containers
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

    if not (from_layer.shape == from_col.shape == from_bay.shape == from_yard.shape):
        raise ValueError("All input arrays must have the same shape/length.")

    N = from_layer.size
    keys = np.vstack((from_yard, from_bay, from_col)).T
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    rehandle_pairs = 0
    total_same_stack_pairs = 0
    
    for gid in range(unique_keys.shape[0]):
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


def calculation_metrics(sequence: np.ndarray, feature: np.ndarray) -> float:
    """
    Calculate rehandle rate metrics.
    
    Args:
        sequence: Trajectory sequence of shape (env, traj, step)
        feature: Feature array of shape (env, node, obs_dim)
        
    Returns:
        Average rehandle rate
    """
    avg_rate = 0
    for i in range(sequence.shape[0]):
        max_traj = 0
        for j in range(sequence.shape[1]):
            from_layer = feature[sequence[i, j], -1]
            from_col = feature[sequence[i, j], -2]
            from_bay = feature[sequence[i, j], -3]
            from_yard = feature[sequence[i, j], -4]
            
            _, _, rate = compute_rehandle_rate(from_layer, from_col, from_bay, from_yard)
            max_traj = max(max_traj, rate)
        avg_rate += max_traj
    avg_rate /= sequence.shape[0]
    return avg_rate


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
        # Adjust zero position in resulting_traj
        arr = resulting_traj[i, 0, :-1]
        zero_index = np.where(arr == 0)[0][0]
        new_arr = np.concatenate([arr, arr])[zero_index:zero_index + len(arr) + 1]
        resulting_traj[i, 0] = new_arr
        
        tau, _ = kendalltau(target[i, 0], resulting_traj[i, 0])
        rho, _ = spearmanr(target[i, 0], resulting_traj[i, 0])
        
        if not np.isnan(tau):
            tau_list.append(tau)
            rho_list.append(rho)
    
    tau_mean = np.mean(tau_list) if tau_list else 0
    rho_mean = np.mean(rho_list) if rho_list else 0
    tau_max = np.max(tau_list) if tau_list else 0
    rho_max = np.max(rho_list) if rho_list else 0
    
    return tau_mean, rho_mean, tau_max, rho_max

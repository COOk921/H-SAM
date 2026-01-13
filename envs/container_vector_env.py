"""
Container Vector Environment for Reinforcement Learning.

This module defines the Container environment for RL training,
handling container scheduling optimization problems.
"""

import random
import pickle
from typing import Dict, Any, Optional, Tuple

import gym
import numpy as np
import pandas as pd
import torch
from gym import spaces
from torch_geometric.data import Data

# Global cache for data
_DATA_CACHE = None


def get_data(
    max_nodes: int,
    current_index: int,
    data_path: str = "./data/processed_container_data_hetero(Spectral_opt).pkl",
    mode: str = 'train'
) -> Tuple[np.ndarray, Data, int, str]:
    """
    Load and preprocess container data.
    
    Args:
        max_nodes: Maximum number of nodes to use
        current_index: Index for test mode
        data_path: Path to the data file
        mode: 'train' or 'test'
        
    Returns:
        Tuple of (nodes, graph, valid_nodes, key)
    """
    global _DATA_CACHE
    selected_columns = [
        'order', 'Unit Weight (kg)', 'Unit POD',
        'from_yard', 'from_bay', 'from_col', 'from_layer'
    ]
   
    if _DATA_CACHE is None:
        print("--- Loading data and dealing with graph (will happen only ONCE) ---")
        with open(data_path, 'rb') as f:
            data = pd.read_pickle(f)

        data = {
            tuple(key) if isinstance(key, np.ndarray) else key: value 
            for key, value in data.items()
        }
        keys = list(data.keys())
        
        # Process graph data
        for key in keys:
            batch_g = data[tuple(key)]['graph']
            node_type = 'container'
            batch_g[node_type].x = batch_g[node_type].x[:max_nodes]
            current_nodes = batch_g[node_type].x.shape[0]
        
            for edge_type in batch_g.edge_types:
                edge_index = batch_g[edge_type].edge_index
                mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
                batch_g[edge_type].edge_index = edge_index[:, mask]
                
                if 'edge_attr' not in batch_g[edge_type] or batch_g[edge_type].edge_attr is None:
                    num_edges = batch_g[edge_type].edge_index.size(1)
                    batch_g[edge_type].edge_attr = torch.zeros(
                        (num_edges, 1), 
                        dtype=torch.float, 
                        device=batch_g[node_type].x.device
                    )
                else:
                    batch_g[edge_type].edge_attr = batch_g[edge_type].edge_attr[mask]
            
            # Pad nodes if needed
            if current_nodes < max_nodes:
                padding_size = max_nodes - current_nodes
                feature_dim = batch_g[node_type].x.shape[1]
                dtype = batch_g[node_type].x.dtype
                device = batch_g[node_type].x.device
                
                padding_features = torch.zeros(
                    padding_size, feature_dim, 
                    dtype=dtype, device=device
                )
                batch_g[node_type].x = torch.cat(
                    [batch_g[node_type].x, padding_features], dim=0
                )
            data[tuple(key)]['graph'] = batch_g
            
        _DATA_CACHE = data
    else:
        keys = list(_DATA_CACHE.keys())

    # Select key based on mode
    if mode == 'train':
        key = random.choice(keys)
    elif mode == 'test':
        if current_index >= len(keys):
            current_index = current_index % len(keys)
        key = keys[current_index]
    
    df = _DATA_CACHE[tuple(key)]
    
    valid_nodes = np.minimum(df['data'].shape[0], max_nodes)
    nodes = df['data'][selected_columns].to_numpy()[:max_nodes]
    
    # Add index column
    indices = np.arange(len(nodes)).reshape(-1, 1)
    nodes = np.hstack((indices, nodes))

    graph = df['graph']
     
    # Pad nodes if needed
    if len(nodes) < max_nodes:
        nodes = np.pad(nodes, ((0, max_nodes - len(nodes)), (0, 0)), mode='constant')
    
    return nodes, graph, valid_nodes, key


def rule_reward(dest_node: np.ndarray, prev_node: np.ndarray) -> np.ndarray:
    """
    Calculate rule-based reward.
    
    Args:
        dest_node: Destination node features of shape (batch, n_traj, dim)
        prev_node: Previous node features of shape (batch, n_traj, dim)
        
    Returns:
        Reward array of shape (batch, n_traj)
    """
    batch, n_traj, dim = dest_node.shape
    reward = np.full((batch, n_traj), -1.0)

    # Rule reward: same yard, bay, col and valid layer order
    condition1 = np.all(dest_node[..., 2:5] == prev_node[..., 2:5], axis=-1)
    condition2 = dest_node[..., -1] < prev_node[..., -1]
    valid_condition = condition1 & condition2
    reward[valid_condition] = 0
    
    # Sequence reward: correct order
    dest_sequence = dest_node[..., 0]
    prev_sequence = prev_node[..., 0]
    valid_condition = (dest_sequence >= prev_sequence)
    reward[valid_condition] = 0

    return reward


def similarity_reward(
    x: np.ndarray, 
    y: np.ndarray, 
    eps: float = 1e-8, 
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Calculate cosine similarity-based reward.
    
    Args:
        x: First vector array
        y: Second vector array
        eps: Small value for numerical stability
        pad_value: Value for padded positions
        
    Returns:
        Similarity scores
    """
    dot_product = np.sum(x * y, axis=-1)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    
    pad_mask = (norm_x == 0) | (norm_y == 0)
    sim = np.full_like(norm_x, pad_value)
    
    valid_mask = ~pad_mask
    if np.any(valid_mask):
        dot_product_valid = dot_product[valid_mask]
        norm_x_valid = norm_x[valid_mask]
        norm_y_valid = norm_y[valid_mask]
        
        cos_sim = dot_product_valid / (norm_x_valid * norm_y_valid + eps)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        sim[valid_mask] = (cos_sim + 1) / 2
    
    return sim


def assign_env_config(env, kwargs: Dict[str, Any]):
    """Set environment attributes from kwargs."""
    for key, value in kwargs.items():
        setattr(env, key, value)


class ContainerVectorEnv(gym.Env):
    """Container environment for RL training."""
    
    def __init__(self, *args, **kwargs):
        # Default settings
        self.max_nodes = 50
        self.n_traj = 50
        self.dim = 6 + 2
        self.hidden_dim = 256
        self.eval_data = True
        self.eval_partition = "test"
        self.eval_data_idx = 0
        self.current_index = 0
        
        assign_env_config(self, kwargs)
        
        self.mode = kwargs['mode']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define observation and action spaces
        obs_dict = {
            "observations": spaces.Box(low=0, high=1, shape=(self.max_nodes, self.dim)),
            "data_key": spaces.Text(max_length=30),
            "action_mask": spaces.MultiBinary([self.n_traj, self.max_nodes]),
            "first_node_idx": spaces.MultiDiscrete([self.max_nodes] * self.n_traj),
            "last_node_idx": spaces.MultiDiscrete([self.max_nodes] * self.n_traj),
            "is_initial_action": spaces.Discrete(1),
            "valid_mask": spaces.Discrete(1),
            "graph_data": spaces.Graph(
                node_space=spaces.Box(low=0, high=1, shape=(self.max_nodes,)), 
                edge_space=None
            )
        }

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        self.reward_space = None
        
        self.current_index = kwargs.get('index', self.current_index)
        self.reset()
      
    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)

    def reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        self.visited = np.zeros((self.n_traj, self.max_nodes), dtype=bool)
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)
        self.first = np.zeros(self.n_traj, dtype=int)

        if self.eval_data:
            self._load_orders()
        else:
            self._generate_orders()
        
        self.state = self._update_state()
        self.info = {}
        self.done = False
        
        return self.state

    def _load_orders(self):
        """Load orders from data file."""
        self.nodes, self.graph, self.valid_nodes, self.key = get_data(
            max_nodes=self.max_nodes,
            current_index=self.current_index, 
            mode=self.mode
        )
        
    def _generate_orders(self):
        """Generate random orders."""
        self.nodes = np.random.rand(self.max_nodes, self.dim)
        
    def step(self, action: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
        """Execute one step in the environment."""
        self._go_to(action)
        self.num_steps += 1
        self.state = self._update_state()
        self.done = (action == self.first) & self.is_all_visited()
        return self.state, self.reward, self.done, self.info

    def is_all_visited(self) -> np.ndarray:
        """Check if all nodes are visited."""
        return self.visited.all(axis=1)

    def _go_to(self, destination: np.ndarray):
        """Move to destination node."""
        self.dest_node = self.nodes[destination]
        self.prev_node = self.nodes[self.last]
        
        # Reward is calculated in syncVectorEnvPomo.py
        if self.num_steps != 0:
            self.reward = 0
        else:
            self.reward = np.zeros(self.n_traj)
            self.first = destination

        self.last = destination
        self.visited[np.arange(self.n_traj), destination] = True

    def _update_state(self) -> Dict[str, Any]:
        """Update and return current state."""
        return {
            "observations": self.nodes,
            "data_key": str(self.key),
            "action_mask": self._update_mask(),
            "first_node_idx": self.first,
            "last_node_idx": self.last,
            "is_initial_action": self.num_steps == 0,
            "valid_mask": self.valid_nodes,
            "graph_data": self.graph,
        }

    def _update_mask(self) -> np.ndarray:
        """Update action mask."""
        action_mask = ~self.visited
        action_mask[np.arange(self.n_traj), self.first] |= self.is_all_visited()
        return action_mask

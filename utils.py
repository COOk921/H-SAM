"""
Utility functions for data processing and graph construction.
"""

import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def single_batch_graph_data(data):
    """
    Build graph data from node features.
    
    Args:
        data: Node features of shape (num_nodes, dim)
        
    Returns:
        PyG Data object with nodes and edges
    """
    if isinstance(data, np.ndarray):
        node_features = torch.tensor(data, dtype=torch.float)
    else:
        node_features = data

    edge_index_list = []

    # Case 1: Columns 3,4,5 have same values, build edges by column 6 (ascending)
    selected_features_1 = node_features[:, 2:5]
    unique_features_1, inverse_indices_1 = torch.unique(
        selected_features_1, dim=0, return_inverse=True
    )
    
    for group_id in range(len(unique_features_1)):
        group_indices = torch.where(inverse_indices_1 == group_id)[0]
        num_nodes_in_group = len(group_indices)
        
        if num_nodes_in_group > 1:
            sorted_indices = group_indices[
                torch.argsort(node_features[group_indices, 5])
            ]
            source_nodes = sorted_indices[:-1]
            target_nodes = sorted_indices[1:]
            edge_index_list.append(
                torch.stack([source_nodes, target_nodes], dim=0)
            )

    # Case 2: Columns 3,4,6 have same values, build edges by column 5 (ascending)
    selected_features_2 = torch.cat(
        [node_features[:, 2:4], node_features[:, 5:6]], dim=1
    )
    unique_features_2, inverse_indices_2 = torch.unique(
        selected_features_2, dim=0, return_inverse=True
    )
    
    for group_id in range(len(unique_features_2)):
        group_indices = torch.where(inverse_indices_2 == group_id)[0]
        num_nodes_in_group = len(group_indices)
        
        if num_nodes_in_group > 1:
            sorted_indices = group_indices[
                torch.argsort(node_features[group_indices, 4])
            ]
            source_nodes = sorted_indices[:-1]
            target_nodes = sorted_indices[1:]
            edge_index_list.append(
                torch.stack([source_nodes, target_nodes], dim=0)
            )

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1).long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data_graph = Data(x=node_features, edge_index=edge_index)
    return data_graph


def save_merged_data(obs, resulting_traj, data_keys, valid_node):
    """
    Save prediction results for later merging.
    
    Args:
        obs: Observation dictionary
        resulting_traj: Resulting trajectory
        data_keys: Data keys for file naming
        valid_node: Number of valid nodes
    """
    observations = obs['observations']
    traj_data = resulting_traj[:, 0, :-1]
    traj_data = traj_data.reshape(observations.shape[0], observations.shape[1], 1)

    merged_data = np.concatenate([observations, traj_data], axis=-1)
    
    result_dir = "./result"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    columns = [
        'target', 'order', 'Unit Weight (kg)', 'Unit POD',
        'from_yard', 'from_bay', 'from_col', 'from_layer', 'pred'
    ]

    for i in range(merged_data.shape[0]):
        df = pd.DataFrame(merged_data[i], columns=columns)
        file_name = os.path.join(result_dir, f"{data_keys[i]}.csv")
        df.to_csv(file_name, index=False)

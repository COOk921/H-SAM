"""
异构图构建模块
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from config import FEATURES_FOR_GRAPH, KNN_K


def build_hetero_graph(df, feature_cols=None, device=None, knn_k=None):
    """
    构建异构图，包含三种类型的边：
    - blocks: 堆叠关系边 (同一堆垛中，上层指向下层)
    - spatial: 空间关系边 (同层不同排)
    - similar: 相似性边 (基于 KNN)
    
    Args:
        df: 预处理后的 DataFrame
        feature_cols: 用于图的特征列
        device: torch 设备
        knn_k: KNN 邻居数
    
    Returns:
        data: HeteroData 异构图对象
    """
    if feature_cols is None:
        feature_cols = FEATURES_FOR_GRAPH
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if knn_k is None:
        knn_k = KNN_K
        
    df = df.reset_index(drop=True)
    N = len(df)
    data = HeteroData()
    
    # 节点特征
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    data['container'].x = x
    
    # 1. blocks 边 (Same Stack, Top -> Below)
    groups = df.groupby(['from_yard', 'from_bay', 'from_col'])
    phy_src, phy_dst = [], []
    for _, group in groups:
        if len(group) < 2:
            continue
        sorted_indices = group.sort_values('from_layer', ascending=True).index.values
        phy_src.append(sorted_indices[:-1])
        phy_dst.append(sorted_indices[1:])
    
    if phy_src:
        edge_index_phy = torch.tensor(
            [np.concatenate(phy_src), np.concatenate(phy_dst)], 
            dtype=torch.long
        )
    else:
        edge_index_phy = torch.empty((2, 0), dtype=torch.long)
    data['container', 'blocks', 'container'].edge_index = edge_index_phy.to(device)

    # 2. spatial 边 (Same Yard/Bay/Layer, Different Col)
    layer_groups = df.groupby(['from_yard', 'from_bay', 'from_layer'])
    spa_src, spa_dst = [], []
    for _, group in layer_groups:
        if len(group) < 2:
            continue
        indices = group.index.values
        cols = group['from_col'].values
        num_nodes = len(indices)
        idx_i, idx_j = np.meshgrid(
            np.arange(num_nodes), np.arange(num_nodes), indexing='ij'
        )
        idx_i, idx_j = idx_i.flatten(), idx_j.flatten()
        mask = cols[idx_i] != cols[idx_j]
        spa_src.append(indices[idx_i[mask]])
        spa_dst.append(indices[idx_j[mask]])
        
    if spa_src:
        edge_index_spa = torch.tensor(
            [np.concatenate(spa_src), np.concatenate(spa_dst)], 
            dtype=torch.long
        )
    else:
        edge_index_spa = torch.empty((2, 0), dtype=torch.long)
    data['container', 'spatial', 'container'].edge_index = edge_index_spa.to(device)

    # 3. similar 边 (KNN)
    X_cpu = x.cpu().numpy()
    actual_k = min(knn_k, len(df) - 1)
    if actual_k > 0:
        knn = NearestNeighbors(n_neighbors=actual_k + 1).fit(X_cpu)
        _, neighbor_indices = knn.kneighbors(X_cpu)
        knn_src = np.repeat(np.arange(N), actual_k)
        knn_dst = neighbor_indices[:, 1:].flatten()
        edge_index_knn = torch.tensor([knn_src, knn_dst], dtype=torch.long)
    else:
        edge_index_knn = torch.empty((2, 0), dtype=torch.long)
    data['container', 'similar', 'container'].edge_index = edge_index_knn.to(device)

    return data

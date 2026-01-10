import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.utils import shuffle
from itertools import permutations
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import copy

from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors
from itertools import permutations

import pdb
from tqdm import tqdm


def read_data(file_path: str,continuous_features:list,categorical_features:list,other_features:list) -> dict:
    with open(file_path, 'rb') as f:
        data = pd.read_pickle(f)
   
    data = {tuple(key) if isinstance(key, np.ndarray) else key: value for key, value in data.items()}
    
    processed_data_local = {}
    for key, df in data.items():
        processed_df = pd.DataFrame(index=df.index)

        for col in other_features:
            processed_df[col] = df[col]

        local_scaler = StandardScaler()
        scaled_continuous = local_scaler.fit_transform(df[continuous_features])
        for i, col_name in enumerate(continuous_features):
            processed_df[col_name] = scaled_continuous[:, i]
        
        local_vocab_mappings = {}
        for col in categorical_features:
            unique_categories = df[col].unique()
            local_vocab_mappings[col] = {category: i + 1 for i, category in enumerate(unique_categories)}
            local_vocab_mappings[col]['[UNK]'] = 0 
            
            processed_df[col] = df[col].map(local_vocab_mappings[col]).fillna(0).astype(int)
    
        processed_data_local[key] = processed_df

    return processed_data_local


from torch_geometric.nn import HeteroConv, GATConv, Linear



def build_hetero_graph(df: pd.DataFrame, feature_cols: list, device: torch.device, knn_k: int = 3):
    """
    构建异构图 (Heterogeneous Graph)，逻辑分离三种关系的边。
    
    Args:
        df: 原始 DataFrame
        feature_cols: 用于节点特征的列名
        device:计算设备
        knn_k: KNN 建图时的邻居数量 (N)
    
    Returns:
        data: torch_geometric.data.HeteroData 对象
    """
    
    # [重要] 1. 确保 DataFrame 索引重置，保证 df 的 index 和 Tensor 的 0~N 索引对齐
    df = df.reset_index(drop=True)
    N = len(df)
    
    # 初始化异构图对象
    data = HeteroData()
    
    # -------------------------------------------------------
    # 1. 节点特征 (Node Features)
    # -------------------------------------------------------
    # 假设所有节点类型都是 'container'
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    data['container'].x = x
    
    # 准备 numpy 数组用于加速处理
    # 提取位置信息用于分组逻辑
    loc_info = df[['from_yard', 'from_bay', 'from_col', 'from_layer']].values
    
    # -------------------------------------------------------
    # 2. 构建边类型 A: 【物理约束边】 ('blocks')
    # 逻辑: Same Stack (Yard, Bay, Col), Layer N -> Layer N-1 (单向)
    # -------------------------------------------------------
    phy_src_list = []
    phy_dst_list = []
    
    # 按 Stack 分组 (Yard, Bay, Col)
    # 注意：为了效率，尽量减少循环内的 pandas 操作，但在逻辑复杂时 groupby 最稳妥
    groups = df.groupby(['from_yard', 'from_bay', 'from_col'])
    
    for _, group in groups:
        if len(group) < 2: continue
        
        # 按层数从高到低排序 (Layer N, Layer N-1, ...)
        sorted_group = group.sort_values('from_layer', ascending=True)
        indices = sorted_group.index.values # 获取全局索引 (0~N-1)
        
        # 建立链式边: Top -> Below
        # src: [Layer 3, Layer 2], dst: [Layer 2, Layer 1]
        src = indices[:-1]
        dst = indices[1:]
        
        phy_src_list.append(src)
        phy_dst_list.append(dst)
        
    if phy_src_list:
        phy_src = np.concatenate(phy_src_list)
        phy_dst = np.concatenate(phy_dst_list)
        edge_index_phy = torch.tensor([phy_src, phy_dst], dtype=torch.long)
    else:
        edge_index_phy = torch.empty((2, 0), dtype=torch.long)

    # 存入 HeteroData
    data['container', 'blocks', 'container'].edge_index = edge_index_phy.to(device)

    # -------------------------------------------------------
    # 3. 构建边类型 B: 【空间邻居边】 ('spatial')
    # 修改后逻辑: Same Yard, Same Bay, Same Layer, DIFFERENT Col (双向)
    # -------------------------------------------------------
    spa_src_list = []
    spa_dst_list = []
    
    # 这样每次循环只处理 "同一Yard、同一Bay、同一Layer" 的集装箱
    layer_groups = df.groupby(['from_yard', 'from_bay', 'from_layer'])
    
    for _, group in layer_groups:
        # 如果这一层只有一个或零个箱子，无法建立空间连接
        if len(group) < 2: continue
        
        # 获取该组内的全局索引和列号
        indices = group.index.values
        cols = group['from_col'].values
        
        # 使用广播机制生成组内两两组合
        num_nodes = len(indices)
        idx_i, idx_j = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes), indexing='ij')
        
        # 展平
        idx_i = idx_i.flatten()
        idx_j = idx_j.flatten()
        
        # 筛选: 列号必须不同 (cols[i] != cols[j])
        # 因为已经是同一层了，所以这里只需要判断列不同即可
        mask = cols[idx_i] != cols[idx_j]
        
        # 获取满足条件的全局索引
        valid_src = indices[idx_i[mask]]
        valid_dst = indices[idx_j[mask]]
        
        spa_src_list.append(valid_src)
        spa_dst_list.append(valid_dst)

    if spa_src_list:
        spa_src = np.concatenate(spa_src_list)
        spa_dst = np.concatenate(spa_dst_list)
        edge_index_spa = torch.tensor([spa_src, spa_dst], dtype=torch.long)
    else:
        edge_index_spa = torch.empty((2, 0), dtype=torch.long)

    # 存入 HeteroData
    data['container', 'spatial', 'container'].edge_index = edge_index_spa.to(device)

    # -------------------------------------------------------
    # 4. 构建边类型 C: 【相似性边】 ('similar')
    # 逻辑: KNN, 为每个箱子选择特征最近的 N 个邻居
    # -------------------------------------------------------
    # 使用 sklearn 的 NearestNeighbors
    # 注意: 需要将 tensor 转回 numpy (cpu) 进行计算
    X_cpu = x.cpu().numpy()
    n_samples = X_cpu.shape[0]
    if n_samples < knn_k + 1:
        knn_k = n_samples - 1 
    
    # n_neighbors = k + 1，因为最近的邻居通常是它自己（距离为0），我们需要排除自己
    knn = NearestNeighbors(n_neighbors=knn_k + 1, algorithm='auto')
    knn.fit(X_cpu)
    
    # 寻找邻居: distances, indices (N, k+1)
    _, neighbor_indices = knn.kneighbors(X_cpu)
    
    # 构建边列表
    # neighbor_indices[:, 0] 是节点自己，[:, 1:] 是真实的 k 个邻居
    knn_src = []
    knn_dst = []
    
    src_indices = np.arange(N) # 0 到 N-1
    real_neighbors = neighbor_indices[:, 1:] # 去掉自己
    
    # 将 src (N, 1) 广播配合 neighbors (N, k)
    # src_repeated: [0,0,0, 1,1,1, ...]
    knn_src = np.repeat(src_indices, knn_k) 
    # dst_flattened: [n0_0, n0_1, ..., n1_0, ...]
    knn_dst = real_neighbors.flatten()
    
    edge_index_knn = torch.tensor([knn_src, knn_dst], dtype=torch.long)
    
    # 存入 HeteroData
    data['container', 'similar', 'container'].edge_index = edge_index_knn.to(device)

    # -------------------------------------------------------
    # 5. 打印统计信息 (可选)
    # -------------------------------------------------------
    # print(f"图构建完成:")
    # print(f"  - 节点数: {N}")
    # print(f"  - 物理边 (blocks): {edge_index_phy.shape[1]}")
    # print(f"  - 空间边 (spatial): {edge_index_spa.shape[1]}")
    # print(f"  - 相似边 (similar): {edge_index_knn.shape[1]}")

    return data


if __name__ == '__main__':
    continuous_features = ['Unit Weight (kg)']
    categorical_features = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer', ]
    other_features = ['order', 'Unit Nbr','Time Completed']
    
    FEATURES_FOR_MODEL = ['Unit Weight (kg)','Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
    FEATURES_FOR_GRAPH = ['Unit Weight (kg)','Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
    
    READ_PATH = "./data/container_data_cluster.pkl"
    WRITE_PATH = "./data/processed_container_data_hetero.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    training_data = read_data(READ_PATH, continuous_features, categorical_features,other_features)

    processed_data_local = {}
    for key, df in tqdm(training_data.items(), desc="Processing training data"):
        graph = build_hetero_graph(df,FEATURES_FOR_GRAPH,device)
       
        processed_data_local[key] = {
            'data': df, 
            'graph': graph 
        }

    with open(WRITE_PATH, 'wb') as f:
        pickle.dump(processed_data_local, f)

        








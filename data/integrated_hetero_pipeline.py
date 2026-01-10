import pandas as pd
import numpy as np
import pickle
import os
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData
from tqdm import tqdm

"""
整合流水线：
1. 读取 ships/ 目录下的 CSV。
2. 进行简单的 KMeans 聚类并增加 'cluster' 标签。
3. 按 'Unit O/B Actual Visit' 和 'cluster' 进行嵌套切分。
4. 对每个切分后的组进行特征预处理和异构图构建。
5. 保存最终的集成 pkl。
"""

# --- 配置参数 ---
INPUT_DIR = './ships/'
OUTPUT_FILE = './processed_container_data_hetero.pkl'

CONTINUOUS_FEATURES = ['Unit Weight (kg)']
CATEGORICAL_FEATURES = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
OTHER_FEATURES = ['order', 'Unit Nbr', 'Time Completed']  # 确保这些列在 CSV 中存在
FEATURES_FOR_GRAPH = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

def get_cluster_labels(df):
    """逻辑来自原 integrated_cluster_and_split.py: 为 DF 添加聚类标签"""
    numeric_cols = ['Unit Weight (kg)']
    categorical_cols = ['from_yard', 'from_bay', 'from_col', 'from_layer', 'Unit POD']
    
    # 预处理用于聚类的特征
    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_cols])
    
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_cols])
    
    processed_data = np.hstack([scaled_numeric, encoded_categorical])
    
    # K-Means
    n_clusters = max(1, int(len(df) / 50))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    return kmeans.fit_predict(processed_data)

def preprocess_group(df, continuous_features, categorical_features, other_features):
    """逻辑来自原 hetero_graph.py 中的 read_data 处理部分"""
    processed_df = pd.DataFrame(index=df.index)
    
    for col in other_features:
        if col in df.columns:
            processed_df[col] = df[col]
            
    # 数值型标准化
    local_scaler = StandardScaler()
    scaled_continuous = local_scaler.fit_transform(df[continuous_features])
    for i, col_name in enumerate(continuous_features):
        processed_df[col_name] = scaled_continuous[:, i]
        
    # 类别型映射
    for col in categorical_features:
        unique_categories = df[col].unique()
        mapping = {cat: i + 1 for i, cat in enumerate(unique_categories)}
        mapping['[UNK]'] = 0
        processed_df[col] = df[col].map(mapping).fillna(0).astype(int)
        
    return processed_df

def build_hetero_graph(df, feature_cols, device, knn_k=3):
    """逻辑来自原 hetero_graph.py: 构建异构图"""
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
        if len(group) < 2: continue
        sorted_indices = group.sort_values('from_layer', ascending=True).index.values
        phy_src.append(sorted_indices[:-1])
        phy_dst.append(sorted_indices[1:])
    
    if phy_src:
        edge_index_phy = torch.tensor([np.concatenate(phy_src), np.concatenate(phy_dst)], dtype=torch.long)
    else:
        edge_index_phy = torch.empty((2, 0), dtype=torch.long)
    data['container', 'blocks', 'container'].edge_index = edge_index_phy.to(device)

    # 2. spatial 边 (Same Yard/Bay/Layer, Different Col)
    layer_groups = df.groupby(['from_yard', 'from_bay', 'from_layer'])
    spa_src, spa_dst = [], []
    for _, group in layer_groups:
        if len(group) < 2: continue
        indices = group.index.values
        cols = group['from_col'].values
        num_nodes = len(indices)
        idx_i, idx_j = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes), indexing='ij')
        idx_i, idx_j = idx_i.flatten(), idx_j.flatten()
        mask = cols[idx_i] != cols[idx_j]
        spa_src.append(indices[idx_i[mask]])
        spa_dst.append(indices[idx_j[mask]])
        
    if spa_src:
        edge_index_spa = torch.tensor([np.concatenate(spa_src), np.concatenate(spa_dst)], dtype=torch.long)
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    all_dfs = []
    print(f"步骤 1/3: 正在读取和聚类 {len(csv_files)} 个文件...")
    for f in tqdm(csv_files):
        path = os.path.join(INPUT_DIR, f)
        df = pd.read_csv(path)
        # 如果原始 CSV 没有 order，可以手动添加一个，防止后续报错
        if 'order' not in df.columns:
            df['order'] = range(len(df))
        df['cluster'] = get_cluster_labels(df)
        all_dfs.append(df)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print("步骤 2/3: 正在进行数据切分...")
    grouped = combined_df.groupby(['Unit O/B Actual Visit', 'cluster'])
    
    final_results = {}
    print(f"步骤 3/3: 正在预处理每个组并构建异构图 (总计 {len(grouped)} 组)...")
    for key, group_df in tqdm(grouped):
        # 保持与 hetero_graph.py 一致的 key 格式（元组）
        group_key = tuple(key) if isinstance(key, (list, np.ndarray)) else key
        
        # 1. 预处理 (标准化/映射)
        preprocessed_df = preprocess_group(group_df, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, OTHER_FEATURES)
        
        # 2. 构建图
        graph = build_hetero_graph(preprocessed_df, FEATURES_FOR_GRAPH, device)
        
        final_results[group_key] = {
            'data': preprocessed_df,
            'graph': graph
        }
        
    print(f"正在保存最终结果到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_results, f)
    print("完成！")

if __name__ == "__main__":
    main()
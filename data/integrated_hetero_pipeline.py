import pandas as pd
import numpy as np
import pickle
import os
import json
import torch
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from torch_geometric.data import HeteroData
from tqdm import tqdm

"""
整合流水线：
1. 读取 ships/ 目录下的 CSV。
2. 使用基于自定义亲和矩阵的谱聚类添加 'cluster' 标签。
3. 按 'Unit O/B Actual Visit' 和 'cluster' 进行嵌套切分。
4. 对每个切分后的组进行特征预处理和异构图构建。
5. 保存最终的集成 pkl。
"""

# --- 配置参数 ---
INPUT_DIR = './ships/'
OUTPUT_FILE = './processed_container_data_hetero(Spectral_opt).pkl'
OUTPUT_STATS_FILE = './pipeline_statistics.json'

CONTINUOUS_FEATURES = ['Unit Weight (kg)']
CATEGORICAL_FEATURES = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
OTHER_FEATURES = ['order', 'Unit Nbr', 'Time Completed']  # 确保这些列在 CSV 中存在
FEATURES_FOR_GRAPH = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# --- 谱聚类超参数 ---
ALPHA = 0.5  # 位置相似度权重
BETA = 0.5   # 重量相似度权重


def compute_median_heuristic_sigma(dist_sq_matrix):
    """
    基于中位数启发式计算高斯核带宽。
    
    原理：取所有点对距离的中位数作为 sigma，保证大约一半的点对
    具有较高的相似度 (sim > exp(-0.5) ≈ 0.6)，从而使图连通性恰到好处。
    
    Args:
        dist_sq_matrix: 距离平方矩阵 (N x N)
    
    Returns:
        sigma: 高斯核带宽参数
    """
    # 提取上三角部分（不含对角线），避免重复计算和自身距离
    upper_tri_indices = np.triu_indices_from(dist_sq_matrix, k=1)
    distances = np.sqrt(dist_sq_matrix[upper_tri_indices])
    
    # 取中位数作为 sigma
    median_dist = np.median(distances)
    
    # 防止 sigma 为 0（当所有点相同时）
    sigma = median_dist if median_dist > 1e-8 else 1.0
    
    return sigma


def compute_affinity_matrix(df, alpha=ALPHA, beta=BETA):
    """
    计算自定义亲和矩阵：
    A_ij = sim_p(i, j) * [α * sim_l(i, j) + β * sim_w(i, j)]
    
    - sim_p: 航次相似度 (相同航次为1，否则为0)
    - sim_w: 重量相似度 (高斯核，带宽由中位数启发式自动确定)
    - sim_l: 存储位置相似度 (高斯核，带宽由中位数启发式自动确定)
    """
    N = len(df)
    
    # --- 1. 计算 sim_p (航次相似度) ---
    voyage = df['Unit O/B Actual Visit'].values
    sim_p = (voyage[:, None] == voyage[None, :]).astype(np.float32)
    
    # --- 2. 计算 sim_w (重量相似度) ---
    weight = df['Unit Weight (kg)'].values.reshape(-1, 1)
    # 标准化重量以获得更稳定的高斯核计算
    weight_scaler = StandardScaler()
    weight_scaled = weight_scaler.fit_transform(weight)
    # 计算欧氏距离的平方
    weight_dist_sq = cdist(weight_scaled, weight_scaled, metric='sqeuclidean')
    # 使用中位数启发式计算 sigma_w
    sigma_w = compute_median_heuristic_sigma(weight_dist_sq)
    sim_w = np.exp(-weight_dist_sq / (2 * sigma_w ** 2))
    
    # --- 3. 计算 sim_l (存储位置相似度) ---
    location_cols = ['from_yard', 'from_bay', 'from_col', 'from_layer']
    location_data = df[location_cols].copy()
    
    # 对类别型位置特征进行 LabelEncoder 编码
    for col in location_cols:
        le = LabelEncoder()
        location_data[col] = le.fit_transform(location_data[col].astype(str))
    
    location_values = location_data.values.astype(np.float32)
    # 标准化位置特征
    location_scaler = StandardScaler()
    location_scaled = location_scaler.fit_transform(location_values)
    # 计算欧氏距离的平方
    location_dist_sq = cdist(location_scaled, location_scaled, metric='sqeuclidean')
    # 使用中位数启发式计算 sigma_l
    sigma_l = compute_median_heuristic_sigma(location_dist_sq)
    sim_l = np.exp(-location_dist_sq / (2 * sigma_l ** 2))
    
    # --- 4. 组合亲和矩阵 ---
    affinity = sim_p * (alpha * sim_l + beta * sim_w)
    
    # 确保对角线为0（避免自环影响）
    np.fill_diagonal(affinity, 0)
    
    return affinity


def get_cluster_labels(df, n_clusters=None):
    """使用谱聚类为 DF 添加聚类标签"""
    N = len(df)
    
    # 动态确定聚类数量
    if n_clusters is None:
        n_clusters = max(1, int(N / 50))
    
    # 边界情况处理
    if N <= n_clusters:
        return np.arange(N)
    
    # 计算自定义亲和矩阵
    affinity_matrix = compute_affinity_matrix(df)
    
    # 使用预计算的亲和矩阵进行谱聚类
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize'
    )
    
    return spectral.fit_predict(affinity_matrix)

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
    
    # 初始化统计信息字典
    statistics = {
        'clustering_stats': [],
        'graph_stats': [],
        'summary': {}
    }
    
    all_dfs = []
    print(f"步骤 1/3: 正在读取和聚类 {len(csv_files)} 个文件...")
    for f in tqdm(csv_files):
        path = os.path.join(INPUT_DIR, f)
        df = pd.read_csv(path)
        # 如果原始 CSV 没有 order，可以手动添加一个，防止后续报错
        if 'order' not in df.columns:
            df['order'] = range(len(df))
        df['cluster'] = get_cluster_labels(df)
        
        # 记录聚类统计信息
        cluster_counts = df['cluster'].value_counts()
        cluster_stats = {
            'csv_filename': f,
            'total_samples': int(len(df)),
            'num_clusters': int(len(cluster_counts)),
            'samples_per_cluster_mean': float(cluster_counts.mean()),
            'samples_per_cluster_median': float(cluster_counts.median()),
            'samples_per_cluster_min': int(cluster_counts.min()),
            'samples_per_cluster_max': int(cluster_counts.max()),
            'samples_per_cluster_std': float(cluster_counts.std())
        }
        statistics['clustering_stats'].append(cluster_stats)
        
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
        
        # 记录图统计信息
        graph_stats = {
            'group_key': str(group_key),  # 转换为字符串以便JSON序列化
            'voyage': str(key[0]) if isinstance(key, (tuple, list)) else str(key),
            'cluster_id': int(key[1]) if isinstance(key, (tuple, list)) and len(key) > 1 else None,
            'num_nodes': int(graph['container'].x.shape[0]),
            'num_edges_blocks': int(graph['container', 'blocks', 'container'].edge_index.shape[1]),
            'num_edges_spatial': int(graph['container', 'spatial', 'container'].edge_index.shape[1]),
            'num_edges_similar': int(graph['container', 'similar', 'container'].edge_index.shape[1]),
            'total_edges': int(
                graph['container', 'blocks', 'container'].edge_index.shape[1] +
                graph['container', 'spatial', 'container'].edge_index.shape[1] +
                graph['container', 'similar', 'container'].edge_index.shape[1]
            ),
            'num_features': int(graph['container'].x.shape[1])
        }
        statistics['graph_stats'].append(graph_stats)
        
        final_results[group_key] = {
            'data': preprocessed_df,
            'graph': graph
        }
    
    # 添加汇总统计
    statistics['summary'] = {
        'total_csv_files': int(len(csv_files)),
        'total_graphs': int(len(final_results)),
        'total_samples': int(len(combined_df)),
        'avg_clusters_per_csv': float(np.mean([s['num_clusters'] for s in statistics['clustering_stats']])),
        'avg_nodes_per_graph': float(np.mean([s['num_nodes'] for s in statistics['graph_stats']])),
        'avg_edges_per_graph': float(np.mean([s['total_edges'] for s in statistics['graph_stats']])),
        'total_nodes_all_graphs': int(sum([s['num_nodes'] for s in statistics['graph_stats']])),
        'total_edges_all_graphs': int(sum([s['total_edges'] for s in statistics['graph_stats']]))
    }
        
    print(f"正在保存最终结果到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"正在保存统计信息到 {OUTPUT_STATS_FILE}...")
    with open(OUTPUT_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print("完成！")
    print(f"\n统计摘要:")
    print(f"  - 处理的CSV文件数: {statistics['summary']['total_csv_files']}")
    print(f"  - 生成的异构图数: {statistics['summary']['total_graphs']}")
    print(f"  - 总样本数: {statistics['summary']['total_samples']}")
    print(f"  - 平均每个CSV的聚类数: {statistics['summary']['avg_clusters_per_csv']:.2f}")
    print(f"  - 平均每个图的节点数: {statistics['summary']['avg_nodes_per_graph']:.2f}")
    print(f"  - 平均每个图的边数: {statistics['summary']['avg_edges_per_graph']:.2f}")

if __name__ == "__main__":
    main()
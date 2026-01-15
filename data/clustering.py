"""
聚类模块：支持谱聚类和 K-means 两种算法
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from scipy.spatial.distance import cdist

from config import (
    ALPHA, BETA, CLUSTERING_METHOD,
    KMEANS_MAX_ITER, KMEANS_N_INIT
)


# ============================================================================
# 特征提取
# ============================================================================

def get_clustering_features(df):
    """
    获取用于聚类和可视化的特征矩阵
    
    Args:
        df: 包含集装箱数据的 DataFrame
    
    Returns:
        features: 标准化后的特征矩阵 (N x 5)
    """
    # 重量特征
    weight = df['Unit Weight (kg)'].values.reshape(-1, 1)
    weight_scaler = StandardScaler()
    weight_scaled = weight_scaler.fit_transform(weight)
    
    
    # 位置特征
    location_cols = ['from_yard', 'from_bay', 'from_col', 'from_layer','Unit POD']
    
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[location_cols])
    
    # 合并特征
    features = np.hstack([weight_scaled, encoded_categorical])
    
    return features


# ============================================================================
# 谱聚类相关
# ============================================================================

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
    upper_tri_indices = np.triu_indices_from(dist_sq_matrix, k=1)
    distances = np.sqrt(dist_sq_matrix[upper_tri_indices])
    median_dist = np.median(distances)
    sigma = median_dist if median_dist > 1e-8 else 1.0
    return sigma


def compute_affinity_matrix(df, alpha=ALPHA, beta=BETA):
    """
    计算自定义亲和矩阵（用于谱聚类）：
    A_ij = sim_p(i, j) * [α * sim_l(i, j) + β * sim_w(i, j)]
    
    - sim_p: 航次相似度 (相同航次为1，否则为0)
    - sim_w: 重量相似度 (高斯核，带宽由中位数启发式自动确定)
    - sim_l: 存储位置相似度 (高斯核，带宽由中位数启发式自动确定)
    
    Args:
        df: 包含集装箱数据的 DataFrame
        alpha: 位置相似度权重
        beta: 重量相似度权重
    
    Returns:
        affinity: 亲和矩阵 (N x N)
    """
    N = len(df)
    
    # --- 1. 计算 sim_p (航次相似度) ---
    pod = df['Unit POD'].values
    sim_p = ((pod[:, None] == pod[None, :])).astype(np.float32)
    
    # --- 2. 计算 sim_w (重量相似度) ---
    weight = df['Unit Weight (kg)'].values.reshape(-1, 1)
    weight_scaler = StandardScaler()
    weight_scaled = weight_scaler.fit_transform(weight)
    weight_dist_sq = cdist(weight_scaled, weight_scaled, metric='sqeuclidean')
    sigma_w = compute_median_heuristic_sigma(weight_dist_sq)
    sim_w = np.exp(-weight_dist_sq / (2 * sigma_w ** 2))
    
    # --- 3. 计算 sim_l (存储位置相似度) ---
    location_cols = ['from_yard', 'from_bay', 'from_col', 'from_layer','Unit POD']
    location_data = df[location_cols].copy()
    
    for col in location_cols:
        le = LabelEncoder()
        location_data[col] = le.fit_transform(location_data[col].astype(str))
    
    location_values = location_data.values.astype(np.float32)
    location_scaler = StandardScaler()
    location_scaled = location_scaler.fit_transform(location_values)
    location_dist_sq = cdist(location_scaled, location_scaled, metric='sqeuclidean')
    sigma_l = compute_median_heuristic_sigma(location_dist_sq)
    sim_l = np.exp(-location_dist_sq / (2 * sigma_l ** 2))
    
    # --- 4. 组合亲和矩阵 ---
    #affinity = sim_p * (alpha * sim_l + beta * sim_w)
    affinity =  (alpha * sim_l + beta * sim_w)
    np.fill_diagonal(affinity, 0)
    
    return affinity


def _spectral_clustering(df, n_clusters):
    """
    使用谱聚类进行聚类
    
    Args:
        df: 包含集装箱数据的 DataFrame
        n_clusters: 聚类数量
    
    Returns:
        labels: 聚类标签数组
    """
    affinity_matrix = compute_affinity_matrix(df)
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize',
        n_jobs=-1
    )
    
    return spectral.fit_predict(affinity_matrix)


# ============================================================================
# K-means 聚类相关
# ============================================================================

def _kmeans_clustering(df, n_clusters):
    """
    使用 K-means 进行聚类
    
    Args:
        df: 包含集装箱数据的 DataFrame
        n_clusters: 聚类数量
    
    Returns:
        labels: 聚类标签数组
    """
    # 获取特征矩阵
    features = get_clustering_features(df)
    # K-means 聚类
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        max_iter=KMEANS_MAX_ITER,
        n_init=KMEANS_N_INIT
    )
    return kmeans.fit_predict(features)


# ============================================================================
# 统一聚类接口
# ============================================================================

def get_cluster_labels(df, n_clusters=None, method=None):
    """
    统一的聚类接口，根据配置选择聚类算法
    
    Args:
        df: 包含集装箱数据的 DataFrame
        n_clusters: 聚类数量，None 时自动确定
        method: 聚类方法 ('spectral' 或 'kmeans')，None 时使用配置文件设置
    
    Returns:
        labels: 聚类标签数组
    """
    N = len(df)
    
    # 使用配置文件中的方法，除非显式指定
    if method is None:
        method = CLUSTERING_METHOD
    
    # 验证方法名称
    valid_methods = ['spectral', 'kmeans']
    if method not in valid_methods:
        raise ValueError(f"不支持的聚类方法: {method}. 可选: {valid_methods}")
    
    # 动态确定聚类数量
    if n_clusters is None:
        n_clusters = max(1, int(N / 50))
    
    # 边界情况处理
    if N <= n_clusters:
        return np.arange(N)
    
    # 根据方法选择聚类算法
    if method == 'spectral':
        return _spectral_clustering(df, n_clusters)
    else:  # kmeans
        return _kmeans_clustering(df, n_clusters)


def get_method_name(method=None):
    """
    获取聚类方法的显示名称
    
    Args:
        method: 聚类方法，None 时使用配置文件设置
    
    Returns:
        name: 方法的中文名称
    """
    if method is None:
        method = CLUSTERING_METHOD
    
    names = {
        'spectral': '谱聚类 (Spectral Clustering)',
        'kmeans': 'K-means'
    }
    return names.get(method, method)
